import numpy as np
import gpflow
import tensorflow as tf
import json
import time
import os
import csv
from pendulum_sim_grok import PendulumSim

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpflow.config.set_default_float(tf.float32)

class PendulumGPTester:
    def __init__(self, num_links=1, sim_time=10.0, dt=0.0125):
        self.num_links = num_links
        self.state_dim = 1 + 2 + 3 * num_links  # time, x, x_dot, cosθ, sinθ, θ_dot
        self.output_dim = 2 + 3 * num_links    # x, x_dot, cosθ, sinθ, θ_dot
        self.sim_time = sim_time
        self.dt = dt
        self.steps = int(sim_time / dt) + 1    # 801 for 10s at 0.0125
        self.policy_params = self._generate_random_policy()
        self.model = None
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        self.mask = None

    def _generate_random_policy(self):
        a = np.random.uniform(-5, 5, (3, self.state_dim-1))
        b = np.random.uniform(-0.5, 0.5, (3, self.state_dim-1, self.state_dim-1))
        c = np.random.uniform(-0.05, 0.05, (3, self.state_dim-1, self.state_dim-1, self.state_dim-1))
        return (tf.constant(a, dtype=tf.float32), 
                tf.constant(b, dtype=tf.float32), 
                tf.constant(c, dtype=tf.float32))

    @tf.function
    def predict_action(self, states):
        a, b, c = self.policy_params
        p = states  # Shape: [N, state_dim-1]
        p2 = tf.einsum('bi,bj->bij', p, p)
        p3 = tf.einsum('bij,bk->bijk', p2, p)
        term1 = tf.reduce_sum(tf.maximum(0., a[0, :] * p + a[1, :]) * a[2, :], axis=1)
        term2 = tf.reduce_sum(tf.maximum(0., b[0, :, :] * p2 + b[1, :, :]) * b[2, :, :], axis=[1, 2])
        term3 = tf.reduce_sum(tf.maximum(0., c[0, :, :, :] * p3 + c[1, :, :, :]) * c[2, :, :, :], axis=[1, 2, 3])
        actions = term1 + term2 + term3
        return tf.clip_by_value(actions, -5.0, 5.0)

    def run_simulation(self, initial_state, idx):
        policy_file = f'policies/policy_ind_{idx}.json'
        sim_done_file = f'policies/sim_done_ind_{idx}.txt'
        sim_data_file = f'policies/sim_data_ind_{idx}.json'
        os.makedirs('policies', exist_ok=True)

        policy = {'type': 'cubic', 'params': {
            'a': self.policy_params[0].numpy().tolist(),
            'b': self.policy_params[1].numpy().tolist(),
            'c': self.policy_params[2].numpy().tolist()
        }}
        with open(policy_file, 'w') as f:
            json.dump(policy, f)

        sim = PendulumSim(lengths=[0.5] * self.num_links, masses=[1.0] * self.num_links)
        start_time = time.time()
        sim.simulate(t_span=(0, self.sim_time/3), initial_state=initial_state, dt=self.dt, 
                     sim_data_file=sim_data_file, sim_done_file=sim_done_file)
        while not os.path.exists(sim_done_file):
            pass
        os.remove(sim_done_file)
        print(f"Simulation {idx} completed in {time.time() - start_time:.2f}s")

        with open(sim_data_file, 'r') as f:
            data = json.load(f)
        states = np.array(data['states'], dtype=np.float32)  # Shape: (801, 7)
        actions = np.array(data['actions'], dtype=np.float32)  # Shape: (801,)
        delta_T = actions * 0 + self.dt
        X = np.hstack([delta_T[:-1, None], np.hstack([states[:-1, 1:], actions[:-1, None]])])  # Shape: (800, 8)
        y = states[1:, 1:]  # Shape: (800, 5)
        return X, y

    def generate_random_initial_state(self):
        x = np.random.uniform(-1.0, 1.0)
        x_dot = np.random.uniform(-1.0, 1.0)
        theta = np.random.uniform(0, 2 * np.pi)
        theta_dot = np.random.uniform(-1.0, 1.0)
        return np.array([self.dt, x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    def fit_gp(self, X_data, y_data):
        subset_size = min(5000, X_data.shape[0])
        indices = np.random.choice(X_data.shape[0], subset_size, replace=False)
        X_subset = X_data[indices].astype(np.float32)
        y_subset = y_data[indices].astype(np.float32)

        self.X_mean = np.mean(X_subset, axis=0)
        self.X_std = np.std(X_subset, axis=0) + 1e-6
        self.mask = self.X_std > 1e-4
        X_subset_clean = X_subset[:, self.mask]
        self.X_mean = self.X_mean[self.mask]
        self.X_std = self.X_std[self.mask]
        X_normalized = (X_subset_clean - self.X_mean) / self.X_std

        self.y_mean = np.mean(y_subset, axis=0)
        self.y_std = np.std(y_subset, axis=0) + 1e-6
        y_normalized = (y_subset - self.y_mean) / self.y_std

        kernel = gpflow.kernels.SquaredExponential(lengthscales=1.0)
        self.model = gpflow.models.GPR(
            data=(X_normalized, y_normalized),
            kernel=kernel
        )
        self.model.likelihood.variance.assign(1.0)
        self.model.likelihood.variance = gpflow.Parameter(
            self.model.likelihood.variance, transform=gpflow.utilities.positive(lower=1e-3)
        )

        start_time = time.time()
        optimizer = gpflow.optimizers.Scipy()
        optimizer.minimize(self.model.training_loss, self.model.trainable_variables, options={'maxiter': 500})
        print(f"GPR fitted in {time.time() - start_time:.2f}s")

    @tf.function
    def predict_trajectory(self, initial_state):
        states = tf.TensorArray(dtype=tf.float32, size=self.steps)
        states = states.write(0, initial_state)
        time_steps = tf.range(self.steps, dtype=tf.float32) * self.dt

        def body(i, prev_state, states_ta):
            action = self.predict_action(prev_state[None, 1:])[0]
            X = tf.concat([prev_state, action[tf.newaxis]], axis=0)[self.mask]
            X_normalized = (X - self.X_mean) / self.X_std
            X_tf = tf.expand_dims(X_normalized, 0)
            mean = self.model.predict_f(X_tf)[0][0]
            next_state = mean * self.y_std + self.y_mean
            state = tf.concat([time_steps[i:i+1], next_state], axis=0)
            x = state[1]
            x_dot = state[2]
            state = tf.where(tf.abs(x) > 3, tf.concat([state[:2], [0.0], state[3:]], 0), state)
            state = tf.where(tf.abs(x) > 3, tf.concat([state[:1], [tf.clip_by_value(x, -3, 3)], state[2:]], 0), state)
            state = tf.ensure_shape(state, [6])
            states_ta = states_ta.write(i, state)
            return i + 1, state, states_ta

        _, _, final_states = tf.while_loop(
            cond=lambda i, *_: i < self.steps,
            body=body,
            loop_vars=(tf.constant(1), initial_state, states),
            shape_invariants=(tf.TensorShape([]), tf.TensorShape([6]), None)
        )

        return final_states.stack()

def main():
    tester = PendulumGPTester(num_links=1, sim_time=10.0, dt=0.0125)
    X_data = []
    y_data = []
    for i in range(20):
        initial_state = tester.generate_random_initial_state()
        print(f"Simulating with initial state: {initial_state}")
        X, y = tester.run_simulation(initial_state, i)
        X_data.append(X)
        y_data.append(y)
    X_data = np.vstack(X_data)  # Shape: (16000, 8)
    y_data = np.vstack(y_data)  # Shape: (16000, 5)
    print(f"Full data shapes - X: {X_data.shape}, y: {y_data.shape}")

    tester.fit_gp(X_data, y_data)
    new_initial_state = tester.generate_random_initial_state()
    print(f"New initial state: {new_initial_state}")
    start_time = time.time()
    trajectory = tester.predict_trajectory(new_initial_state)
    print(f"Prediction took {time.time() - start_time:.2f}s")

    with open('policies/predicted_trajectory.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'x', 'x_dot', 'angle_cos', 'angle_sin', 'theta_dot'])
        for state in trajectory:
            writer.writerow(state.numpy())

if __name__ == "__main__":
    main()