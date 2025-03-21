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
    def __init__(self, num_links=1, sim_time=10.0, dt=0.05):
        self.num_links = num_links
        self.state_dim = 1 + 2 + 3 * num_links  # time, x, x_dot, cosθ, sinθ, θ_dot
        self.output_dim = 2 + 3 * num_links    # diffs: x, x_dot, cosθ, sinθ, θ_dot
        self.sim_time = sim_time
        self.dt = dt
        self.steps = int(sim_time / dt) + 1  # 201 for 10s at 0.05

        # Random policy (cubic, fixed)
        self.policy_params = self._generate_random_policy()

    def _generate_random_policy(self):
        a = np.random.uniform(-5, 5, (3, self.state_dim))
        b = np.random.uniform(-0.5, 0.5, (3, self.state_dim, self.state_dim))
        c = np.random.uniform(-0.05, 0.05, (3, self.state_dim, self.state_dim, self.state_dim))
        return (a, b, c)

    def predict_action(self, states):
        a, b, c = self.policy_params
        p = states
        p2 = np.einsum('bi,bj->bij', p, p)
        p3 = np.einsum('bij,bk->bijk', p2, p)
        term1 = np.sum(np.maximum(0, a[0, :] * p + a[1, :]) * a[2, :], axis=1)
        term2 = np.sum(np.maximum(0, b[0, :, :] * p2 + b[1, :, :]) * b[2, :, :], axis=(1, 2))
        term3 = np.sum(np.maximum(0, c[0, :, :, :] * p3 + c[1, :, :, :]) * c[2, :, :, :], axis=(1, 2, 3))
        actions = term1 + term2 + term3
        return np.clip(actions, -5.0, 5.0)

    def run_simulation(self, initial_state, idx):
        policy_file = f'policies/policy_ind_{idx}.json'
        sim_done_file = f'policies/sim_done_ind_{idx}.txt'
        sim_data_file = f'policies/sim_data_ind_{idx}.json'
        os.makedirs('policies', exist_ok=True)

        policy = {'type': 'cubic', 'params': {
            'a': self.policy_params[0].tolist(),
            'b': self.policy_params[1].tolist(),
            'c': self.policy_params[2].tolist()
        }}
        with open(policy_file, 'w') as f:
            json.dump(policy, f)

        sim = PendulumSim(lengths=[0.5] * self.num_links, masses=[1.0] * self.num_links)
        if os.path.exists(sim_done_file):
            os.remove(sim_done_file)
        start_time = time.time()
        sim.simulate(t_span=(0, self.sim_time), initial_state=initial_state, dt=self.dt, 
                     sim_data_file=sim_data_file, sim_done_file=sim_done_file)
        print(f"Simulation {idx} completed in {time.time() - start_time:.2f}s")

        with open(sim_data_file, 'r') as f:
            data = json.load(f)
        states = np.array(data['states'], dtype=np.float32)  # Shape: (201, 7)
        actions = np.array(data['actions'], dtype=np.float32)  # Shape: (201,)
        X = np.hstack([states[:-1], actions[:-1, None]])  # Shape: (200, 8)
        y = states[1:, 1:] - states[:-1, 1:]  # Shape: (200, 5)
        return X, y

    def generate_random_initial_state(self):
        x = np.random.uniform(-1.0, 1.0)
        x_dot = np.random.uniform(-1.0, 1.0)
        theta = np.random.uniform(0, 2 * np.pi)
        theta_dot = np.random.uniform(-1.0, 1.0)
        return np.array([self.dt, x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    def fit_gp(self, X_data, y_data):
        subset_size = 1000
        indices = np.random.choice(X_data.shape[0], subset_size, replace=False)
        X_subset = X_data[indices]
        y_subset = y_data[indices]
        print(f"Subset shapes - X: {X_subset.shape}, y: {y_subset.shape}")
        print(f"y_subset mean: {np.mean(y_subset, axis=0)}, std: {np.std(y_subset, axis=0)}")

        self.X_mean = np.mean(X_subset, axis=0)
        self.X_std = np.std(X_subset, axis=0) + 1e-6
        self.y_mean = np.mean(y_subset, axis=0)
        self.y_std = np.std(y_subset, axis=0) + 1e-6
        X_normalized = (X_subset - self.X_mean) / self.X_std
        y_normalized = (y_subset - self.y_mean) / self.y_std

        rbf = gpflow.kernels.SquaredExponential(lengthscales=0.5 * tf.ones(self.state_dim + 1, dtype=tf.float32))  # 8 features
        matern = gpflow.kernels.Matern52(lengthscales=0.5 * tf.ones(self.state_dim + 1, dtype=tf.float32))
        white = gpflow.kernels.White(variance=0.01)
        kernel = rbf + matern + white
        self.model = gpflow.models.GPR(
            data=(tf.constant(X_normalized, dtype=tf.float32), tf.constant(y_normalized, dtype=tf.float32)),
            kernel=kernel
        )

        # Custom batch loss with variance floor
        def batch_loss(X_batch, y_batch):
            mean, var = self.model.predict_f(X_batch, full_cov=False)
            var = tf.maximum(var, 1e-6)
            residual = y_batch - mean
            log_prob = -0.5 * (tf.math.log(2 * np.pi) + tf.math.log(var) + tf.square(residual) / var)
            nll = -tf.reduce_mean(tf.reduce_sum(log_prob, axis=-1))
            return nll

        # Batch fitting with Adam
        batch_size = 200
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        data = (X_normalized, y_normalized)
        num_batches = int(np.ceil(subset_size / batch_size))

        print("Fitting GP with batch optimization")
        start_time = time.time()
        for epoch in range(50):
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, subset_size)
                X_batch = tf.constant(data[0][start:end], dtype=tf.float32)
                y_batch = tf.constant(data[1][start:end], dtype=tf.float32)

                with tf.GradientTape() as tape:
                    loss = batch_loss(X_batch, y_batch)
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")
        print(f"GP fitted in {time.time() - start_time:.2f}s")

    def predict_trajectory(self, initial_state):
        state = initial_state.copy()
        trajectories = [state.copy()]

        print(f"Predicting trajectory over {self.steps} steps")
        start_time = time.time()
        for step in range(self.steps - 1):
            action = self.predict_action(state[None, :])[0]
            X = np.hstack([state, action])[None, :]  # Shape: (1, 8)
            X_normalized = (X - self.X_mean) / self.X_std
            X_tf = tf.constant(X_normalized, dtype=tf.float32)

            mean, var = self.model.predict_f(X_tf)
            next_state_diff_normalized = mean.numpy()[0]
            next_state_diff = next_state_diff_normalized * self.y_std + self.y_mean
            print(f"Step {step}, state: {state}, action: {action}, next_state_diff: {next_state_diff}, var: {var.numpy()[0]}")

            state_diff = np.hstack([0.0, next_state_diff])  # No dt scaling
            state = state + state_diff
            state[0] += self.dt

            x = state[1]
            x_dot = state[2]
            state[2] = np.where(np.abs(x) > 3, 0.0, x_dot)
            state[1] = np.where(np.abs(x) > 3, np.clip(x, -3, 3), x)
            trajectories.append(state.copy())

        print(f"Prediction took {time.time() - start_time:.2f}s")
        return np.stack(trajectories, axis=0)

def main():
    tester = PendulumGPTester(num_links=1, sim_time=10.0, dt=0.05)

    # Simulate 20 individuals
    X_data = []
    y_data = []
    for i in range(20):
        initial_state = tester.generate_random_initial_state()
        print(f"Simulating with initial state: {initial_state}")
        X, y = tester.run_simulation(initial_state, i)
        X_data.append(X)
        y_data.append(y)
    X_data = np.vstack(X_data)  # Shape: (4000, 8)
    y_data = np.vstack(y_data)  # Shape: (4000, 5)
    print(f"Full data shapes - X: {X_data.shape}, y: {y_data.shape}")

    # Fit GP with subset
    tester.fit_gp(X_data, y_data)

    # Predict new trajectory
    new_initial_state = tester.generate_random_initial_state()
    print(f"New initial state: {new_initial_state}")
    trajectory = tester.predict_trajectory(new_initial_state)

    # Save to CSV
    with open('policies/predicted_trajectory.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'x', 'x_dot', 'angle_cos', 'angle_sin', 'theta_dot'])
        for t, state in enumerate(trajectory):
            writer.writerow([t * tester.dt, state[1], state[2], state[3], state[4], state[5]])
    print("Trajectory saved to policies/predicted_trajectory.csv")

if __name__ == "__main__":
    main()