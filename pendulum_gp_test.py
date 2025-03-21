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
        self.output_dim = 2 + 3 * num_links    # diffs: x, x_dot, cosθ, sinθ, θ_dot
        self.sim_time = sim_time
        self.dt = dt
        self.steps = int(sim_time / dt) + 1  # 801 for 10s at 0.0125

        # Random policy (cubic, fixed)
        self.policy_params = self._generate_random_policy()

    def _generate_random_policy(self):
        a = np.random.uniform(-5, 5, (3, self.state_dim-1))
        b = np.random.uniform(-0.5, 0.5, (3, self.state_dim-1, self.state_dim-1))
        c = np.random.uniform(-0.05, 0.05, (3, self.state_dim-1, self.state_dim-1, self.state_dim-1))
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
        delta_T = actions*0+self.dt
        X = np.hstack([delta_T[:-1,None],np.hstack([states[:-1,1:], actions[:-1, None]])])  # Shape: (800, 8)
        #y = states[1:, 1:] - states[:-1, 1:]  # Shape: (800, 5)
        y = states[1:,1:]
        return X, y

    def generate_random_initial_state(self):
        x = np.random.uniform(-1.0, 1.0)
        x_dot = np.random.uniform(-1.0, 1.0)
        theta = np.random.uniform(0, 2 * np.pi)
        theta_dot = np.random.uniform(-1.0, 1.0)
        return np.array([self.dt, x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    def fit_gp(self, X_data, y_data):
        # Subset to 5000 points
        subset_size = min(5000, X_data.shape[0])
        indices = np.random.choice(X_data.shape[0], subset_size, replace=False)
        X_subset = X_data[indices].astype(np.float32)
        y_subset = y_data[indices].astype(np.float32)  # y_subset is now states
        print(f"Subset shapes - X: {X_subset.shape}, y: {y_subset.shape}")

        # Diagnostic: Check data scales
        print(f"X_subset min: {np.min(X_subset)}, max: {np.max(X_subset)}")
        print(f"y_subset min: {np.min(y_subset)}, max: {np.max(y_subset)}")

        # Normalize X and remove near-constant features
        self.X_mean = np.mean(X_subset, axis=0, dtype=np.float32)
        self.X_std = np.std(X_subset, axis=0, dtype=np.float32) + 1e-6
        mask = self.X_std > 1e-4
        self.mask = mask
        X_subset_clean = X_subset[:, mask]
        self.X_mean = self.X_mean[mask]
        self.X_std = self.X_std[mask]
        X_normalized = (X_subset_clean - self.X_mean) / self.X_std
        print(f"Cleaned X shape: {X_normalized.shape}, X_std: {self.X_std}")

        # Normalize y (mandatory for states with large range)
        self.y_mean = np.mean(y_subset, axis=0, dtype=np.float32)
        self.y_std = np.std(y_subset, axis=0, dtype=np.float32) + 1e-6
        mask_y = self.y_std > 1e-4  # Remove near-constant y dimensions if any
        y_subset_clean = y_subset[:, mask_y]
        self.y_mean = self.y_mean[mask_y]
        self.y_std = self.y_std[mask_y]
        y_normalized = (y_subset_clean - self.y_mean) / self.y_std
        print(f"Cleaned y shape: {y_normalized.shape}, y_std: {self.y_std}")

        # Define kernel
        kernel = gpflow.kernels.SquaredExponential(lengthscales=tf.constant(1.0, dtype=tf.float32))

        # Try exact GP first
        print("Trying GPR")
        self.model = gpflow.models.GPR(
            data=(tf.constant(X_normalized, dtype=tf.float32), tf.constant(y_normalized, dtype=tf.float32)),
            kernel=kernel
        )
        self.model.likelihood.variance.assign(1.0)  # Larger initial jitter
        self.model.likelihood.variance = gpflow.Parameter(
            self.model.likelihood.variance, transform=gpflow.utilities.positive(lower=1e-3)
        )

        # Optimize
        start_time = time.time()
        optimizer = gpflow.optimizers.Scipy()
        try:
            optimizer.minimize(
                self.model.training_loss,
                self.model.trainable_variables,
                options=dict(maxiter=1000)
            )
            print(f"GPR fitted in {time.time() - start_time:.2f}s")
        except tf.errors.InvalidArgumentError as e:
            print(f"GPR failed: {e}")
            # Fallback to sparse GP
            print("Switching to SGPR")
            num_inducing = 500
            inducing_points = X_normalized[np.random.choice(X_normalized.shape[0], num_inducing, replace=False)]
            self.model = gpflow.models.SGPR(
                data=(tf.constant(X_normalized, dtype=tf.float32), tf.constant(y_normalized, dtype=tf.float32)),
                kernel=kernel,
                inducing_variable=inducing_points
            )
            self.model.likelihood.variance.assign(1.0)
            self.model.likelihood.variance = gpflow.Parameter(
                self.model.likelihood.variance, transform=gpflow.utilities.positive(lower=1e-3)
            )
            start_time = time.time()
            optimizer.minimize(
                self.model.training_loss,
                self.model.trainable_variables,
                options=dict(maxiter=1000)
            )
            print(f"SGPR fitted in {time.time() - start_time:.2f}s")

        # Check learned parameters
        print(f"Lengthscale: {self.model.kernel.lengthscales.numpy()}")
        print(f"Likelihood variance: {self.model.likelihood.variance.numpy()}")

        # Test predictions
        X_test = X_normalized[:5]
        y_pred, y_var = self.model.predict_f(tf.constant(X_test, dtype=tf.float32))
        y_pred_denorm = y_pred * self.y_std + self.y_mean
        print(f"Predictions:\n{y_pred_denorm.numpy()}")
        print(f"True values:\n{y_subset_clean[:5]}")

    def predict_trajectory(self, initial_state):
        state = initial_state.copy()
        trajectories = [state.copy()]

        print(f"Predicting trajectory over {self.steps} steps")
        start_time = time.time()
        for step in range(self.steps - 1):
            action = self.predict_action(state[None, 1:])[0]
            # Include time in X to match training data
            X = np.hstack([state, action])[self.mask]  # Shape: (1, 8) - [time, x, x_dot, cosθ, sinθ, θ_dot, action]
            X_normalized = (X - self.X_mean) / self.X_std
            X_tf = tf.constant(X_normalized, dtype=tf.float32)
            X_tf = tf.expand_dims(X_tf, axis=0)

            mean, _ = self.model.predict_f(X_tf)
            next_state_normalized = mean.numpy()[0]
            next_state = next_state_normalized * self.y_std + self.y_mean  # If y was normalized

            #state_diff = np.hstack([0.0, next_state_diff])
            #state = state + state_diff * self.dt
            #state = state_diff
            #state[0] += self.dt  # Increment time
            state = np.hstack([self.dt, next_state])
            x = state[1]
            x_dot = state[2]
            state[2] = np.where(np.abs(x) > 3, 0.0, x_dot)
            state[1] = np.where(np.abs(x) > 3, np.clip(x, -3, 3), x)
            trajectories.append(state.copy())

        print(f"Prediction took {time.time() - start_time:.2f}s")
        return np.stack(trajectories, axis=0)

def main():
    tester = PendulumGPTester(num_links=1, sim_time=10.0, dt=0.0125)

    # Simulate 20 individuals
    X_data = []
    y_data = []
    for i in range(20):
        initial_state = tester.generate_random_initial_state()
        print(f"Simulating with initial state: {initial_state}")
        X, y = tester.run_simulation(initial_state, i)
        print("y")
        print(y)
        X_data.append(X)
        y_data.append(y)
    X_data = np.vstack(X_data)  # Shape: (16000, 8)
    y_data = np.vstack(y_data)  # Shape: (16000, 5)
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
        for index,state in enumerate(trajectory):
            print(state)
            writer.writerow([state[0], state[1], state[2], state[3], state[4], state[5]])
    print("Trajectory saved to policies/predicted_trajectory.csv")

if __name__ == "__main__":
    main()