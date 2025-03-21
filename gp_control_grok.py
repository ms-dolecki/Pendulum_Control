import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import json
from pendulum_sim_grok import PendulumSim

class GPController:
    def __init__(self):
        self.sim = PendulumSim()
        self.X_cache = None
        self.y_cache = None
    
    def write_policy(self, policy_type='zero', params={}):
        policy = {'type': policy_type, 'params': params}
        with open('policy.json', 'w') as f:
            json.dump(policy, f)
    
    def collect_data(self, initial_state=[0.1, 0, 1, 0], n_runs=2, max_points=50000):
        X_all, y_all = [], []
        for _ in range(n_runs):
            self.sim.policy = self.sim.load_policy()  # Reload policy each run
            self.sim.simulate(initial_state=initial_state)
            with open('sim_data.json', 'r') as f:
                data = json.load(f)
            states = np.array(data['states'])
            actions = np.array(data['actions'])
            X = np.hstack((states[:-1], actions[:-1].reshape(-1, 1)))
            y = states[1:, :]
            X_all.append(X)
            y_all.append(y)
        X_new, y_new = np.vstack(X_all), np.vstack(y_all)
        
        if self.X_cache is None:
            self.X_cache, self.y_cache = X_new, y_new
        else:
            X_combined = np.vstack([self.X_cache, X_new])
            y_combined = np.vstack([self.y_cache, y_new])
            indices = np.random.permutation(X_combined.shape[0])
            X_shuffled = X_combined[indices]
            y_shuffled = y_combined[indices]
            self.X_cache = X_shuffled[-max_points:]
            self.y_cache = y_shuffled[-max_points:]
        
        return self.X_cache, self.y_cache
    
    def train_gp(self, initial_state=[0.1, 0, 1, 0]):
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e5))
        self.gp = GaussianProcessRegressor(kernel=kernel)
        X, y = self.collect_data(initial_state=initial_state)
        self.gp.fit(X, y)

if __name__ == "__main__":
    controller = GPController()
    controller.write_policy('cubic', {'a': [[0]*4]*3, 'b': [[[0]*4]*4]*3, 'c': [[[[0]*4]*4]*4]*3})
    controller.train_gp()