import numpy as np
from scipy.integrate import solve_ivp
import json
import csv
import tensorflow as tf
import os
import time
from numpy.linalg import inv

class PendulumSim:
    def __init__(self, lengths=[1.0], masses=[1.0], friction=0.1, g=9.81):
        self.lengths = np.array(lengths)
        self.masses = np.array(masses)
        self.num_links = len(lengths)
        self.b = friction
        self.g = g
        self.policy = None

    def load_policy(self, policy_file):
        try:
            with open(policy_file, 'r') as f:
                policy = json.load(f)
                policy['params']['a'] = tf.constant(policy['params']['a'], dtype=tf.float32)
                policy['params']['b'] = tf.constant(policy['params']['b'], dtype=tf.float32)
                policy['params']['c'] = tf.constant(policy['params']['c'], dtype=tf.float32)
                return policy
        except Exception as e:
            print(f"Failed to load policy.json: {e}, using default zeros")
            state_dim = 1 + 2 + 3 * self.num_links
            return {
                'type': 'cubic',
                'params': {
                    'a': tf.zeros([3, state_dim], dtype=tf.float32),
                    'b': tf.zeros([3, state_dim, state_dim], dtype=tf.float32),
                    'c': tf.zeros([3, state_dim, state_dim, state_dim], dtype=tf.float32)
                }
            }

    @tf.function
    def vector_square_tf(self, vector):
        return tf.tensordot(vector, vector, axes=0)

    @tf.function
    def vector_cube_tf(self, vector):
        p2 = self.vector_square_tf(vector)
        return tf.tensordot(vector, p2, axes=0)

    @tf.function
    def get_action(self, state):
        #tf.print(state)
        policy_type = self.policy['type']
        if policy_type != 'cubic':
            return tf.constant(0.0, dtype=tf.float32)
        
        a, b, c = self.policy['params']['a'], self.policy['params']['b'], self.policy['params']['c']
        state_tf = tf.cast(state, dtype=tf.float32) if isinstance(state, tf.Tensor) else tf.constant(state, dtype=tf.float32)
        
        p = state_tf
        p2 = self.vector_square_tf(state_tf)
        p3 = self.vector_cube_tf(state_tf)
        
        #tf.print(a[0])
        term1 = tf.reduce_sum(tf.nn.leaky_relu(a[0] * p + a[1]) * a[2])
        term2 = tf.reduce_sum(tf.nn.leaky_relu(b[0] * p2 + b[1]) * b[2])
        term3 = tf.reduce_sum(tf.nn.leaky_relu(c[0] * p3 + c[1]) * c[2])
        
        action = term1 + term2 + term3
        return tf.clip_by_value(action, -10.0, 10.0)

    def K_matrix(self, angles):
        K = np.zeros((self.num_links, self.num_links))
        for i in range(self.num_links):
            for j in range(self.num_links):
                mass_above_ij = sum(self.masses[max(i, j):])
                K[i, j] = mass_above_ij * self.lengths[j] * np.sin(angles[i] - angles[j])
        return K

    def L_matrix(self, angles):
        L = np.zeros((self.num_links, self.num_links))
        for i in range(self.num_links):
            for j in range(self.num_links):
                mass_above_ij = sum(self.masses[max(i, j):])
                L[i, j] = mass_above_ij * self.lengths[j] * np.cos(angles[i] - angles[j])
        return L

    def T_vector(self, angles, external_acceleration):
        T = np.zeros((self.num_links, 1))
        for i in range(self.num_links):
            mass_above_i = sum(self.masses[i:])
            T[i, 0] = mass_above_i * (-self.g * np.sin(angles[i]) - external_acceleration * np.cos(angles[i]))
        return T

    def dynamics(self, t, state):
        delta_t = state[0]
        x = state[1]
        x_dot = state[2]
        angles = []
        theta_dots = []
        for i in range(self.num_links):
            angle_cos, angle_sin, theta_dot = state[3 + i*3 : 6 + i*3]
            theta = np.arctan2(angle_sin, angle_cos)
            angles.append(theta)
            theta_dots.append(theta_dot)
        
        accel = self.get_action(state[1:]).numpy()
        if abs(x) > 3:
            x_dot = 0.0
            accel = 0.0
        
        K = self.K_matrix(angles)
        L = self.L_matrix(angles)
        Linv = inv(L)
        T = self.T_vector(angles, accel)
        angle_dot_squareds = np.array(theta_dots)[:, np.newaxis]**2
        theta_ddots = -Linv @ (T + K @ angle_dot_squareds)
        
        derivs = [0.0]
        derivs.extend([x_dot, accel])
        for i in range(self.num_links):
            angle_cos_dot = -theta_dots[i] * np.sin(angles[i])
            angle_sin_dot = theta_dots[i] * np.cos(angles[i])
            derivs.extend([angle_cos_dot, angle_sin_dot, theta_ddots[i, 0]])
        
        return derivs

    def simulate(self, t_span=(0, 5), initial_state=None, dt=0.1, sim_data_file='policies/sim_data.json', sim_done_file='policies/sim_done.txt', policy_file='policies/policy.json'):
        self.policy = self.load_policy(policy_file)
        if initial_state is None:
            initial_state = [dt] + [0.0] * (2 + 3 * self.num_links)
        elif len(initial_state) != 1 + 2 + 3 * self.num_links:
            raise ValueError(f"Initial state length {len(initial_state)} does not match expected {1 + 2 + 3 * self.num_links}")
        print(f"Simulating with initial state: {initial_state}")
        t_eval = np.arange(t_span[0], t_span[1], dt)
        start = time.time()
        sol = solve_ivp(self.dynamics, t_span, initial_state, 
                       t_eval=t_eval, method='RK45')
        print(f"Simulation took {time.time() - start:.2f}s")
        
        actions = [float(self.get_action(state[1:])) for state in sol.y.T]
        
        with open('policies/trajectory_data.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(len(sol.t)):
                row = [sol.t[i], sol.y[1][i], sol.y[2][i]]
                for j in range(self.num_links):
                    row.extend([sol.y[3 + j*3][i], sol.y[4 + j*3][i], sol.y[5 + j*3][i]])
                writer.writerow(row)
        
        data = {
            'time': sol.t.tolist(),
            'states': sol.y.T.tolist(),
            'actions': actions
        }
        with open(sim_data_file, 'w') as f:
            json.dump(data, f)
        
        with open(sim_done_file, 'w') as f:
            f.write(f"Simulation completed at {time.time()}")
        
        return sol

if __name__ == "__main__":
    sim = PendulumSim(lengths=[1.0])
    sim.simulate(initial_state=[0.1, 0, 0, 1.0, 0.0, 0])
    with open('final_policy.json', 'r') as f:
        policy = json.load(f)
        sim.policy = policy
    sim.simulate(initial_state=[0.1, 0, 0, 1.0, 0.0, 0])