import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import json
import argparse
from pendulum_sim_grok import PendulumSim
import time
import os
import csv
from pathlib import Path

class PolicyOptimizer:
    def __init__(self, num_links=1):
        self.num_links = num_links
        self.state_dim = 1 + 2 + 3 * num_links
        self.X_cache = None
        self.y_cache = None
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e5))
        self.gp = GaussianProcessRegressor(kernel=kernel)
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
    
    @tf.function
    def predict_action(self, state, params):
        a, b, c = params
        p = state
        p2 = tf.tensordot(p, p, axes=0)
        p3 = tf.tensordot(p2, p, axes=0)
        term1 = tf.reduce_sum(tf.nn.leaky_relu(a[0] * p + a[1]) * a[2])
        term2 = tf.reduce_sum(tf.nn.leaky_relu(b[0] * p2 + b[1]) * b[2])
        term3 = tf.reduce_sum(tf.nn.leaky_relu(c[0] * p3 + c[1]) * c[2])
        action = term1 + term2 + term3
        return tf.clip_by_value(action, -3.0, 3.0)
    
    def predict_trajectory(self, params, initial_state, steps=50, dt=0.1):
        state = tf.constant(initial_state, dtype=tf.float64)
        trajectory = [state]
        times = [0.0]
        
        for i in range(steps-1):
            action = self.predict_action(state, params)
            X = tf.concat([state, tf.reshape(action, [1])], axis=0)[tf.newaxis, :]
            X_scaled = tf.constant(self.X_scaler.transform(X.numpy()), dtype=tf.float64)
            next_state_scaled = tf.constant(self.gp.predict(X_scaled.numpy()), dtype=tf.float64)
            next_state = tf.constant(self.y_scaler.inverse_transform(next_state_scaled.numpy()), dtype=tf.float64)[0]
            #print(f"Step {i}: Action {action:.2f}, Next state {next_state.numpy()}")  # Debug
            state = next_state
            trajectory.append(state)
            times.append((i + 1) * dt)
        
        return tf.stack(trajectory), times
    
    def evaluate_policy(self, individual, initial_state):
        a = tf.constant(np.array(individual[:3*self.state_dim]).reshape(3, self.state_dim), dtype=tf.float64)
        b = tf.constant(np.array(individual[3*self.state_dim:3*self.state_dim + 3*self.state_dim**2]).reshape(3, self.state_dim, self.state_dim), dtype=tf.float64)
        c = tf.constant(np.array(individual[3*self.state_dim + 3*self.state_dim**2:]).reshape(3, self.state_dim, self.state_dim, self.state_dim), dtype=tf.float64)
        params = (a, b, c)
        
        traj = self.predict_trajectory(params, initial_state=initial_state)[0]
        weights = tf.constant([10.0] * (1 + self.num_links) + [1.0] * self.num_links + [10.0, 1.0], dtype=tf.float64)
        angles = [tf.atan2(traj[:, 4 + i*3], traj[:, 3 + i*3]) for i in range(self.num_links)]
        angle_dots = [traj[:, 5 + i*3] for i in range(self.num_links)]
        cost_terms = tf.concat([tf.stack(angles, axis=1), tf.stack(angle_dots, axis=1), traj[:, 1:3]], axis=1)
        base_cost = tf.reduce_sum(tf.square(cost_terms) * weights[1:])
        x_values = traj[:, self.state_dim - 2]
        penalty = tf.reduce_sum(tf.where(tf.abs(x_values) > 2, tf.constant(500.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64)))
        total_cost = base_cost + penalty
        
        return total_cost.numpy(),

    def run_sim_and_update(self, individuals, initial_state, sim_time=5.0, dt=0.1):
        for idx, individual in enumerate(individuals):
            a = np.array(individual[:3*self.state_dim]).reshape(3, self.state_dim).tolist()
            b = np.array(individual[3*self.state_dim:3*self.state_dim + 3*self.state_dim**2]).reshape(3, self.state_dim, self.state_dim).tolist()
            c = np.array(individual[3*self.state_dim + 3*self.state_dim**2:]).reshape(3, self.state_dim, self.state_dim, self.state_dim).tolist()
            
            policy = {
                'type': 'cubic',
                'params': {'a': a, 'b': b, 'c': c}
            }
            with open('policy.json', 'w') as f:
                json.dump(policy, f)
            
            sim = PendulumSim(lengths=[initial_state[3 + i*3] for i in range(self.num_links)], masses=[1.0]*self.num_links)
            if os.path.exists('sim_done.txt'):
                os.remove('sim_done.txt')
            
            sim.simulate(t_span=(0, sim_time), initial_state=initial_state, dt=dt)
            
            max_wait_time = 2
            start_time = time.time()
            while not os.path.exists('sim_done.txt'):
                if time.time() - start_time > max_wait_time:
                    raise TimeoutError("Simulation did not complete within expected time")
                time.sleep(0.01)
            
            with open('sim_data.json', 'r') as f:
                data = json.load(f)
            states = np.array(data['states'])
            actions = np.array(data['actions'])
            X_new = np.hstack((states[:-1], actions[:-1].reshape(-1, 1)))
            y_new = states[1:]
            
            if self.X_cache is None and idx == 0:
                self.X_cache, self.y_cache = X_new, y_new
            else:
                self.X_cache = np.vstack([self.X_cache, X_new])[:]
                self.y_cache = np.vstack([self.y_cache, y_new])[:]
        
        X_scaled = self.X_scaler.fit_transform(self.X_cache)
        y_scaled = self.y_scaler.fit_transform(self.y_cache)
        self.gp.fit(X_scaled, y_scaled)

def init_individual(ind_class, state_dim):
    a = np.random.uniform(-100, 100, size=3*state_dim)
    b = np.random.uniform(-10, 10, size=3*state_dim*state_dim)
    c = np.random.uniform(-1, 1, size=3*state_dim*state_dim*state_dim)
    return ind_class(np.concatenate([a, b, c]))

def custom_mutation(individual, indpb, state_dim):
    sigma_a = 20.0
    sigma_b = 2.0
    sigma_c = 0.5
    
    for i in range(3*state_dim):
        if np.random.random() < indpb:
            individual[i] += np.random.normal(0, sigma_a)
    
    for i in range(3*state_dim, 3*state_dim + 3*state_dim**2):
        if np.random.random() < indpb:
            individual[i] += np.random.normal(0, sigma_b)
    
    for i in range(3*state_dim + 3*state_dim**2, len(individual)):
        if np.random.random() < indpb:
            individual[i] += np.random.normal(0, sigma_c)
    
    return individual,

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def optimize(initial_state, num_links, sim_time=5.0, sample_rate=10.0):
    dt = 1.0 / sample_rate
    print(f"Sample rate: {sample_rate}, dt: {dt}")
    state_dim = 1 + 2 + 3 * num_links
    toolbox.register("individual", init_individual, creator.Individual, state_dim=state_dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    optimizer = PolicyOptimizer(num_links=num_links)
    
    os.makedirs('policies', exist_ok=True)
    
    pop = toolbox.population(n=1000)
    
    print("Running initial simulations with 10 random policies")
    for i in range(10):
        individual = pop[i]
        print(f"Simulating random policy {i+1}/10")
        optimizer.run_sim_and_update([individual], initial_state=initial_state, sim_time=sim_time, dt=dt)
    
    toolbox.register("evaluate", optimizer.evaluate_policy, initial_state=initial_state)
    toolbox.register("mate", tools.cxBlend, alpha=1.0)
    toolbox.register("mutate", custom_mutation, indpb=0.5, state_dim=state_dim)
    toolbox.register("select", tools.selTournament, tournsize=10)
    
    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    
    for gen in range(100):
        if gen % 5 == 0 and gen > 0:
            optimizer.run_sim_and_update(hof[:3], initial_state=initial_state, sim_time=sim_time, dt=dt)
            for ind in hof:
                ind.fitness.values = optimizer.evaluate_policy(ind, initial_state=initial_state)
            hof.update(hof)
        
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.5, 
                                     ngen=1, stats=stats, halloffame=hof, 
                                     verbose=False)
        
        best_cost = hof[0].fitness.values[0]
        best_policy = {
            'a': np.array(hof[0][:3*state_dim]).reshape(3, state_dim).tolist(),
            'b': np.array(hof[0][3*state_dim:3*state_dim + 3*state_dim**2]).reshape(3, state_dim, state_dim).tolist(),
            'c': np.array(hof[0][3*state_dim + 3*state_dim**2:]).reshape(3, state_dim, state_dim, state_dim).tolist()
        }
        best_traj, times = optimizer.predict_trajectory(
            (tf.constant(np.array(hof[0][:3*state_dim]).reshape(3, state_dim), dtype=tf.float64),
             tf.constant(np.array(hof[0][3*state_dim:3*state_dim + 3*state_dim**2]).reshape(3, state_dim, state_dim), dtype=tf.float64),
             tf.constant(np.array(hof[0][3*state_dim + 3*state_dim**2:]).reshape(3, state_dim, state_dim, state_dim), dtype=tf.float64)),
            initial_state=initial_state, steps=int(sim_time/dt) + 1, dt=dt
        )
        
        policy_file = f'policies/best_policy_gen_{gen}.json'
        with open(policy_file, 'w') as f:
            json.dump({'type': 'cubic', 'params': best_policy, 'cost': best_cost}, f, indent=2)
        
        traj_file = f'policies/best_trajectory_gen_{gen}.csv'
        with open(traj_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for t, state in zip(times, best_traj.numpy()):
                row = [t, state[1], state[2]]
                for i in range(optimizer.num_links):
                    row.extend([state[3 + i*3], state[4 + i*3], state[5 + i*3]])
                writer.writerow(row)
        
        print(f"Gen {gen}: Best cost = {best_cost:.2f}, policy saved to {policy_file}, trajectory saved to {traj_file}")
    
    best_policy = hof[0]
    with open('policies/final_policy.json', 'w') as f:
        json.dump({'type': 'cubic', 'params': best_policy}, f, indent=2)
    print(f"Best policy found with {len(best_policy)} parameters, saved to policies/final_policy.json")
    return best_policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize multi-link pendulum policy")
    parser.add_argument('--sim_time', type=float, default=5.0, help="Simulation time (seconds)")
    parser.add_argument('--sample_rate', type=float, default=10.0, help="Sample rate (Hz)")
    parser.add_argument('--num_links', type=int, default=1, help="Number of pendulum links")
    parser.add_argument('--x', type=float, default=0.0, help="Initial x (m)")
    parser.add_argument('--x_dot', type=float, default=0.0, help="Initial x_dot (m/s)")
    for i in range(1, 11):
        parser.add_argument(f'--length{i}', type=float, default=1.0, help=f"Length of link {i} (m)")
        parser.add_argument(f'--angle{i}', type=float, default=0.0, help=f"Initial angle of link {i} (rad)")
        parser.add_argument(f'--angle_dot{i}', type=float, default=0.0, help=f"Initial angle_dot of link {i} (rad/s)")
    args = parser.parse_args()
    
    lengths = [getattr(args, f'length{i}') for i in range(1, args.num_links + 1)]
    initial_state = [1.0 / args.sample_rate, args.x, args.x_dot]
    for i in range(1, args.num_links + 1):
        angle = getattr(args, f'angle{i}')
        angle_dot = getattr(args, f'angle_dot{i}')
        initial_state.extend([np.cos(angle), np.sin(angle), angle_dot])
    
    print(f"Running optimization with initial state: {initial_state}, lengths: {lengths}")
    best_policy = optimize(initial_state=initial_state, num_links=args.num_links, 
                          sim_time=args.sim_time, sample_rate=args.sample_rate)