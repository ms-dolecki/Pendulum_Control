import numpy as np
from deap import base, creator, tools, algorithms
import gpflow
import tensorflow as tf
import json
import argparse
from pendulum_sim_grok import PendulumSim
import time
import os
from pathlib import Path

class PolicyOptimizer:
    def __init__(self):
        self.state_dim = 4
        self.X_cache = None
        self.y_cache = None
        kernel = gpflow.kernels.Sum([
            gpflow.kernels.SquaredExponential(lengthscales=0.5),
            gpflow.kernels.Matern32(lengthscales=0.5),
            gpflow.kernels.White(variance=0.1)
        ])
        self.kernel = kernel
        self.gps = None
        self.optimizer = gpflow.optimizers.Scipy()
    
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
    
    def predict_trajectory(self, params, initial_state=[0.1, 0, 1, 0], steps=50):
        state = tf.constant(initial_state, dtype=tf.float64)
        trajectory = [state]
        #print(f"Initial state shape: {state.shape}")
        
        for i in range(steps-1):
            action = self.predict_action(state, params)
            #print(f"Step {i}: Action shape: {action.shape}")
            X = tf.concat([state, tf.reshape(action, [1])], axis=0)[tf.newaxis, :]
            #print(f"Step {i}: X shape: {X.shape}")
            next_state_means = [gp.predict_f(X)[0] for gp in self.gps]
            #print(f"Step {i}: Next state means shapes: {[m.shape for m in next_state_means]}")
            state = tf.concat([m[0] for m in next_state_means], axis=0)
            #print(f"Step {i}: New state shape: {state.shape}")
            trajectory.append(state)
        
        return tf.stack(trajectory)
    
    def evaluate_policy(self, individual, initial_state=[0.1, 0, 1, 0]):
        a = tf.constant(np.array(individual[:12]).reshape(3, 4), dtype=tf.float64)
        b = tf.constant(np.array(individual[12:60]).reshape(3, 4, 4), dtype=tf.float64)
        c = tf.constant(np.array(individual[60:252]).reshape(3, 4, 4, 4), dtype=tf.float64)
        params = (a, b, c)
        
        traj = self.predict_trajectory(params, initial_state=initial_state)
        weights = tf.constant([20.0, 1.0, 20.0, 1.0], dtype=tf.float64)
        base_cost = tf.reduce_sum(tf.square(traj) * weights)
        x_values = traj[:, 2]
        penalty = tf.reduce_sum(tf.where(tf.abs(x_values) > 2, tf.constant(1000.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64)))
        total_cost = base_cost + penalty
        
        return total_cost.numpy(),

    def run_sim_and_update(self, individual, initial_state=[0.1, 0, 1, 0]):
        a = np.array(individual[:12]).reshape(3, 4).tolist()
        b = np.array(individual[12:60]).reshape(3, 4, 4).tolist()
        c = np.array(individual[60:252]).reshape(3, 4, 4, 4).tolist()
        
        policy = {
            'type': 'cubic',
            'params': {'a': a, 'b': b, 'c': c}
        }
        with open('policy.json', 'w') as f:
            json.dump(policy, f)
        
        sim = PendulumSim()
        if os.path.exists('sim_done.txt'):
            os.remove('sim_done.txt')
        
        sim.simulate(t_span=(0, 5), initial_state=initial_state)
        
        max_wait_time = 10
        start_time = time.time()
        while not os.path.exists('sim_done.txt'):
            if time.time() - start_time > max_wait_time:
                raise TimeoutError("Simulation did not complete within expected time")
            time.sleep(0.1)
        
        with open('sim_data.json', 'r') as f:
            data = json.load(f)
        states = np.array(data['states'])
        actions = np.array(data['actions'])
        X_new = np.hstack((states[:-1], actions[:-1].reshape(-1, 1)))
        y_new = states[1:]
        
        if self.X_cache is None:
            self.X_cache, self.y_cache = X_new, y_new
        else:
            self.X_cache = np.vstack([self.X_cache, X_new])[-5000:]
            self.y_cache = np.vstack([self.y_cache, y_new])[-5000:]
        
        X_tf = tf.constant(self.X_cache, dtype=tf.float64)
        y_tf = tf.constant(self.y_cache, dtype=tf.float64)
        #print(f"X_cache shape: {X_tf.shape}, y_cache shape: {y_tf.shape}")
        
        if self.gps is None:
            self.gps = []
            for i in range(self.state_dim):
                gp = gpflow.models.SVGP(
                    kernel=self.kernel,
                    likelihood=gpflow.likelihoods.Gaussian(),
                    inducing_variable=X_tf[:100],
                    num_data=len(self.X_cache)
                )
                self.gps.append(gp)
        
        for i, gp in enumerate(self.gps):
            gp.data = (X_tf, y_tf[:, i:i+1])
            closure = gp.training_loss_closure(data=(X_tf, y_tf[:, i:i+1]), compile=True)
            self.optimizer.minimize(closure, gp.trainable_variables, options={'maxiter': 100})

def init_individual(ind_class):
    a = np.random.uniform(-100, 100, size=12)
    b = np.random.uniform(-10, 10, size=48)
    c = np.random.uniform(-1, 1, size=192)
    return ind_class(np.concatenate([a, b, c]))

def custom_mutation(individual, indpb):
    sigma_a = 20.0
    sigma_b = 2.0
    sigma_c = 0.5
    
    for i in range(12):
        if np.random.random() < indpb:
            individual[i] += np.random.normal(0, sigma_a)
    
    for i in range(12, 60):
        if np.random.random() < indpb:
            individual[i] += np.random.normal(0, sigma_b)
    
    for i in range(60, 252):
        if np.random.random() < indpb:
            individual[i] += np.random.normal(0, sigma_c)
    
    return individual,

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("individual", init_individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def optimize(initial_state=[0.1, 0, 1, 0]):
    optimizer = PolicyOptimizer()
    
    pop = toolbox.population(n=1000)
    
    print("Running initial simulations with 10 random policies")
    for i in range(10):
        individual = pop[i]
        print(f"Simulating random policy {i+1}/10")
        optimizer.run_sim_and_update(individual, initial_state=initial_state)
    
    toolbox.register("evaluate", optimizer.evaluate_policy, initial_state=initial_state)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", custom_mutation, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=25)
    
    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    
    for gen in range(20):
        if gen % 10 == 0 and gen > 0:
            optimizer.run_sim_and_update(hof[0], initial_state=initial_state)
        
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, 
                                     ngen=1, stats=stats, halloffame=hof, 
                                     verbose=False)
        
        best_cost = hof[0].fitness.values[0]
        best_policy = {
            'a': np.array(hof[0][:12]).reshape(3, 4).tolist(),
            'b': np.array(hof[0][12:60]).reshape(3, 4, 4).tolist(),
            'c': np.array(hof[0][60:252]).reshape(3, 4, 4, 4).tolist()
        }
        print(f"Gen {gen}: Best cost = {best_cost:.2f}, Best policy = {json.dumps(best_policy, indent=2)}")
    
    best_policy = hof[0]
    print(f"Best policy found with {len(best_policy)} parameters")
    with open('final_policy.json', 'w') as f:
        json.dump({'type': 'cubic', 'params': best_policy}, f)
    return best_policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize pendulum policy with initial state")
    parser.add_argument('--theta', type=float, default=0.1, help="Initial theta (rad)")
    parser.add_argument('--theta_dot', type=float, default=0.0, help="Initial theta_dot (rad/s)")
    parser.add_argument('--x', type=float, default=1.0, help="Initial x (m)")
    parser.add_argument('--x_dot', type=float, default=0.0, help="Initial x_dot (m/s)")
    args = parser.parse_args()
    
    initial_state = [args.theta, args.theta_dot, args.x, args.x_dot]
    print(f"Running optimization with initial state: {initial_state}")
    best_policy = optimize(initial_state=initial_state)