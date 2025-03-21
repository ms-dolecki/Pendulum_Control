import numpy as np
from deap import base, creator, tools, algorithms
from gp_control_grok import GPController
import json
import tensorflow as tf
import argparse

class PolicyOptimizer:
    def __init__(self):
        self.controller = GPController()
        self.state_dim = 4
    
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
        state = tf.constant(initial_state, dtype=tf.float32)
        trajectory = [state]
        
        for _ in range(steps-1):
            action = self.predict_action(state, params)
            X = tf.concat([state, tf.reshape(action, [1])], axis=0)
            next_state = self.controller.gp.predict([X.numpy()])[0]
            state = tf.constant(next_state, dtype=tf.float32)
            trajectory.append(state)
        
        return tf.stack(trajectory)
    
    def evaluate_policy(self, individual, initial_state=[0.1, 0, 1, 0]):
        a = tf.constant(np.array(individual[:12]).reshape(3, 4), dtype=tf.float32)
        b = tf.constant(np.array(individual[12:60]).reshape(3, 4, 4), dtype=tf.float32)
        c = tf.constant(np.array(individual[60:252]).reshape(3, 4, 4, 4), dtype=tf.float32)
        params = (a, b, c)
        
        traj = self.predict_trajectory(params, initial_state=initial_state)
        weights = tf.constant([20.0, 1.0, 20.0, 1.0], dtype=tf.float32)
        base_cost = tf.reduce_sum(tf.square(traj) * weights)
        x_values = traj[:, 2]
        penalty = tf.reduce_sum(tf.where(tf.abs(x_values) > 2, 1000.0, 0.0))
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
        self.controller.train_gp(initial_state=initial_state)

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
    
    # Generate initial population
    pop = toolbox.population(n=1000)
    
    # Simulate with 10 random policies from the population
    print("Running initial simulations with 10 random policies")
    for i in range(10):
        individual = pop[i]
        print(f"Simulating random policy {i+1}/10")
        optimizer.run_sim_and_update(individual, initial_state=initial_state)
    
    # Proceed with optimization
    toolbox.register("evaluate", optimizer.evaluate_policy, initial_state=initial_state)
    toolbox.register("mate", tools.cxBlend, alpha=0.75)
    toolbox.register("mutate", custom_mutation, indpb=0.4)
    toolbox.register("select", tools.selTournament, tournsize=10)
    
    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    
    for gen in range(100):
        if gen % 5 == 0 and gen > 0:
            optimizer.run_sim_and_update(hof[0], initial_state=initial_state)
            optimizer.run_sim_and_update(hof[1], initial_state=initial_state)
            optimizer.run_sim_and_update(hof[2], initial_state=initial_state)
        
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