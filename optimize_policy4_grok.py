import numpy as np
from deap import base, creator, tools, algorithms
import gpflow
import tensorflow as tf
import json
import argparse
from pendulum_sim_grok import PendulumSim
import time
import os
import csv
from pathlib import Path

gpflow.config.set_default_float(tf.float64)

class PolicyOptimizer:
    def __init__(self, num_links=1):
        self.num_links = num_links
        self.state_dim = 1 + 2 + 3 * num_links
        self.X_cache = None
        self.y_cache = None
        kernel = gpflow.kernels.Sum([
            gpflow.kernels.SquaredExponential(lengthscales=0.5),
            gpflow.kernels.White(variance=1e-6)
        ])
        self.model = gpflow.models.GPR(data=(np.empty((0, self.state_dim + 1)), np.empty((0, self.state_dim))), 
                                      kernel=kernel)
        self.X_scaler_tf = self.TFStandardScaler()
        self.y_scaler_tf = self.TFStandardScaler()

    class TFStandardScaler:
        def __init__(self):
            self.mean = None
            self.std = None

        def fit(self, X):
            self.mean = tf.reduce_mean(X, axis=0)
            self.std = tf.math.reduce_std(X, axis=0)
            self.std = tf.where(self.std < 1e-7, 1e-7, self.std)

        @tf.function(reduce_retracing=True)
        def transform(self, X):
            if self.mean is None or self.std is None:
                raise ValueError("Scaler not fitted")
            return (X - self.mean) / self.std

        @tf.function(reduce_retracing=True)
        def inverse_transform(self, Y):
            if self.mean is None or self.std is None:
                raise ValueError("Scaler not fitted")
            return (Y * self.std) + self.mean

        @tf.function(reduce_retracing=True)
        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)

    @tf.function
    def predict_action(self, states, params):
        a, b, c = params
        p = tf.ensure_shape(states, [None, self.state_dim])  # (batch_size, state_dim)
        p2 = tf.einsum('bi,bj->bij', p, p)  # (batch_size, state_dim, state_dim)
        p3 = tf.einsum('bij,bk->bijk', p2, p)  # (batch_size, state_dim, state_dim, state_dim)
        
        term1 = tf.reduce_sum(tf.nn.leaky_relu(a[:, 0, :] * p + a[:, 1, :]) * a[:, 2, :], axis=1)
        term2 = tf.reduce_sum(tf.nn.leaky_relu(b[:, 0, :, :] * p2 + b[:, 1, :, :]) * b[:, 2, :, :], axis=[1, 2])
        term3 = tf.reduce_sum(tf.nn.leaky_relu(c[:, 0, :, :, :] * p3 + c[:, 1, :, :, :]) * c[:, 2, :, :, :], axis=[1, 2, 3])
        actions = term1 + term2 + term3
        return tf.clip_by_value(actions, -3.0, 3.0)

    @tf.function
    def predict_trajectory_batch(self, params_batch, initial_state, steps=50, dt=0.1):
        batch_size = tf.shape(params_batch[0])[0]
        state = tf.tile(initial_state[tf.newaxis, :], [batch_size, 1])
        trajectories = [state]
        
        for _ in range(steps-1):
            actions = self.predict_action(state, params_batch)
            X = tf.concat([state, actions[:, tf.newaxis]], axis=1)
            X_scaled = self.X_scaler_tf.transform(X)
            mean, _ = self.model.predict_f(X_scaled)
            state = self.y_scaler_tf.inverse_transform(mean)
            trajectories.append(state)
        
        return tf.stack(trajectories, axis=1)

    @tf.function
    def evaluate_policy_batch(self, individuals, initial_state):
        batch_size = tf.shape(individuals)[0]
        a = tf.reshape(individuals[:, :3*self.state_dim], [batch_size, 3, self.state_dim])
        b = tf.reshape(individuals[:, 3*self.state_dim:3*self.state_dim + 3*self.state_dim**2], 
                      [batch_size, 3, self.state_dim, self.state_dim])
        c = tf.reshape(individuals[:, 3*self.state_dim + 3*self.state_dim**2:], 
                      [batch_size, 3, self.state_dim, self.state_dim, self.state_dim])
        params_batch = (a, b, c)
        
        traj = self.predict_trajectory_batch(params_batch, initial_state)
        angles = [tf.atan2(traj[:, :, 4 + i*3], traj[:, :, 3 + i*3]) for i in range(self.num_links)]
        angle_dots = [traj[:, :, 5 + i*3] for i in range(self.num_links)]
        angle_cost = tf.reduce_sum(tf.square(tf.stack(angles, axis=2)), axis=[1, 2]) * 20.0
        velocity_cost = tf.reduce_sum(tf.square(tf.stack(angle_dots, axis=2)), axis=[1, 2]) * 1.0
        x_cost = tf.reduce_sum(tf.square(traj[:, :, 1]), axis=1) * 5.0
        x_values = tf.ensure_shape(traj[:, :, self.state_dim - 2], [None, None])
        penalty = tf.reduce_sum(tf.where(tf.abs(x_values) > 2, 50.0 * (tf.abs(x_values) - 2), 0.0), axis=1)
        total_cost = angle_cost + velocity_cost + x_cost + penalty
        return total_cost

    def run_sim_and_update(self, individuals, initial_state, sim_time=5.0, dt=0.1):
        for idx, individual in enumerate(individuals):
            a = np.array(individual[:3*self.state_dim]).reshape(3, self.state_dim)
            b = np.array(individual[3*self.state_dim:3*self.state_dim + 3*self.state_dim**2]).reshape(3, self.state_dim, self.state_dim)
            c = np.array(individual[3*self.state_dim + 3*self.state_dim**2:]).reshape(3, self.state_dim, self.state_dim, self.state_dim)
            
            policy = {'type': 'cubic', 'params': {'a': a.tolist(), 'b': b.tolist(), 'c': c.tolist()}}
            with open('policy.json', 'w') as f:
                json.dump(policy, f)
            
            sim = PendulumSim(lengths=[initial_state[3 + i*3] for i in range(self.num_links)], masses=[1.0]*self.num_links)
            if os.path.exists('sim_done.txt'):
                os.remove('sim_done.txt')
            
            sim.simulate(t_span=(0, sim_time), initial_state=initial_state, dt=dt)
            
            max_wait_time = 5
            start_time = time.time()
            while not os.path.exists('sim_done.txt'):
                if time.time() - start_time > max_wait_time:
                    raise TimeoutError(f"Simulation {idx} did not complete within {max_wait_time}s")
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
                self.X_cache = np.vstack([self.X_cache, X_new])
                self.y_cache = np.vstack([self.y_cache, y_new])
            
        X_scaled = tf.constant(self.X_cache, dtype=tf.float64)
        y_scaled = tf.constant(self.y_cache, dtype=tf.float64)
        self.X_scaler_tf.fit(X_scaled)
        self.y_scaler_tf.fit(y_scaled)
        self.model.data = (self.X_scaler_tf.transform(X_scaled), self.y_scaler_tf.transform(y_scaled))
        optimizer = gpflow.optimizers.Scipy()
        optimizer.minimize(self.model.training_loss, self.model.trainable_variables, options=dict(maxiter=100))

def init_individual(ind_class, state_dim):
    a = np.random.uniform(-100, 100, size=3*state_dim)
    b = np.random.uniform(-10, 10, size=3*state_dim*state_dim)
    c = np.random.uniform(-1, 1, size=3*state_dim*state_dim*state_dim)
    return ind_class(np.concatenate([a, b, c]))

def custom_mutation(individual, indpb, state_dim, generation, max_gens):
    scale = 1.0 - generation / max_gens
    sigma_a = 20.0 * scale
    sigma_b = 2.0 * scale
    sigma_c = 0.5 * scale
    
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

def optimize(initial_state, num_links, sim_time=5.0, sample_rate=10.0):
    dt = 1.0 / sample_rate
    print(f"Sample rate: {sample_rate}, dt: {dt}")
    state_dim = 1 + 2 + 3 * num_links
    optimizer = PolicyOptimizer(num_links=num_links)
    
    os.makedirs('policies', exist_ok=True)
    
    num_subpops = 5
    subpop_size = 20
    alphas = [0.5, 0.8, 1.0, 1.2, 1.5]
    subpops = []
    toolboxes = []
    hofs = []
    
    for i in range(num_subpops):
        toolbox = base.Toolbox()
        toolbox.register("individual", init_individual, creator.Individual, state_dim=state_dim)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", lambda inds: optimizer.evaluate_policy_batch(tf.constant(inds, dtype=tf.float64), initial_state).numpy())
        toolbox.register("mate", tools.cxBlend, alpha=alphas[i])
        toolbox.register("mutate", custom_mutation, indpb=0.5, state_dim=state_dim, generation=0, max_gens=100)
        toolbox.register("select", tools.selTournament, tournsize=5)
        
        subpops.append(toolbox.population(n=subpop_size))
        toolboxes.append(toolbox)
        hofs.append(tools.HallOfFame(3))
    
    print("Running initial simulations with 10 random policies")
    start_time = time.time()
    for i in range(10):
        subpop_idx = i % num_subpops
        individual = subpops[subpop_idx][i // num_subpops]
        print(f"Simulating random policy {i+1}/10 in subpop {subpop_idx}")
        optimizer.run_sim_and_update([individual], initial_state=initial_state.numpy(), sim_time=sim_time, dt=dt)
    print(f"Initial sims took {time.time() - start_time:.2f} seconds")
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    
    for gen in range(100):
        start_gen = time.time()
        print(f"Starting gen {gen} at {start_gen:.2f}")
        if gen % 5 == 0 and gen > 0:
            best_individuals = [hof[0] for hof in hofs if hof and hof[0].fitness.valid]
            if best_individuals:
                optimizer.run_sim_and_update(best_individuals, initial_state=initial_state.numpy(), sim_time=sim_time, dt=dt)
        
        for i, (pop, toolbox, hof) in enumerate(zip(subpops, toolboxes, hofs)):
            toolbox.register("mutate", custom_mutation, indpb=0.5, state_dim=state_dim, generation=gen, max_gens=100)
            print(f"Starting eaSimple for subpop {i} at {time.time():.2f}")
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            fitnesses = toolbox.evaluate(np.array(invalid_ind))  # Batch eval
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit,)
            pop[:] = toolbox.select(pop, len(pop))
            pop[:] = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.8)
            if gen % 10 == 0 and gen > 0:
                for j in range(int(0.1 * subpop_size)):
                    pop[j] = toolbox.individual()
            if hof is not None:
                hof.update(pop)
            pop.sort(key=lambda x: x.fitness.values[0] if x.fitness.valid else float('inf'))
            print(f"Finished eaSimple for subpop {i} at {time.time():.2f}")
            
            if gen % 10 == 0 and gen > 0:
                for ind in hof[:3]:
                    toolbox.mutate(ind)
        
        valid_hof_entries = [hof[0] for hof in hofs if hof and hof[0].fitness.valid]
        if not valid_hof_entries:
            print(f"No valid individuals in Hall of Fame at gen {gen} - skipping best_ind update")
            continue
        best_ind = min(valid_hof_entries, key=lambda x: x.fitness.values[0])
        best_cost = best_ind.fitness.values[0]
        best_policy = {
            'a': np.array(best_ind[:3*state_dim]).reshape(3, state_dim).tolist(),
            'b': np.array(best_ind[3*state_dim:3*state_dim + 3*state_dim**2]).reshape(3, state_dim, state_dim).tolist(),
            'c': np.array(best_ind[3*state_dim + 3*state_dim**2:]).reshape(3, state_dim, state_dim, state_dim).tolist()
        }
        best_traj = optimizer.predict_trajectory_batch(
            (tf.constant(np.array(best_ind[:3*state_dim]).reshape(1, 3, state_dim), dtype=tf.float64),
             tf.constant(np.array(best_ind[3*state_dim:3*state_dim + 3*state_dim**2]).reshape(1, 3, state_dim, state_dim), dtype=tf.float64),
             tf.constant(np.array(best_ind[3*state_dim + 3*state_dim**2:]).reshape(1, 3, state_dim, state_dim, state_dim), dtype=tf.float64)),
            initial_state, steps=int(sim_time/dt) + 1, dt=dt
        )
        
        policy_file = f'policies/best_policy_gen_{gen}.json'
        with open(policy_file, 'w') as f:
            json.dump({'type': 'cubic', 'params': best_policy, 'cost': float(best_cost)}, f, indent=2)
        
        traj_file = f'policies/best_trajectory_gen_{gen}.csv'
        with open(traj_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'x', 'x_dot'] + [f'cos_theta{i+1}' for i in range(optimizer.num_links)] + 
                           [f'sin_theta{i+1}' for i in range(optimizer.num_links)] + 
                           [f'theta_dot{i+1}' for i in range(optimizer.num_links)])
            times = np.arange(0, sim_time + dt, dt)[:best_traj.shape[1]]  # Steps dim
            #print(f"best_traj shape: {best_traj.shape}, len(times): {len(times)}")  # Debug
            for t, state in zip(times, best_traj[0].numpy()):  # Index batch dim 0
                #print(f"State shape: {state.shape}, State: {state}")  # Debug
                row = [float(t), float(state[1]), float(state[2])]
                for i in range(optimizer.num_links):
                    row.extend([float(state[3 + i*3]), float(state[4 + i*3]), float(state[5 + i*3])])
                #print(f"CSV row: {row}")  # Debug
                writer.writerow(row)
        
        print(f"Gen {gen}: Best cost = {best_cost:.2f}, took {time.time() - start_gen:.2f} seconds")

    valid_hof_entries = [hof[0] for hof in hofs if hof and hof[0].fitness.valid]
    if not valid_hof_entries:
        raise ValueError("No valid individuals in final Hall of Fame")
    best_policy = min(valid_hof_entries, key=lambda x: x.fitness.values[0])
    final_policy_dict = {
        'type': 'cubic',
        'params': {
            'a': np.array(best_policy[:3*state_dim]).reshape(3, state_dim).tolist(),
            'b': np.array(best_policy[3*state_dim:3*state_dim + 3*state_dim**2]).reshape(3, state_dim, state_dim).tolist(),
            'c': np.array(best_policy[3*state_dim + 3*state_dim**2:]).reshape(3, state_dim, state_dim, state_dim).tolist()
        }
    }
    with open('policies/final_policy.json', 'w') as f:
        json.dump(final_policy_dict, f, indent=2)
    print(f"Final policy saved to policies/final_policy.json")
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
    initial_state = tf.constant([1.0 / args.sample_rate, args.x, args.x_dot], dtype=tf.float64)
    for i in range(1, args.num_links + 1):
        angle = getattr(args, f'angle{i}')
        angle_dot = getattr(args, f'angle_dot{i}')
        initial_state = tf.concat([initial_state, tf.constant([np.cos(angle), np.sin(angle), angle_dot], dtype=tf.float64)], axis=0)
    
    print(f"Running optimization with initial state: {initial_state.numpy()}, lengths: {lengths}")
    best_policy = optimize(initial_state=initial_state, num_links=args.num_links, 
                          sim_time=args.sim_time, sample_rate=args.sample_rate)