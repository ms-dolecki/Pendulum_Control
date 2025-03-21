import numpy as np
from deap import base, creator, tools, algorithms
import gpflow
import tensorflow as tf
import json
import argparse
import random
import time
import os
import csv
from pendulum_sim_grok import PendulumSim
from multiprocessing import Pool

# Disable GPU to avoid CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set default float type for GPflow
gpflow.config.set_default_float(tf.float32)

# Define DEAP creator classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

class PolicyOptimizer:
    def __init__(self, num_links=1):
        self.num_links = num_links
        self.state_dim = 1 + 2 + 3 * num_links  # time, x, x_dot, plus 3 per link (cos, sin, theta_dot)
        self.output_dim = 2 + 3 * num_links  # x, x_dot, cos_theta, sin_theta, theta_dot (no time)
        # Initial data for GP model
        self.X_cache = np.array([[0.01, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                 [0.01, 2.95, 1.0, 0.7, 0.7, 2.0, 1.0]], dtype=np.float32)
        self.y_cache = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.01, 0.0, -0.1, 0.1, 0.2]], dtype=np.float32)
        self.X_mean = np.mean(self.X_cache[:, 1:], axis=0)
        self.X_std = np.std(self.X_cache[:, 1:], axis=0) + 1e-6
        # Define composite kernel
        rbf = gpflow.kernels.SquaredExponential(lengthscales=0.5 * tf.ones(self.state_dim, dtype=tf.float32))
        matern = gpflow.kernels.Matern52(lengthscales=0.5 * tf.ones(self.state_dim, dtype=tf.float32))
        white = gpflow.kernels.White(variance=0.01)
        kernel = rbf + matern + white
        X_normalized = (self.X_cache[:, 1:] - self.X_mean) / self.X_std
        num_inducing = 100  # Fixed number of inducing points
        if X_normalized.shape[0] < num_inducing:
            indices = np.random.choice(X_normalized.shape[0], num_inducing, replace=True)
        else:
            indices = np.random.choice(X_normalized.shape[0], num_inducing, replace=False)
        inducing_variable = X_normalized[indices]
        self.model = gpflow.models.SVGP(kernel=kernel, likelihood=gpflow.likelihoods.Gaussian(),
                                        inducing_variable=inducing_variable, num_data=X_normalized.shape[0])
        self.model.likelihood.variance.assign(0.1)
        # Initial training to set output dimension
        data = (tf.constant(X_normalized, dtype=tf.float32), tf.constant(self.y_cache, dtype=tf.float32))
        optimizer = gpflow.optimizers.Scipy()
        optimizer.minimize(lambda: self.model.training_loss(data), self.model.trainable_variables, options={'maxiter': 50})

    def predict_action(self, states, params):
        """Predict actions based on cubic policy parameters."""
        a, b, c = params
        p = states
        p2 = np.einsum('bi,bj->bij', p, p)
        p3 = np.einsum('bij,bk->bijk', p2, p)
        term1 = np.sum(np.maximum(0, a[:, 0, :] * p + a[:, 1, :]) * a[:, 2, :], axis=1)
        term2 = np.sum(np.maximum(0, b[:, 0, :, :] * p2 + b[:, 1, :, :]) * b[:, 2, :, :], axis=(1, 2))
        term3 = np.sum(np.maximum(0, c[:, 0, :, :, :] * p3 + c[:, 1, :, :, :]) * c[:, 2, :, :, :], axis=(1, 2, 3))
        actions = term1 + term2 + term3
        return np.clip(actions, -5.0, 5.0)

    def predict_trajectory_batch(self, params_batch, initial_state, steps=10, dt=0.01):
        """Predict trajectories for a batch of policies."""
        batch_size = params_batch[0].shape[0]
        state = np.tile(initial_state[None, :], [batch_size, 1]).astype(np.float32)  # Shape: (batch_size, 6)
        trajectories = [state.copy()]
        for _ in range(steps - 1):
            actions = self.predict_action(state, params_batch)  # Shape: (batch_size,)
            X = np.hstack([state[:, 1:], actions[:, None]])  # Shape: (batch_size, 6)
            X_normalized = (X - self.X_mean) / self.X_std
            mean, _ = self.model.predict_f(tf.constant(X_normalized, dtype=tf.float32))  # Should be (batch_size, 5)
            print(f"mean shape from predict_f: {mean.shape}")
            next_state_diff = mean.numpy()  # Shape: (batch_size, 5)
            # Pad with zero for time to match state shape
            state_diff = np.hstack([np.zeros((batch_size, 1), dtype=np.float32), next_state_diff])  # Shape: (batch_size, 6)
            state = state + state_diff  # Shape: (batch_size, 6)
            x = state[:, 1]
            x_dot = state[:, 2]
            state[:, 2] = np.where(np.abs(x) > 3, 0.0, x_dot)
            state[:, 1] = np.where(np.abs(x) > 3, np.clip(x, -3, 3), x)
            trajectories.append(state.copy())
        return np.stack(trajectories, axis=1)

    def evaluate_policy_batch(self, individuals, initial_state):
        """Evaluate a batch of policies based on predicted trajectories."""
        batch_size = individuals.shape[0]
        a = individuals[:, :3*self.state_dim].reshape(batch_size, 3, self.state_dim)
        b = individuals[:, 3*self.state_dim:3*self.state_dim + 3*self.state_dim**2].reshape(batch_size, 3, self.state_dim, self.state_dim)
        c = individuals[:, 3*self.state_dim + 3*self.state_dim**2:].reshape(batch_size, 3, self.state_dim, self.state_dim, self.state_dim)
        params_batch = (a, b, c)
        traj = self.predict_trajectory_batch(params_batch, initial_state, steps=10)
        angles = [np.arctan2(traj[:, :, 4 + i*3], traj[:, :, 3 + i*3]) for i in range(self.num_links)]
        angle_dots = [traj[:, :, 5 + i*3] for i in range(self.num_links)]
        energy = 0.5 * angle_dots[0]**2 + np.cos(angles[0])
        energy_cost = np.mean(np.maximum(1.0 - energy, 0.0)**2, axis=1) * 60.0
        angle_cost = np.mean(angles[0]**2, axis=1) * 100.0
        x_cost = np.mean(traj[:, :, 1]**2, axis=1) * 5.0
        x_dot_cost = np.mean(traj[:, :, 2]**2, axis=1) * 0.5
        vel_cost = np.mean(angle_dots[0]**2, axis=1) * 1.0
        total_cost = energy_cost + angle_cost + x_cost + x_dot_cost + vel_cost
        return total_cost

    def run_simulation(self, args):
        """Run a single simulation for an individual."""
        individual, idx, initial_state, sim_time, dt, subpop_id = args
        policy_file = f'policies/policy_subpop_{subpop_id}_ind_{idx}.json' if subpop_id else f'policies/policy_ind_{idx}.json'
        sim_done_file = f'policies/sim_done_subpop_{subpop_id}_ind_{idx}.txt' if subpop_id else f'policies/sim_done_ind_{idx}.txt'
        sim_data_file = f'policies/sim_data_subpop_{subpop_id}_ind_{idx}.json' if subpop_id else f'policies/sim_data_ind_{idx}.json'
        print(f"Starting simulation {idx} for subpop {subpop_id} with initial state: {initial_state}")
        a = np.array(individual[:3*self.state_dim]).reshape(3, self.state_dim)
        b = np.array(individual[3*self.state_dim:3*self.state_dim + 3*self.state_dim**2]).reshape(3, self.state_dim, self.state_dim)
        c = np.array(individual[3*self.state_dim + 3*self.state_dim**2:]).reshape(3, self.state_dim, self.state_dim, self.state_dim)
        policy = {'type': 'cubic', 'params': {'a': a.tolist(), 'b': b.tolist(), 'c': c.tolist()}}
        with open(policy_file, 'w') as f:
            json.dump(policy, f)
        sim = PendulumSim(lengths=[0.5] * self.num_links, masses=[1.0] * self.num_links)
        if os.path.exists(sim_done_file):
            os.remove(sim_done_file)
        start_time = time.time()
        sim.simulate(t_span=(0, sim_time), initial_state=initial_state, dt=dt, 
                     sim_data_file=sim_data_file, sim_done_file=sim_done_file)
        print(f"Simulation {idx} for subpop {subpop_id} completed in {time.time() - start_time:.2f}s")
        with open(sim_data_file, 'r') as f:
            data = json.load(f)
        states = np.array(data['states'], dtype=np.float32)  # Shape: (timesteps, 6)
        actions = np.array(data['actions'], dtype=np.float32)  # Shape: (timesteps,)
        X_new = np.hstack([states[:-1], actions[:-1, None]])  # Shape: (timesteps-1, 6)
        y_new = states[1:, 1:] - states[:-1, 1:]  # Shape: (timesteps-1, 5)
        print(f"y_new shape after simulation {idx}: {y_new.shape}")
        return X_new, y_new

    def run_sim_and_update(self, individuals, initial_state, sim_time=10.0, dt=0.1, generation=0, is_initial=False, subpop_id=None):
        """Run simulations sequentially and update the GP model."""
        results = []
        for idx, individual in enumerate(individuals):
            args = (individual, idx, initial_state, sim_time, dt, subpop_id)
            result = self.run_simulation(args)
            results.append(result)
        print(f"All {len(individuals)} simulations completed for generation {generation}, subpop {subpop_id}")
        
        for X_new, y_new in results:
            x_vals = X_new[:, 1]
            mask_inner = np.abs(x_vals) < 2.95
            mask_outer = np.abs(x_vals) >= 2.95
            X_inner, y_inner = X_new[mask_inner], y_new[mask_inner]
            X_outer, y_outer = X_new[mask_outer], y_new[mask_outer]
            if X_outer.shape[0] > 0:
                num_outer_samples = max(1, int(0.01 * X_outer.shape[0]))
                idx_outer = np.random.choice(X_outer.shape[0], num_outer_samples, replace=False)
                X_outer, y_outer = X_outer[idx_outer], y_outer[idx_outer]
            X_new = np.vstack([X_inner, X_outer]) if X_outer.shape[0] > 0 else X_inner
            y_new = np.vstack([y_inner, y_outer]) if y_outer.shape[0] > 0 else y_inner
            _, unique_idx = np.unique(np.round(X_new, 3), axis=0, return_index=True)
            X_new, y_new = X_new[unique_idx], y_new[unique_idx]
            if X_new.shape[0] > 10:
                idx = np.random.choice(X_new.shape[0], 10, replace=False)
                X_new, y_new = X_new[idx], y_new[idx]
            self.X_cache = np.vstack([self.X_cache, X_new])
            self.y_cache = np.vstack([self.y_cache, y_new])
            if self.X_cache.shape[0] > 5000:
                idx = np.random.choice(self.X_cache.shape[0], 5000, replace=False)
                self.X_cache = self.X_cache[idx]
                self.y_cache = self.y_cache[idx]
            self.X_mean = np.mean(self.X_cache[:, 1:], axis=0)
            self.X_std = np.std(self.X_cache[:, 1:], axis=0) + 1e-6
        
        X_normalized = (self.X_cache[:, 1:] - self.X_mean) / self.X_std
        print(f"X_normalized shape: {X_normalized.shape}, y_cache shape: {self.y_cache.shape}")
        num_inducing = 100  # Match the fixed number from __init__
        if X_normalized.shape[0] < num_inducing:
            indices = np.random.choice(X_normalized.shape[0], num_inducing, replace=True)
        else:
            indices = np.random.choice(X_normalized.shape[0], num_inducing, replace=False)
        inducing_variable = X_normalized[indices]
        self.model.inducing_variable.Z.assign(inducing_variable)
        self.model.num_data = X_normalized.shape[0]
        
        # Train with current data
        data = (tf.constant(X_normalized, dtype=tf.float32), tf.constant(self.y_cache, dtype=tf.float32))
        loss_closure = lambda: self.model.training_loss(data)
        optimizer = gpflow.optimizers.Scipy()
        optimizer.minimize(loss_closure, self.model.trainable_variables, options={'maxiter': 50})

def seed_individual(ind_class, state_dim, seed_type):
    """Create seeded individuals with specific initial behaviors."""
    a = np.zeros((3, state_dim), dtype=np.float32)
    b = np.zeros((3, state_dim, state_dim), dtype=np.float32)
    c = np.zeros((3, state_dim, state_dim, state_dim), dtype=np.float32)
    if seed_type == 'zero':
        a[0, 1] = 3.0
    elif seed_type == 'linear':
        a[0, 2] = 4.0
        a[1, 1] = -2.0
    elif seed_type == 'oscillatory':
        a[0, 4] = 5.0
        b[0, 5, 5] = 1.0
    a += np.random.uniform(-1.0, 1.0, a.shape)
    b += np.random.uniform(-0.2, 0.2, b.shape)
    c += np.random.uniform(-0.02, 0.02, c.shape)
    return ind_class(np.concatenate([a.flatten(), b.flatten(), c.flatten()]))

def init_individual(ind_class, state_dim):
    """Initialize a random individual."""
    a = np.random.uniform(-5, 5, size=3*state_dim)
    b = np.random.uniform(-0.5, 0.5, size=3*state_dim*state_dim)
    c = np.random.uniform(-0.05, 0.05, size=3*state_dim*state_dim*state_dim)
    return ind_class(np.concatenate([a, b, c]))

def custom_mutation(individual, indpb, state_dim, generation, max_gens):
    """Custom mutation with generation-dependent scaling."""
    scale = 1.0 - generation / max_gens
    sigma_a = 3.0 * scale
    sigma_b = 0.3 * scale
    sigma_c = 0.03 * scale
    for i in range(3*state_dim):
        if np.random.random() < indpb:
            individual[i] = np.clip(individual[i] + np.random.normal(0, sigma_a), -5, 5)
    for i in range(3*state_dim, 3*state_dim + 3*state_dim**2):
        if np.random.random() < indpb:
            individual[i] = np.clip(individual[i] + np.random.normal(0, sigma_b), -0.5, 0.5)
    for i in range(3*state_dim + 3*state_dim**2, len(individual)):
        if np.random.random() < indpb:
            individual[i] = np.clip(individual[i] + np.random.normal(0, sigma_c), -0.05, 0.05)
    return individual,

def migrate(subpops, migration_rate=0.1):
    """Migrate individuals between subpopulations."""
    for i in range(len(subpops)):
        for j in range(i + 1, len(subpops)):
            num_migrate = int(len(subpops[i]) * migration_rate)
            migrants_i = random.sample(subpops[i], num_migrate)
            migrants_j = random.sample(subpops[j], num_migrate)
            subpops[i][:] = [ind for ind in subpops[i] if ind not in migrants_i] + migrants_j
            subpops[j][:] = [ind for ind in subpops[j] if ind not in migrants_j] + migrants_i

def optimize(initial_state, num_links, sim_time=10.0, sample_rate=10.0, pop_size=5000, tourney_size=5):
    """Main optimization function using genetic algorithm."""
    dt = 1.0 / sample_rate
    state_dim = 1 + 2 + 3 * num_links
    optimizer = PolicyOptimizer(num_links=num_links)
    os.makedirs('policies', exist_ok=True)
    
    # Initialize subpopulations
    num_subpops = 5
    subpop_size = pop_size // num_subpops
    seed_types = ['zero', 'linear', 'oscillatory', 'zero', 'linear']
    subpops, toolboxes, hofs = [], [], []
    for i in range(num_subpops):
        toolbox = base.Toolbox()
        if i < 3:
            toolbox.register("seed_ind", seed_individual, creator.Individual, state_dim=state_dim, seed_type=seed_types[i])
            toolbox.register("population", tools.initRepeat, list, toolbox.seed_ind, n=1)
            pop = toolbox.population() + tools.initRepeat(list, lambda: init_individual(creator.Individual, state_dim), subpop_size - 1)
        else:
            toolbox.register("individual", init_individual, creator.Individual, state_dim=state_dim)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            pop = toolbox.population(n=subpop_size)
        toolbox.register("evaluate", lambda inds: optimizer.evaluate_policy_batch(np.array(inds, dtype=np.float32), initial_state))
        toolbox.register("mate", tools.cxBlend, alpha=0.5 + 0.2 * i)
        toolbox.register("mutate", custom_mutation, indpb=0.5, state_dim=state_dim, generation=0, max_gens=200)
        toolbox.register("select", tools.selTournament, tournsize=tourney_size)
        subpops.append(pop)
        toolboxes.append(toolbox)
        hofs.append(tools.HallOfFame(3))
    
    # Initial simulations
    print("Running initial simulations with 20 random policies")
    start_time = time.time()
    initial_individuals = [subpops[i % num_subpops][i // num_subpops] for i in range(20)]
    optimizer.run_sim_and_update(initial_individuals, initial_state, sim_time=sim_time, dt=dt, generation=0, is_initial=True)
    print(f"Initial sims took {time.time() - start_time:.2f} seconds")
    
    # Assign initial fitness
    for i, (pop, toolbox, hof) in enumerate(zip(subpops, toolboxes, hofs)):
        fitnesses = toolbox.evaluate(np.array(pop, dtype=np.float32))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = (fit,)
        hof.update(pop)
    
    # Evolution loop
    for gen in range(200):
        for i, (pop, toolbox, hof) in enumerate(zip(subpops, toolboxes, hofs)):
            toolbox.register("mutate", custom_mutation, indpb=0.5, state_dim=state_dim, generation=gen, max_gens=200)
            offspring = toolbox.select(pop, len(pop))
            offspring = algorithms.varAnd(offspring, toolbox, cxpb=0.5, mutpb=0.9)
            fitnesses = toolbox.evaluate(np.array(offspring, dtype=np.float32))
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = (fit,)
            pop[:] = offspring
            hof.update(pop)
        
        if gen % 10 == 0 and gen > 0:
            migrate(subpops)
        
        if gen % 5 == 0:
            best_subpop_inds = [min(hof, key=lambda x: x.fitness.values[0]) for hof in hofs if hof]
            if best_subpop_inds:
                optimizer.run_sim_and_update(best_subpop_inds, initial_state, sim_time=sim_time, dt=dt, generation=gen, subpop_id=f"gen_{gen}")
                for i, best_ind in enumerate(best_subpop_inds):
                    fitness = best_ind.fitness.values[0]
                    recalc = optimizer.evaluate_policy_batch(np.array([best_ind], dtype=np.float32), initial_state)[0]
                    print(f"Gen {gen}: Subpop {i}: HoF={fitness:.2f}, Recalc={recalc:.2f}")
                    with open(f'policies/best_traj_gen_{gen}_subpop_{i}.csv', 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['time', 'x', 'x_dot', 'cos_theta1', 'sin_theta1', 'theta_dot1'])
                        if os.path.exists('policies/trajectory_data.csv'):
                            with open('policies/trajectory_data.csv', 'r') as traj_file:
                                reader = csv.reader(traj_file)
                                for row in reader:
                                    writer.writerow(row)
                    best_params = (
                        np.array(best_ind[:3*state_dim]).reshape(1, 3, state_dim),
                        np.array(best_ind[3*state_dim:3*state_dim + 3*state_dim**2]).reshape(1, 3, state_dim, state_dim),
                        np.array(best_ind[3*state_dim + 3*state_dim**2:]).reshape(1, 3, state_dim, state_dim, state_dim)
                    )
                    pred_traj = optimizer.predict_trajectory_batch(best_params, initial_state)
                    with open(f'policies/pred_traj_gen_{gen}_subpop_{i}.csv', 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['time', 'x', 'x_dot', 'cos_theta1', 'sin_theta1', 'theta_dot1'])
                        for t, state in enumerate(pred_traj[0]):
                            writer.writerow([t * dt] + state.tolist())
    
    # Save final best policy
    final_hof_entries = [hof[0] for hof in hofs if hof]
    best_policy = min(final_hof_entries, key=lambda x: x.fitness.values[0])
    final_policy_dict = {
        'type': 'cubic',
        'params': {
            'a': np.array(best_policy[:3*state_dim]).reshape(3, state_dim).tolist(),
            'b': np.array(best_policy[3*state_dim:3*state_dim + 3*state_dim**2]).reshape(3, state_dim, state_dim).tolist(),
            'c': np.array(best_policy[3*state_dim + 3*state_dim**2:]).reshape(3, state_dim, state_dim, state_dim).tolist()
        }
    }
    with open('policies/final_policy.json', 'w') as f:
        json.dump(final_policy_dict, f)
    print("Final policy saved to policies/final_policy.json")
    return best_policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize multi-link pendulum policy")
    parser.add_argument('--sim_time', type=float, default=10.0)
    parser.add_argument('--sample_rate', type=float, default=10.0)
    parser.add_argument('--num_links', type=int, default=1)
    parser.add_argument('--x', type=float, default=0.0)
    parser.add_argument('--x_dot', type=float, default=0.0)
    parser.add_argument('--pop_size', type=int, default=5000)
    parser.add_argument('--tourney_size', type=int, default=20)
    for i in range(1, 11):
        parser.add_argument(f'--length{i}', type=float, default=0.5)
        parser.add_argument(f'--angle{i}', type=float, default=np.pi)
        parser.add_argument(f'--angle_dot{i}', type=float, default=0.0)
    args = parser.parse_args()
    initial_state = np.array([1.0 / args.sample_rate, args.x, args.x_dot], dtype=np.float32)
    for i in range(1, args.num_links + 1):
        angle = getattr(args, f'angle{i}')
        angle_dot = getattr(args, f'angle_dot{i}')
        initial_state = np.concatenate([initial_state, [np.cos(angle), np.sin(angle), angle_dot]], dtype=np.float32)
    print(f"Running optimization with initial state: {initial_state}")
    best_policy = optimize(initial_state=initial_state, num_links=args.num_links, 
                          sim_time=args.sim_time, sample_rate=args.sample_rate, 
                          pop_size=args.pop_size, tourney_size=args.tourney_size)