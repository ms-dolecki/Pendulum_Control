import numpy as np
import gpflow
import tensorflow as tf
import json
import time
import os
import csv
from pendulum_sim_grok import PendulumSim
from deap import base, creator, tools, algorithms

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpflow.config.set_default_float(tf.float32)

# DEAP creator classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

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
        a = np.random.uniform(-100, 100, (3, self.state_dim-1))  # -100, 100
        b = np.random.uniform(-10, 10, (3, self.state_dim-1, self.state_dim-1))  # -10, 10
        c = np.random.uniform(-1, 1, (3, self.state_dim-1, self.state_dim-1, self.state_dim-1))  # -1, 1
        return (tf.constant(a, dtype=tf.float32), 
                tf.constant(b, dtype=tf.float32), 
                tf.constant(c, dtype=tf.float32))

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 5], dtype=tf.float32)])
    def predict_action(self, states):
        a, b, c = self.policy_params
        batch_size = tf.shape(states)[0]
        a = tf.ensure_shape(a, [None, 3, self.state_dim-1])
        b = tf.ensure_shape(b, [None, 3, self.state_dim-1, self.state_dim-1])
        c = tf.ensure_shape(c, [None, 3, self.state_dim-1, self.state_dim-1, self.state_dim-1])
        a = a[:batch_size]
        b = b[:batch_size]
        c = c[:batch_size]
        p = states  # Shape: [batch_size, state_dim-1]
        p2 = tf.einsum('bi,bj->bij', p, p)
        p3 = tf.einsum('bij,bk->bijk', p2, p)
        term1 = tf.reduce_sum(tf.maximum(0., a[:, 0, :] * p + a[:, 1, :]) * a[:, 2, :], axis=1)
        term2 = tf.reduce_sum(tf.maximum(0., b[:, 0, :, :] * p2 + b[:, 1, :, :]) * b[:, 2, :, :], axis=[1, 2])
        term3 = tf.reduce_sum(tf.maximum(0., c[:, 0, :, :, :] * p3 + c[:, 1, :, :, :]) * c[:, 2, :, :, :], axis=[1, 2, 3])
        actions = term1 + term2 + term3  # Shape: [batch_size]
        return tf.clip_by_value(actions, -10.0, 10.0)  # Restored to -5, 5

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
        subset_size = min(1000, X_data.shape[0])
        indices = np.random.choice(X_data.shape[0], subset_size, replace=False)
        X_subset = X_data[indices].astype(np.float32)
        y_subset = y_data[indices].astype(np.float32)

        self.X_mean = tf.constant(np.mean(X_subset, axis=0), dtype=tf.float32)
        self.X_std = tf.constant(np.std(X_subset, axis=0) + 1e-6, dtype=tf.float32)
        mask = self.X_std > 1e-4
        self.mask = tf.where(mask)[:, 0]  # Tensor of indices
        X_subset_clean = X_subset[:, mask]
        self.X_mean = self.X_mean[mask]
        self.X_std = self.X_std[mask]
        X_normalized = (X_subset_clean - self.X_mean.numpy()) / self.X_std.numpy()

        self.y_mean = tf.constant(np.mean(y_subset, axis=0), dtype=tf.float32)
        self.y_std = tf.constant(np.std(y_subset, axis=0) + 1e-6, dtype=tf.float32)
        y_normalized = (y_subset - self.y_mean.numpy()) / self.y_std.numpy()

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

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 6], dtype=tf.float32)])
    def predict_trajectory(self, initial_states):
        batch_size = tf.shape(initial_states)[0]
        states = tf.TensorArray(dtype=tf.float32, size=self.steps, dynamic_size=False)
        states = states.write(0, initial_states)  # Shape: [batch_size, 6]

        @tf.function
        def body(i, prev_states, states_ta):
            actions = self.predict_action(prev_states[:, 1:])  # Shape: [batch_size]
            X_full = tf.concat([prev_states, actions[:, tf.newaxis]], axis=1)  # Shape: [batch_size, 7]
            X = tf.gather(X_full, self.mask, axis=1)  # Shape: [batch_size, len(mask)]
            X_normalized = (X - self.X_mean) / self.X_std  # Shape: [batch_size, len(mask)]
            mean = self.model.predict_f(X_normalized)[0]  # Shape: [batch_size, 5]
            next_states = mean * self.y_std + self.y_mean  # Shape: [batch_size, 5]
            # Limited debug: Print only for first 5 individuals, first 10 steps
            tf.cond(i < 10, lambda: tf.print("Next states (x, x_dot, first 5):", next_states[:5, :2]), lambda: tf.no_op())
            time_batch = tf.ones([batch_size, 1], dtype=tf.float32) * self.dt  # Shape: [batch_size, 1]
            states_new = tf.concat([time_batch, next_states], axis=1)  # Shape: [batch_size, 6]
            x = states_new[:, 1]
            x_dot = states_new[:, 2]
            tf.cond(i < 10, lambda: tf.print("x, x_dot before clipping (first 5):", x[:5], x_dot[:5]), lambda: tf.no_op())
            zeros_batch = tf.zeros([batch_size, 1], dtype=tf.float32)
            clip_batch = tf.clip_by_value(states_new[:, 1:2], -10, 10)
            states_new_xdot = tf.concat([states_new[:, :2], zeros_batch, states_new[:, 3:]], axis=1)
            states_new = tf.where(tf.abs(x)[:, tf.newaxis] > 10, states_new_xdot, states_new)
            states_new_x = tf.concat([states_new[:, :1], clip_batch, states_new[:, 2:]], axis=1)
            states_new = tf.where(tf.abs(x)[:, tf.newaxis] > 10, states_new_x, states_new)
            tf.cond(i < 10, lambda: tf.print("States new (first 5):", states_new[:5]), lambda: tf.no_op())
            states_ta = states_ta.write(i, states_new)
            return i + 1, states_new, states_ta

        _, _, final_states = tf.while_loop(
            cond=lambda i, *_: i < self.steps,
            body=body,
            loop_vars=(tf.constant(1), initial_states, states),
            shape_invariants=(tf.TensorShape([]), tf.TensorShape([None, 6]), None)
        )

        return final_states.stack()  # Shape: [steps, batch_size, 6]

class PolicyOptimizer:
    def __init__(self, tester, initial_state, pop_size=10000, generations=50):
        self.tester = tester
        self.initial_state = initial_state
        self.pop_size = pop_size
        self.generations = generations
        self.state_dim = tester.state_dim - 1  # Exclude time for policy
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.init_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_policy_batch)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", self.custom_mutation, indpb=0.9)
        self.toolbox.register("select", tools.selTournament, tournsize=100)

    def init_individual(self):
        a = np.random.uniform(-100, 100, 3 * self.state_dim)  # -100, 100
        b = np.random.uniform(-10, 10, 3 * self.state_dim * self.state_dim)  # -10, 10
        c = np.random.uniform(-1, 1, 3 * self.state_dim * self.state_dim * self.state_dim)  # -1, 1
        return creator.Individual(np.concatenate([a, b, c]))

    def custom_mutation(self, individual, indpb):
        for i in range(3 * self.state_dim):
            if np.random.random() < indpb:
                individual[i] = np.clip(individual[i] + np.random.normal(0, 10.0), -100, 100)
        for i in range(3 * self.state_dim, 3 * self.state_dim + 3 * self.state_dim**2):
            if np.random.random() < indpb:
                individual[i] = np.clip(individual[i] + np.random.normal(0, 1.0), -10, 10)
        for i in range(3 * self.state_dim + 3 * self.state_dim**2, len(individual)):
            if np.random.random() < indpb:
                individual[i] = np.clip(individual[i] + np.random.normal(0, 0.1), -1, 1)
        return individual,

    def evaluate_policy_batch(self, individuals):
        start_time = time.time()
        print("Entering evaluate_policy_batch")
        batch_size = len(individuals)
        print(f"Batch size: {batch_size}")
        a = tf.constant(np.array([ind[:3*self.state_dim] for ind in individuals]).reshape(batch_size, 3, self.state_dim), dtype=tf.float32)
        b = tf.constant(np.array([ind[3*self.state_dim:3*self.state_dim + 3*self.state_dim**2] for ind in individuals]).reshape(batch_size, 3, self.state_dim, self.state_dim), dtype=tf.float32)
        c = tf.constant(np.array([ind[3*self.state_dim + 3*self.state_dim**2:] for ind in individuals]).reshape(batch_size, 3, self.state_dim, self.state_dim, self.state_dim), dtype=tf.float32)
        self.tester.policy_params = (a, b, c)
        initial_states = tf.tile(self.initial_state[tf.newaxis, :], [batch_size, 1])  # Shape: [batch_size, 6]
        print("Calling predict_trajectory")
        traj_start = time.time()
        trajectories = self.tester.predict_trajectory(initial_states)  # Shape: [steps, batch_size, 6]
        print(f"Predict trajectory took {time.time() - traj_start:.2f}s")
        trajectories = tf.transpose(trajectories, [1, 0, 2])  # Shape: [batch_size, steps, 6]
        
        # Fitness: Minimize deviation from upright (cosθ = 1, sinθ = 0), penalize energy and x deviation
        angles = tf.atan2(trajectories[:, :, 4], trajectories[:, :, 3])  # Shape: [batch_size, steps]
        angle_dots = trajectories[:, :, 5]  # Shape: [batch_size, steps]
        energy = 0.5 * angle_dots**2 + tf.cos(angles)  # Shape: [batch_size, steps]
        energy_cost = tf.reduce_mean(tf.maximum(1.0 - energy, 0.0), axis=1) * 30.0
        angle_cost = tf.reduce_mean(angles**2, axis=1) * 100.0
        x_cost = tf.reduce_mean(trajectories[:, :, 1]**2, axis=1) * 5.0
        x_dot_cost = tf.reduce_mean(trajectories[:, :, 2]**2, axis=1) * 0.5
        vel_cost = tf.reduce_mean(angle_dots**2, axis=1) * 1.0
        total_cost = energy_cost + angle_cost + x_cost + x_dot_cost + vel_cost
        print("Costs (first 5):", total_cost[:5].numpy())
        print(f"Evaluate_policy_batch took {time.time() - start_time:.2f}s")
        return total_cost

    def optimize(self):
        print("Starting optimize")
        start_time = time.time()
        population = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        print(f"Population size: {len(population)}, creation took {time.time() - start_time:.2f}s")
        
        # Initial evaluation
        print("Evaluating initial population")
        fitnesses = self.evaluate_policy_batch(population).numpy()
        print(f"Initial fitnesses (min, max, mean): {fitnesses.min():.2f}, {fitnesses.max():.2f}, {fitnesses.mean():.2f}")
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = (fit,)
        hof.update(population)
        print("Initial population evaluated")

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("min", tf.reduce_min)
        # Evolution loop
        for gen in range(self.generations):
            gen_start = time.time()
            print(f"Generation {gen} starting")
            offspring = self.toolbox.select(population, len(population))
            #offspring = algorithms.varAnd(offspring, self.toolbox, cxpb=0.7, mutpb=0.9)
            offspring, log = algorithms.eaSimple(offspring, self.toolbox, cxpb=0.5, mutpb=0.5, 
                                     ngen=1, stats=self.stats, halloffame=hof, 
                                     verbose=False)
            print(f"Evaluating offspring, size: {len(offspring)}")
            fitnesses = self.evaluate_policy_batch(offspring).numpy()
            print(f"Offspring fitnesses (min, max, mean): {fitnesses.min():.2f}, {fitnesses.max():.2f}, {fitnesses.mean():.2f}")
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = (fit,)
            population[:] = offspring
            hof.update(population)
            print(f"Gen {gen}: Best fitness = {hof[0].fitness.values[0]:.2f}, took {time.time() - gen_start:.2f}s")

            # Save best trajectory every 5 generations
            if gen % 5 == 0:
                best_ind = hof[0]
                self.tester.policy_params = (
                    tf.constant(np.array(best_ind[:3*self.state_dim]).reshape(1, 3, self.state_dim), dtype=tf.float32),
                    tf.constant(np.array(best_ind[3*self.state_dim:3*self.state_dim + 3*self.state_dim**2]).reshape(1, 3, self.state_dim, self.state_dim), dtype=tf.float32),
                    tf.constant(np.array(best_ind[3*self.state_dim + 3*self.state_dim**2:]).reshape(1, 3, self.state_dim, self.state_dim, self.state_dim), dtype=tf.float32)
                )
                traj = self.tester.predict_trajectory(self.initial_state[None, :])  # Shape: [steps, 1, 6]
                with open(f'policies/best_traj_gen_{gen}.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['time', 'x', 'x_dot', 'cos_theta', 'sin_theta', 'theta_dot'])
                    for state in traj[:, 0]:
                        row = [float(x) for x in state.numpy()]
                        writer.writerow(row)

        # Save final best policy
        best_ind = hof[0]
        final_policy = {
            'type': 'cubic',
            'params': {
                'a': np.array(best_ind[:3*self.state_dim]).reshape(3, self.state_dim).tolist(),
                'b': np.array(best_ind[3*self.state_dim:3*self.state_dim + 3*self.state_dim**2]).reshape(3, self.state_dim, self.state_dim).tolist(),
                'c': np.array(best_ind[3*self.state_dim + 3*self.state_dim**2:]).reshape(3, self.state_dim, self.state_dim, self.state_dim).tolist()
            }
        }
        with open('policies/final_policy.json', 'w') as f:
            json.dump(final_policy, f)
        print("Final policy saved to policies/final_policy.json")
        return best_ind

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

    # Optimize policy with GA
    initial_state = tf.constant([0.0125, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=tf.float32)  # Start hanging down
    optimizer = PolicyOptimizer(tester, initial_state, pop_size=500, generations=50)
    print("Calling optimize")
    best_policy = optimizer.optimize()

    # Test final policy
    tester.policy_params = (
        tf.constant(np.array(best_policy[:3*(tester.state_dim-1)]).reshape(1, 3, tester.state_dim-1), dtype=tf.float32),
        tf.constant(np.array(best_policy[3*(tester.state_dim-1):3*(tester.state_dim-1) + 3*(tester.state_dim-1)**2]).reshape(1, 3, tester.state_dim-1, tester.state_dim-1), dtype=tf.float32),
        tf.constant(np.array(best_policy[3*(tester.state_dim-1) + 3*(tester.state_dim-1)**2:]).reshape(1, 3, tester.state_dim-1, tester.state_dim-1, tester.state_dim-1), dtype=tf.float32)
    )
    traj = tester.predict_trajectory(initial_state[None, :])  # Shape: [steps, 1, 6]
    with open('policies/final_trajectory.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'x', 'x_dot', 'cos_theta', 'sin_theta', 'theta_dot'])
        for state in traj[:, 0]:
            row = [float(x) for x in state.numpy()]
            writer.writerow(row)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        raise