import tensorflow as tf
import gpflow
import json
import time
import os
import csv
from pendulum_sim_grok import PendulumSim
from deap import base, creator, tools, algorithms

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Remove if GPU available
gpflow.config.set_default_float(tf.float32)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

class PendulumGPTester:
    def __init__(self, num_links=1, sim_time=10.0, dt=0.0125):
        self.num_links = num_links
        self.state_dim = 1 + 2 + 3 * num_links
        self.output_dim = 2 + 3 * num_links
        self.sim_time = sim_time
        self.dt = dt
        self.steps = int(sim_time / dt) + 1
        self.model = None
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        self.mask = None
        self.X_data = None
        self.y_data = None

    @tf.function
    def _generate_random_policy(self):
        a = tf.random.uniform([3, self.state_dim-1], minval=-20, maxval=20, dtype=tf.float32)
        b = tf.random.uniform([3, self.state_dim-1, self.state_dim-1], minval=-10, maxval=10, dtype=tf.float32)
        c = tf.random.uniform([3, self.state_dim-1, self.state_dim-1, self.state_dim-1], minval=-2, maxval=2, dtype=tf.float32)
        return a, b, c

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 5], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 3, 5], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 3, 5, 5], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 3, 5, 5, 5], dtype=tf.float32)
    ])
    def predict_action(self, states, a, b, c):
        batch_size = tf.shape(states)[0]
        p = states
        p2 = tf.einsum('bi,bj->bij', p, p)
        p3 = tf.einsum('bij,bk->bijk', p2, p)
        term1 = tf.reduce_sum(tf.maximum(0., a[:, 0, :] * p + a[:, 1, :]) * a[:, 2, :], axis=1)
        term2 = tf.reduce_sum(tf.maximum(0., b[:, 0, :, :] * p2 + b[:, 1, :, :]) * b[:, 2, :, :], axis=[1, 2])
        term3 = tf.reduce_sum(tf.maximum(0., c[:, 0, :, :, :] * p3 + c[:, 1, :, :, :]) * c[:, 2, :, :, :], axis=[1, 2, 3])
        actions = term1 + term2 + term3
        return tf.clip_by_value(actions, -10.0, 10.0)

    def run_simulation(self, initial_state, idx):
        policy_file = f'policies/policy_ind_{idx}.json'
        sim_done_file = f'policies/sim_done_ind_{idx}.txt'
        sim_data_file = f'policies/sim_data_ind_{idx}.json'
        os.makedirs('policies', exist_ok=True)

        policy_params = self._generate_random_policy()
        policy = {'type': 'cubic', 'params': {
            'a': policy_params[0].numpy().tolist()[0],
            'b': policy_params[1].numpy().tolist()[0],
            'c': policy_params[2].numpy().tolist()[0]
        }}
        with open(policy_file, 'w') as f:
            json.dump(policy, f)

        sim = PendulumSim(lengths=[0.5] * self.num_links, masses=[1.0] * self.num_links)
        sim.simulate(t_span=(0, self.sim_time/3), initial_state=initial_state, dt=self.dt, 
                     sim_data_file=sim_data_file, sim_done_file=sim_done_file, policy_file=policy_file)
        while not os.path.exists(sim_done_file):
            pass
        os.remove(sim_done_file)

        with open(sim_data_file, 'r') as f:
            data = json.load(f)
        states = tf.constant(data['states'], dtype=tf.float32)
        actions = tf.constant(data['actions'], dtype=tf.float32)
        delta_T = actions * 0 + self.dt
        X = tf.concat([delta_T[:-1, tf.newaxis], tf.concat([states[:-1, 1:], actions[:-1, tf.newaxis]], axis=1)], axis=1)
        y = states[1:, 1:]
        return X, y

    def run_simulation_with_policy(self, policy_params, initial_state, idx, gen):
        policy_file = f'policies/policy_gen_{gen}_ind_{idx}.json'
        sim_done_file = f'policies/sim_done_gen_{gen}_ind_{idx}.txt'
        sim_data_file = f'policies/sim_data_gen_{gen}_ind_{idx}.json'
        traj_file = f'policies/best_traj_sim_gen_{gen}.csv'
        os.makedirs('policies', exist_ok=True)

        a, b, c = policy_params
        policy = {'type': 'cubic', 'params': {
            'a': a.numpy().tolist()[0],
            'b': b.numpy().tolist()[0],
            'c': c.numpy().tolist()[0]
        }}
        with open(policy_file, 'w') as f:
            json.dump(policy, f)

        sim = PendulumSim(lengths=[0.5] * self.num_links, masses=[1.0] * self.num_links)
        sim.simulate(t_span=(0, self.sim_time), initial_state=initial_state, dt=self.dt, 
                     sim_data_file=sim_data_file, sim_done_file=sim_done_file, policy_file=policy_file)
        while not os.path.exists(sim_done_file):
            pass
        os.remove(sim_done_file)

        with open(sim_data_file, 'r') as f:
            data = json.load(f)
        states = tf.constant(data['states'], dtype=tf.float32)
        actions = tf.constant(data['actions'], dtype=tf.float32)

        with open(traj_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'x', 'x_dot', 'cos_theta', 'sin_theta', 'theta_dot'])
            for state in states:
                row = [float(x) for x in state.numpy()]
                writer.writerow(row)

        delta_T = actions * 0 + self.dt
        X = tf.concat([delta_T[:-1, tf.newaxis], tf.concat([states[:-1, 1:], actions[:-1, tf.newaxis]], axis=1)], axis=1)
        y = states[1:, 1:]
        
        if self.X_data is None:
            self.X_data = X
            self.y_data = y
        else:
            self.X_data = tf.concat([self.X_data, X], axis=0)
            self.y_data = tf.concat([self.y_data, y], axis=0)
        
        return X, y

    @tf.function
    def generate_random_initial_state(self):
        x = tf.random.uniform([], minval=-1.0, maxval=1.0, dtype=tf.float32)
        x_dot = tf.random.uniform([], minval=-1.0, maxval=1.0, dtype=tf.float32)
        theta = tf.random.uniform([], minval=0, maxval=2 * tf.constant(3.141592653589793, dtype=tf.float32), dtype=tf.float32)
        theta_dot = tf.random.uniform([], minval=-1.0, maxval=1.0, dtype=tf.float32)
        return tf.stack([self.dt, x, x_dot, tf.cos(theta), tf.sin(theta), theta_dot])

    def fit_gp(self):
        subset_size = min(2000, self.X_data.shape[0])
        indices = tf.random.shuffle(tf.range(self.X_data.shape[0]))[:subset_size]
        X_subset = tf.gather(self.X_data, indices)
        y_subset = tf.gather(self.y_data, indices)

        self.X_mean = tf.reduce_mean(X_subset, axis=0)
        self.X_std = tf.math.reduce_std(X_subset, axis=0) + 1e-6
        mask = self.X_std > 1e-4
        self.mask = tf.where(mask)[:, 0]
        X_subset_clean = tf.gather(X_subset, self.mask, axis=1)
        self.X_mean = tf.gather(self.X_mean, self.mask)
        self.X_std = tf.gather(self.X_std, self.mask)
        X_normalized = (X_subset_clean - self.X_mean) / self.X_std

        self.y_mean = tf.reduce_mean(y_subset, axis=0)
        self.y_std = tf.math.reduce_std(y_subset, axis=0) + 1e-6
        y_normalized = (y_subset - self.y_mean) / self.y_std

        kernel = gpflow.kernels.SquaredExponential(lengthscales=1.0)
        self.model = gpflow.models.GPR(
            data=(X_normalized, y_normalized),
            kernel=kernel
        )
        tf.print("GP model reset with data shape:", tf.shape(X_normalized))
        self.model.likelihood.variance.assign(1.0)
        self.model.likelihood.variance = gpflow.Parameter(
            self.model.likelihood.variance, transform=gpflow.utilities.positive(lower=1e-3)
        )

        optimizer = gpflow.optimizers.Scipy()
        optimizer.minimize(self.model.training_loss, self.model.trainable_variables, options={'maxiter': 500})

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 6], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 3, 5], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 3, 5, 5], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 3, 5, 5, 5], dtype=tf.float32)
    ])
    def predict_trajectory(self, initial_states, a, b, c):
        batch_size = tf.shape(initial_states)[0]
        states = tf.TensorArray(dtype=tf.float32, size=self.steps)
        states = states.write(0, initial_states)

        def body(i, prev_states, states_ta):
            actions = self.predict_action(prev_states[:, 1:], a, b, c)
            X_full = tf.concat([prev_states, actions[:, tf.newaxis]], axis=1)
            X = tf.gather(X_full, self.mask, axis=1)
            X_normalized = (X - self.X_mean) / self.X_std
            mean = self.model.predict_f(X_normalized)[0]
            next_states = mean * self.y_std + self.y_mean
            time_batch = tf.ones([batch_size, 1], dtype=tf.float32) * self.dt
            states_new = tf.concat([time_batch, next_states], axis=1)
            x = states_new[:, 1]
            zeros_batch = tf.zeros([batch_size, 1], dtype=tf.float32)
            clip_batch = tf.clip_by_value(states_new[:, 1:2], -3, 3)
            states_new_xdot = tf.concat([states_new[:, :2], zeros_batch, states_new[:, 3:]], axis=1)
            states_new = tf.where(tf.abs(x)[:, tf.newaxis] > 3, states_new_xdot, states_new)
            states_new_x = tf.concat([states_new[:, :1], clip_batch, states_new[:, 2:]], axis=1)
            states_new = tf.where(tf.abs(x)[:, tf.newaxis] > 3, states_new_x, states_new)
            states_ta = states_ta.write(i, states_new)
            return i + 1, states_new, states_ta

        _, _, final_states = tf.while_loop(
            cond=lambda i, *_: i < self.steps,
            body=body,
            loop_vars=(tf.constant(1), initial_states, states),
            shape_invariants=(tf.TensorShape([]), tf.TensorShape([None, 6]), None)
        )
        return final_states.stack()

class PolicyOptimizer:
    def __init__(self, tester, initial_state, pop_size=500, generations=50):
        self.tester = tester
        self.initial_state = initial_state
        self.pop_size = pop_size
        self.generations = generations
        self.state_dim = tester.state_dim - 1
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.init_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self.mate_tf)
        self.toolbox.register("mutate", self.custom_mutation_tf, indpb=0.9)
        self.toolbox.register("elite_mutate", self.elite_mutation_tf, indpb=0.5)
        self.toolbox.register("select", tools.selTournament, tournsize=50)
        self.toolbox.register("select_best", tools.selBest)

    def init_individual(self):
        a = tf.random.uniform([3 * self.state_dim], minval=-20, maxval=20, dtype=tf.float32)
        b = tf.random.uniform([3 * self.state_dim * self.state_dim], minval=-10, maxval=10, dtype=tf.float32)
        c = tf.random.uniform([3 * self.state_dim * self.state_dim * self.state_dim], minval=-2, maxval=2, dtype=tf.float32)
        params = tf.concat([a, b, c], axis=0)
        return creator.Individual(params.numpy().tolist())

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.float32)
    ])
    def mate_tf(self, ind1_tensor, ind2_tensor, alpha):
        delta = ind2_tensor - ind1_tensor
        ind1_new = ind1_tensor + alpha * delta
        ind2_new = ind2_tensor - alpha * delta
        return ind1_new, ind2_new

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.float32)
    ])
    def custom_mutation_tf(self, ind_tensor, indpb):
        a_size = 3 * self.state_dim
        b_size = 3 * self.state_dim * self.state_dim
        mask = tf.random.uniform([tf.shape(ind_tensor)[0]]) < indpb
        noise = tf.where(
            mask,
            tf.concat([
                tf.random.normal([a_size], 0, 15.0),
                tf.random.normal([b_size], 0, 1.5),
                tf.random.normal([tf.shape(ind_tensor)[0] - a_size - b_size], 0, 0.15)
            ], axis=0),
            tf.zeros([tf.shape(ind_tensor)[0]], dtype=tf.float32)
        )
        mutated = ind_tensor + noise
        a_clip = tf.clip_by_value(mutated[:a_size], -20, 20)
        b_clip = tf.clip_by_value(mutated[a_size:a_size + b_size], -10, 10)
        c_clip = tf.clip_by_value(mutated[a_size + b_size:], -2, 2)
        mutated = tf.concat([a_clip, b_clip, c_clip], axis=0)
        return mutated

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.float32)
    ])
    def elite_mutation_tf(self, ind_tensor, indpb):
        a_size = 3 * self.state_dim
        b_size = 3 * self.state_dim * self.state_dim
        mask = tf.random.uniform([tf.shape(ind_tensor)[0]]) < indpb
        noise = tf.where(
            mask,
            tf.concat([
                tf.random.normal([a_size], 0, 5.0),
                tf.random.normal([b_size], 0, 0.5),
                tf.random.normal([tf.shape(ind_tensor)[0] - a_size - b_size], 0, 0.05)
            ], axis=0),
            tf.zeros([tf.shape(ind_tensor)[0]], dtype=tf.float32)
        )
        mutated = ind_tensor + noise
        a_clip = tf.clip_by_value(mutated[:a_size], -20, 20)
        b_clip = tf.clip_by_value(mutated[a_size:a_size + b_size], -10, 10)
        c_clip = tf.clip_by_value(mutated[a_size + b_size:], -2, 2)
        mutated = tf.concat([a_clip, b_clip, c_clip], axis=0)
        return mutated

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32)
    ])
    def evaluate_policy_batch_tf(self, ind_tensor):
        batch_size = tf.shape(ind_tensor)[0]
        tf.print("evaluate_policy_batch_tf - ind_tensor shape:", tf.shape(ind_tensor))
        a = tf.reshape(ind_tensor[:, :3*self.state_dim], [batch_size, 3, self.state_dim])
        b = tf.reshape(ind_tensor[:, 3*self.state_dim:3*self.state_dim + 3*self.state_dim**2], 
                       [batch_size, 3, self.state_dim, self.state_dim])
        c = tf.reshape(ind_tensor[:, 3*self.state_dim + 3*self.state_dim**2:], 
                       [batch_size, 3, self.state_dim, self.state_dim, self.state_dim])
        initial_states = tf.tile(self.initial_state[tf.newaxis, :], [batch_size, 1])
        trajectories = self.tester.predict_trajectory(initial_states, a, b, c)
        trajectories = tf.transpose(trajectories, [1, 0, 2])
        
        angles = tf.atan2(trajectories[:, :, 4], trajectories[:, :, 3])
        angle_dots = trajectories[:, :, 5]
        energy = 0.5 * angle_dots**2 + tf.cos(angles)
        energy_cost = tf.reduce_mean(tf.maximum(1.0 - energy, 0.0)**2, axis=1) * 60.0
        angle_cost = tf.reduce_mean(angles**2, axis=1) * 100.0
        x_cost = tf.reduce_mean(trajectories[:, :, 1]**2, axis=1) * 50.0
        x_dot_cost = tf.reduce_mean(trajectories[:, :, 2]**2, axis=1) * 0.5
        vel_cost = tf.reduce_mean(angle_dots**2, axis=1) * 1.0
        total_cost = energy_cost + angle_cost + x_cost + x_dot_cost + vel_cost
        
        min_idx = tf.argmin(total_cost)
        if tf.reduce_min(total_cost) < 400:
            tf.print("True min policy costs - Energy:", energy_cost[min_idx], "Angle:", angle_cost[min_idx], 
                     "X:", x_cost[min_idx], "X_dot:", x_dot_cost[min_idx], "Vel:", vel_cost[min_idx], 
                     "Total:", total_cost[min_idx])
        
        tf.print("evaluate_policy_batch_tf - total_cost shape:", tf.shape(total_cost))
        return total_cost

    def evaluate_policy_batch(self, individuals):
        ind_tensor = tf.constant([ind[:] for ind in individuals], dtype=tf.float32)
        return self.evaluate_policy_batch_tf(ind_tensor)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32)
    ])
    def compute_gradient_batch_tf(self, base_params):
        num_elites = tf.shape(base_params)[0]
        num_params = tf.shape(base_params)[1]
        epsilon = tf.constant(0.01, dtype=tf.float32)
        
        tf.print("compute_gradient_batch_tf - base_params shape:", tf.shape(base_params))
        
        # Tile base_params to create 2325 individuals (5 elites * 465 params)
        tiled_params = tf.tile(base_params, [num_params, 1])  # [2325, 465]
        indices_plus = tf.tile(tf.range(num_params)[tf.newaxis, :], [num_elites, 1])
        elite_indices = tf.repeat(tf.range(num_elites), num_params)
        indices_plus = tf.stack([elite_indices, tf.reshape(indices_plus, [-1])], axis=1)  # [2325, 2]
        
        updates_plus = tf.fill([num_elites * num_params], epsilon)
        updates_minus = tf.fill([num_elites * num_params], -epsilon)
        
        perturbed_plus = tiled_params + tf.scatter_nd(indices_plus, updates_plus, tf.shape(tiled_params))
        perturbed_minus = tiled_params + tf.scatter_nd(indices_plus, updates_minus, tf.shape(tiled_params))
        
        tf.print("compute_gradient_batch_tf - perturbed_plus shape:", tf.shape(perturbed_plus))
        
        fitness_plus = self.evaluate_policy_batch_tf(perturbed_plus)
        fitness_minus = self.evaluate_policy_batch_tf(perturbed_minus)
        base_fitness = self.evaluate_policy_batch_tf(base_params)
        
        tf.print("compute_gradient_batch_tf - fitness_plus shape:", tf.shape(fitness_plus))
        
        gradients = (fitness_plus - fitness_minus) / (2 * epsilon)
        tf.print("compute_gradient_batch_tf - gradients before reshape shape:", tf.shape(gradients))
        gradients = tf.reshape(gradients, [num_elites, num_params])
        tf.print("compute_gradient_batch_tf - gradients after reshape shape:", tf.shape(gradients))
        
        return gradients, base_fitness, perturbed_plus, fitness_plus, perturbed_minus, fitness_minus

    def compute_gradient_batch(self, elites):
        base_params = tf.constant([ind[:] for ind in elites], dtype=tf.float32)
        gradients, base_fitness, perturbed_plus, fitness_plus, perturbed_minus, fitness_minus = self.compute_gradient_batch_tf(base_params)
        perturbed_plus_batch = [creator.Individual(p.numpy().tolist()) for p in tf.unstack(perturbed_plus)]
        perturbed_minus_batch = [creator.Individual(p.numpy().tolist()) for p in tf.unstack(perturbed_minus)]
        return gradients, base_fitness, perturbed_plus_batch, fitness_plus, perturbed_minus_batch, fitness_minus

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.float32)
    ])
    def gradient_descent_batch_tf(self, elites_tensor, learning_rate):
        gradients, base_fitness, perturbed_plus, fitness_plus, perturbed_minus, fitness_minus = self.compute_gradient_batch_tf(elites_tensor)
        updated_params = elites_tensor - learning_rate * gradients
        a_size = 3 * self.state_dim
        b_size = 3 * self.state_dim * self.state_dim
        a_clip = tf.clip_by_value(updated_params[:, :a_size], -20, 20)
        b_clip = tf.clip_by_value(updated_params[:, a_size:a_size + b_size], -10, 10)
        c_clip = tf.clip_by_value(updated_params[:, a_size + b_size:], -2, 2)
        updated_params = tf.concat([a_clip, b_clip, c_clip], axis=1)
        return updated_params, perturbed_plus, fitness_plus, perturbed_minus, fitness_minus

    def gradient_descent_batch(self, elites, learning_rate=0.1):
        elites_tensor = tf.constant([ind[:] for ind in elites], dtype=tf.float32)
        updated_params, perturbed_plus, fitness_plus, perturbed_minus, fitness_minus = self.gradient_descent_batch_tf(elites_tensor, learning_rate)
        improved_elites = [creator.Individual(p.numpy().tolist()) for p in tf.unstack(updated_params)]
        perturbed_plus_batch = [creator.Individual(p.numpy().tolist()) for p in tf.unstack(perturbed_plus)]
        perturbed_minus_batch = [creator.Individual(p.numpy().tolist()) for p in tf.unstack(perturbed_minus)]
        return improved_elites, perturbed_plus_batch, fitness_plus, perturbed_minus_batch, fitness_minus

    def optimize(self):
        population = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(5)
        
        fitnesses = self.evaluate_policy_batch(population)
        tf.print("Initial fitnesses (min, max, mean):", 
                 tf.reduce_min(fitnesses), tf.reduce_max(fitnesses), tf.reduce_mean(fitnesses))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = (float(fit.numpy()),)
        hof.update(population)

        for gen in range(self.generations):
            elites = list(map(self.toolbox.clone, hof.items[:5]))
            breeding_pool = self.toolbox.select(population, self.pop_size // 2)
            offspring = list(map(self.toolbox.clone, breeding_pool))
            offspring.extend(list(map(self.toolbox.clone, breeding_pool)))
            offspring = offspring[:self.pop_size - 5]

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if tf.random.uniform([]) < 0.5:
                    child1_tensor = tf.constant(child1[:], dtype=tf.float32)
                    child2_tensor = tf.constant(child2[:], dtype=tf.float32)
                    child1_new, child2_new = self.toolbox.mate(child1_tensor, child2_tensor, 0.5)
                    del child1.fitness.values
                    del child2.fitness.values
                    child1[:] = child1_new.numpy().tolist()
                    child2[:] = child2_new.numpy().tolist()

            for i in range(len(offspring)):
                if tf.random.uniform([]) < 0.7:
                    mutant_tensor = tf.constant(offspring[i][:], dtype=tf.float32)
                    mutant_new = self.toolbox.mutate(mutant_tensor)
                    del offspring[i].fitness.values
                    offspring[i][:] = mutant_new.numpy().tolist()

            elite_offspring = list(map(self.toolbox.clone, hof.items[:5]))
            for elite1, elite2 in zip(elite_offspring[::2], elite_offspring[1::2]):
                if tf.random.uniform([]) < 0.7:
                    elite1_tensor = tf.constant(elite1[:], dtype=tf.float32)
                    elite2_tensor = tf.constant(elite2[:], dtype=tf.float32)
                    elite1_new, elite2_new = self.toolbox.mate(elite1_tensor, elite2_tensor, 0.5)
                    del elite1.fitness.values
                    del elite2.fitness.values
                    elite1[:] = elite1_new.numpy().tolist()
                    elite2[:] = elite2_new.numpy().tolist()
            for i in range(len(elite_offspring)):
                if tf.random.uniform([]) < 0.5:
                    elite_tensor = tf.constant(elite_offspring[i][:], dtype=tf.float32)
                    elite_new = self.toolbox.elite_mutate(elite_tensor)
                    del elite_offspring[i].fitness.values
                    elite_offspring[i][:] = elite_new.numpy().tolist()

            gradient_improved, perturbed_plus_batch, fitness_plus, perturbed_minus_batch, fitness_minus = self.gradient_descent_batch(hof.items[:5], learning_rate=0.1)
            print("gradient improved")
            gradient_fitnesses = self.evaluate_policy_batch(gradient_improved)
            for ind, fit in zip(gradient_improved, gradient_fitnesses):
                ind.fitness.values = (float(fit.numpy()),)

            all_perturbed = perturbed_plus_batch + perturbed_minus_batch
            all_fitnesses = tf.concat([fitness_plus, fitness_minus], axis=0)
            best_perturbed_idx = tf.argmin(all_fitnesses)
            best_perturbed = all_perturbed[best_perturbed_idx]
            best_perturbed.fitness.values = (float(all_fitnesses[best_perturbed_idx].numpy()),)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                print("invalid")
                fitnesses = self.evaluate_policy_batch(invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = (float(fit.numpy()),)

            combined = elites + gradient_improved + [best_perturbed] + population + offspring + elite_offspring
            combined_fits = [ind.fitness.values[0] for ind in combined if ind.fitness.valid]
            tf.print(f"Gen {gen} combined fitnesses (min, max, mean):", 
                     min(combined_fits) if combined_fits else float('inf'), 
                     max(combined_fits) if combined_fits else float('-inf'), 
                     sum(combined_fits) / len(combined_fits) if combined_fits else 0)
            
            population[:] = self.toolbox.select_best(combined, self.pop_size)

            fits = [ind.fitness.values[0] for ind in population]
            if max(fits) - min(fits) < 10.0:
                random_ind = self.toolbox.population(n=self.pop_size // 10)
                population[-len(random_ind):] = random_ind
                print("diversity")
                fitnesses = self.evaluate_policy_batch(random_ind)
                for ind, fit in zip(random_ind, fitnesses):
                    ind.fitness.values = (float(fit.numpy()),)

            hof.update(population)
            fits = [ind.fitness.values[0] for ind in population]
            tf.print(f"Gen {gen} stats - min: {min(fits)}, max: {max(fits)}, mean: {sum(fits) / len(fits)}")

            best_ind = hof[0]
            params = tf.constant(best_ind, dtype=tf.float32)
            a = tf.reshape(params[:3*self.state_dim], [1, 3, self.state_dim])
            b = tf.reshape(params[3*self.state_dim:3*self.state_dim + 3*self.state_dim**2], 
                           [1, 3, self.state_dim, self.state_dim])
            c = tf.reshape(params[3*self.state_dim + 3*self.state_dim**2:], 
                           [1, 3, self.state_dim, self.state_dim, self.state_dim])
            policy_params = (a, b, c)
            self.tester.run_simulation_with_policy(policy_params, self.initial_state, idx=0, gen=gen)

            if gen % 5 == 0:
                traj = self.tester.predict_trajectory(self.initial_state[tf.newaxis, :], a, b, c)
                traj = tf.transpose(traj, [1, 0, 2])
                with open(f'policies/best_traj_gen_{gen}.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['time', 'x', 'x_dot', 'cos_theta', 'sin_theta', 'theta_dot'])
                    for state in traj[0]:
                        row = [float(x) for x in state.numpy()]
                        writer.writerow(row)

            if gen % 5 == 0 and gen > 0:
                tf.print(f"Updating GP with all simulation data at Gen {gen}")
                tf.print("Total GP data size:", tf.shape(self.tester.X_data), tf.shape(self.tester.y_data))
                best_pre_refit = hof[0]
                self.tester.fit_gp()
                print("hof")
                hof_fitnesses = self.evaluate_policy_batch(hof.items)
                tf.print(f"Gen {gen} raw HoF fitnesses after GP update:", hof_fitnesses)
                for ind, fit in zip(hof.items, hof_fitnesses):
                    ind.fitness.values = (float(fit.numpy()),)

                best_pre_fitness = self.evaluate_policy_batch([best_pre_refit])
                tf.print("Best pre-refit policy re-scored:", best_pre_fitness[0])

                fitnesses = self.evaluate_policy_batch(population)
                tf.print(f"Gen {gen} raw population fitnesses after GP update:", 
                         tf.reduce_min(fitnesses), tf.reduce_max(fitnesses), tf.reduce_mean(fitnesses))
                for ind, fit in zip(population, fitnesses):
                    ind.fitness.values = (float(fit.numpy()),)
                hof.clear()
                hof.update(population)
                fits = [ind.fitness.values[0] for ind in population]
                tf.print(f"Gen {gen} stats after GP update - min: {min(fits)}, max: {max(fits)}, mean: {sum(fits) / len(fits)}")
                tf.print("Top 5 population fitnesses after update:", sorted(fits)[:5])
                tf.print(f"Gen {gen} HoF fitnesses after population update:", [ind.fitness.values[0] for ind in hof.items])

        best_ind = hof[0]
        final_policy = {
            'type': 'cubic',
            'params': {
                'a': tf.reshape(best_ind[:3*self.state_dim], [3, self.state_dim]).numpy().tolist(),
                'b': tf.reshape(best_ind[3*self.state_dim:3*self.state_dim + 3*self.state_dim**2], 
                               [3, self.state_dim, self.state_dim]).numpy().tolist(),
                'c': tf.reshape(best_ind[3*self.state_dim + 3*self.state_dim**2:], 
                               [3, self.state_dim, self.state_dim, self.state_dim]).numpy().tolist()
            }
        }
        with open('policies/final_policy.json', 'w') as f:
            json.dump(final_policy, f)

        params = tf.constant(best_ind, dtype=tf.float32)
        a = tf.reshape(params[:3*self.state_dim], [1, 3, self.state_dim])
        b = tf.reshape(params[3*self.state_dim:3*self.state_dim + 3*self.state_dim**2], 
                       [1, 3, self.state_dim, self.state_dim])
        c = tf.reshape(params[3*self.state_dim + 3*self.state_dim**2:], 
                       [1, 3, self.state_dim, self.state_dim, self.state_dim])
        traj = self.tester.predict_trajectory(self.initial_state[tf.newaxis, :], a, b, c)
        traj = tf.transpose(traj, [1, 0, 2])
        with open('policies/final_trajectory.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'x', 'x_dot', 'cos_theta', 'sin_theta', 'theta_dot'])
            for state in traj[0]:
                row = [float(x) for x in state.numpy()]
                writer.writerow(row)

        return best_ind

def main():
    tester = PendulumGPTester(num_links=1, sim_time=10.0, dt=0.0125)
    X_data = []
    y_data = []
    for i in range(20):
        initial_state = tester.generate_random_initial_state()
        X, y = tester.run_simulation(initial_state.numpy(), i)
        if tester.X_data is None:
            tester.X_data = X
            tester.y_data = y
        else:
            tester.X_data = tf.concat([tester.X_data, X], axis=0)
            tester.y_data = tf.concat([tester.y_data, y], axis=0)
    
    tester.fit_gp()

    initial_state = tf.constant([0.0125, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=tf.float32)
    optimizer = PolicyOptimizer(tester, initial_state, pop_size=500, generations=50)
    best_policy = optimizer.optimize()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tf.print(f"Error occurred: {e}")
        raise