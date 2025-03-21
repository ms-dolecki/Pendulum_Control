import tensorflow as tf
import gpflow
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
        self.state_dim = 1 + 2 + 3 * num_links
        self.output_dim = 2 + 3 * num_links
        self.sim_time = sim_time
        self.dt = dt
        self.steps = int(sim_time / dt) + 1
        self.model = None  # No self.policy_params needed
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        self.mask = None

    def _generate_random_policy(self):
        a = tf.random.uniform([3, self.state_dim-1], minval=-20, maxval=20, dtype=tf.float32)
        b = tf.random.uniform([3, self.state_dim-1, self.state_dim-1], minval=-10, maxval=10, dtype=tf.float32)
        c = tf.random.uniform([3, self.state_dim-1, self.state_dim-1, self.state_dim-1], minval=-2, maxval=2, dtype=tf.float32)
        return (a, b, c)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 5], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 3, 5], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 3, 5, 5], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 3, 5, 5, 5], dtype=tf.float32)
    ])
    def predict_action(self, states, a, b, c):
        batch_size = tf.shape(states)[0]
        #tf.print(batch_size)
        #tf.print(tf.shape(a))  # Keep your debug print, adjusted for new param
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

        policy_params = self._generate_random_policy()  # Use local var, not self.policy_params
        policy = {'type': 'cubic', 'params': {
            'a': policy_params[0].numpy().tolist(),
            'b': policy_params[1].numpy().tolist(),
            'c': policy_params[2].numpy().tolist()
        }}
        with open(policy_file, 'w') as f:
            json.dump(policy, f)

        sim = PendulumSim(lengths=[0.5] * self.num_links, masses=[1.0] * self.num_links)
        sim.simulate(t_span=(0, self.sim_time/3), initial_state=initial_state, dt=self.dt, 
                     sim_data_file=sim_data_file, sim_done_file=sim_done_file)
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

    def generate_random_initial_state(self):
        x = tf.random.uniform([], minval=-1.0, maxval=1.0, dtype=tf.float32)
        x_dot = tf.random.uniform([], minval=-1.0, maxval=1.0, dtype=tf.float32)
        theta = tf.random.uniform([], minval=0, maxval=2 * tf.constant(3.141592653589793, dtype=tf.float32), dtype=tf.float32)
        theta_dot = tf.random.uniform([], minval=-1.0, maxval=1.0, dtype=tf.float32)
        return tf.stack([self.dt, x, x_dot, tf.cos(theta), tf.sin(theta), theta_dot])

    def fit_gp(self, X_data, y_data):
        subset_size = min(1000, X_data.shape[0])
        indices = tf.random.shuffle(tf.range(X_data.shape[0]))[:subset_size]
        X_subset = tf.gather(X_data, indices)
        y_subset = tf.gather(y_data, indices)

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
        tf.print("Predict trajectory batch size:", batch_size)
        states = tf.TensorArray(dtype=tf.float32, size=self.steps, dynamic_size=False)
        states = states.write(0, initial_states)

        @tf.function
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
            clip_batch = tf.clip_by_value(states_new[:, 1:2], -10, 10)
            states_new_xdot = tf.concat([states_new[:, :2], zeros_batch, states_new[:, 3:]], axis=1)
            states_new = tf.where(tf.abs(x)[:, tf.newaxis] > 10, states_new_xdot, states_new)
            states_new_x = tf.concat([states_new[:, :1], clip_batch, states_new[:, 2:]], axis=1)
            states_new = tf.where(tf.abs(x)[:, tf.newaxis] > 10, states_new_x, states_new)
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
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", self.custom_mutation, indpb=0.9)
        self.toolbox.register("elite_mutate", self.elite_mutation, indpb=0.5)
        self.toolbox.register("select", tools.selTournament, tournsize=50)
        self.toolbox.register("select_best", tools.selBest)

    def init_individual(self):
        a = tf.random.uniform([3 * self.state_dim], minval=-20, maxval=20, dtype=tf.float32)
        b = tf.random.uniform([3 * self.state_dim * self.state_dim], minval=-10, maxval=10, dtype=tf.float32)
        c = tf.random.uniform([3 * self.state_dim * self.state_dim * self.state_dim], minval=-2, maxval=2, dtype=tf.float32)
        return creator.Individual(tf.concat([a, b, c], axis=0).numpy().tolist())

    def custom_mutation(self, individual, indpb):
        ind_tensor = tf.constant(individual, dtype=tf.float32)
        a_size = 3 * self.state_dim
        b_size = 3 * self.state_dim * self.state_dim
        mask = tf.random.uniform([len(individual)]) < indpb
        noise = tf.where(
            mask,
            tf.concat([
                tf.random.normal([a_size], 0, 15.0),
                tf.random.normal([b_size], 0, 1.5),
                tf.random.normal([len(individual) - a_size - b_size], 0, 0.15)
            ], axis=0),
            tf.zeros([len(individual)], dtype=tf.float32)
        )
        mutated = ind_tensor + noise
        a_clip = tf.clip_by_value(mutated[:a_size], -20, 20)
        b_clip = tf.clip_by_value(mutated[a_size:a_size + b_size], -10, 10)
        c_clip = tf.clip_by_value(mutated[a_size + b_size:], -2, 2)
        mutated = tf.concat([a_clip, b_clip, c_clip], axis=0)
        return creator.Individual(mutated.numpy().tolist()),

    def elite_mutation(self, individual, indpb):
        ind_tensor = tf.constant(individual, dtype=tf.float32)
        a_size = 3 * self.state_dim
        b_size = 3 * self.state_dim * self.state_dim
        mask = tf.random.uniform([len(individual)]) < indpb
        noise = tf.where(
            mask,
            tf.concat([
                tf.random.normal([a_size], 0, 5.0),
                tf.random.normal([b_size], 0, 0.5),
                tf.random.normal([len(individual) - a_size - b_size], 0, 0.05)
            ], axis=0),
            tf.zeros([len(individual)], dtype=tf.float32)
        )
        mutated = ind_tensor + noise
        a_clip = tf.clip_by_value(mutated[:a_size], -20, 20)
        b_clip = tf.clip_by_value(mutated[a_size:a_size + b_size], -10, 10)
        c_clip = tf.clip_by_value(mutated[a_size + b_size:], -2, 2)
        mutated = tf.concat([a_clip, b_clip, c_clip], axis=0)
        return creator.Individual(mutated.numpy().tolist()),

    def compute_gradient_batch(self, elites):
        num_elites = len(elites)
        num_params = len(elites[0])
        epsilon = 0.01
        perturbed_plus = []
        perturbed_minus = []
        base_params = tf.constant([ind[:] for ind in elites], dtype=tf.float32)

        for elite_idx, elite in enumerate(elites):
            for param_idx in range(num_params):
                delta_plus = tf.zeros_like(base_params[elite_idx])
                delta_minus = tf.zeros_like(base_params[elite_idx])
                delta_plus = tf.tensor_scatter_nd_add(delta_plus, [[param_idx]], [epsilon])
                delta_minus = tf.tensor_scatter_nd_add(delta_minus, [[param_idx]], [-epsilon])
                perturbed_plus.append(base_params[elite_idx] + delta_plus)
                perturbed_minus.append(base_params[elite_idx] + delta_minus)

        tf.print("Perturbed plus length:", len(perturbed_plus))
        perturbed_plus_batch = [creator.Individual(p.numpy().tolist()) for p in perturbed_plus]
        perturbed_minus_batch = [creator.Individual(p.numpy().tolist()) for p in perturbed_minus]
        fitness_plus = self.evaluate_policy_batch(perturbed_plus_batch)
        fitness_minus = self.evaluate_policy_batch(perturbed_minus_batch)
        base_fitness = self.evaluate_policy_batch(elites)

        gradients = []
        for elite_idx in range(num_elites):
            elite_grad = []
            for param_idx in range(num_params):
                plus_idx = elite_idx * num_params + param_idx
                grad = (fitness_plus[plus_idx] - fitness_minus[plus_idx]) / (2 * epsilon)
                elite_grad.append(grad.numpy())
            gradients.append(tf.constant(elite_grad, dtype=tf.float32))

        return gradients, base_fitness

    def gradient_descent_batch(self, elites, learning_rate=0.1):
        gradients, base_fitness = self.compute_gradient_batch(elites)
        improved_elites = []
        for elite, grad in zip(elites, gradients):
            params = tf.constant(elite[:], dtype=tf.float32)
            updated_params = params - learning_rate * grad
            a_size = 3 * self.state_dim
            b_size = 3 * self.state_dim * self.state_dim
            a_clip = tf.clip_by_value(updated_params[:a_size], -20, 20)
            b_clip = tf.clip_by_value(updated_params[a_size:a_size + b_size], -10, 10)
            c_clip = tf.clip_by_value(updated_params[a_size + b_size:], -2, 2)
            updated_params = tf.concat([a_clip, b_clip, c_clip], axis=0)
            improved_elite = creator.Individual(updated_params.numpy().tolist())
            improved_elites.append(improved_elite)
        return improved_elites

    def evaluate_policy_batch(self, individuals):
        batch_size = len(individuals)
        #tf.print("Batch size:", batch_size)
        ind_tensor = tf.constant([ind[:] for ind in individuals], dtype=tf.float32)
        a = tf.reshape(ind_tensor[:, :3*self.state_dim], [batch_size, 3, self.state_dim])
        b = tf.reshape(ind_tensor[:, 3*self.state_dim:3*self.state_dim + 3*self.state_dim**2], 
                       [batch_size, 3, self.state_dim, self.state_dim])
        c = tf.reshape(ind_tensor[:, 3*self.state_dim + 3*self.state_dim**2:], 
                       [batch_size, 3, self.state_dim, self.state_dim, self.state_dim])
        #tf.print("Policy params shape in evaluate:", tf.shape(a))
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
        return total_cost

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
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if tf.random.uniform([]) < 0.7:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            elite_offspring = list(map(self.toolbox.clone, hof.items[:5]))
            for elite1, elite2 in zip(elite_offspring[::2], elite_offspring[1::2]):
                if tf.random.uniform([]) < 0.7:
                    self.toolbox.mate(elite1, elite2)
                    del elite1.fitness.values
                    del elite2.fitness.values
            for elite in elite_offspring:
                if tf.random.uniform([]) < 0.5:
                    self.toolbox.elite_mutate(elite)
                    del elite.fitness.values

            gradient_improved = self.gradient_descent_batch(hof.items[:5], learning_rate=0.1)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                fitnesses = self.evaluate_policy_batch(invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = (float(fit.numpy()),)

            combined = elites + gradient_improved + population + offspring + elite_offspring
            population[:] = self.toolbox.select_best(combined, self.pop_size)

            fits = [ind.fitness.values[0] for ind in population]
            if max(fits) - min(fits) < 10.0:
                random_ind = self.toolbox.population(n=self.pop_size // 10)
                population[-len(random_ind):] = random_ind
                fitnesses = self.evaluate_policy_batch(random_ind)
                for ind, fit in zip(random_ind, fitnesses):
                    ind.fitness.values = (float(fit.numpy()),)

            hof.update(population)
            fits = [ind.fitness.values[0] for ind in population]
            tf.print(f"Gen {gen} stats - min: {min(fits)}, max: {max(fits)}, mean: {sum(fits) / len(fits)}")

            if gen % 5 == 0:
                best_ind = hof[0]
                params = tf.constant(best_ind, dtype=tf.float32)
                a = tf.reshape(params[:3*self.state_dim], [1, 3, self.state_dim])
                b = tf.reshape(params[3*self.state_dim:3*self.state_dim + 3*self.state_dim**2], 
                              [1, 3, self.state_dim, self.state_dim])
                c = tf.reshape(params[3*self.state_dim + 3*self.state_dim**2:], 
                              [1, 3, self.state_dim, self.state_dim, self.state_dim])
                traj = self.tester.predict_trajectory(self.initial_state[tf.newaxis, :], a, b, c)
                traj = tf.transpose(traj, [1, 0, 2])
                tf.print(traj)
                with open(f'policies/best_traj_gen_{gen}.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['time', 'x', 'x_dot', 'cos_theta', 'sin_theta', 'theta_dot'])
                    for state in traj[0]:
                        row = [float(x) for x in state.numpy()]
                        writer.writerow(row)

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
        return best_ind

def main():
    tester = PendulumGPTester(num_links=1, sim_time=10.0, dt=0.0125)
    X_data = []
    y_data = []
    for i in range(20):
        initial_state = tester.generate_random_initial_state()
        X, y = tester.run_simulation(initial_state, i)
        X_data.append(X)
        y_data.append(y)
    X_data = tf.concat(X_data, axis=0)
    y_data = tf.concat(y_data, axis=0)

    tester.fit_gp(X_data, y_data)

    initial_state = tf.constant([0.0125, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=tf.float32)
    optimizer = PolicyOptimizer(tester, initial_state, pop_size=500, generations=50)
    best_policy = optimizer.optimize()

    params = tf.constant(best_policy, dtype=tf.float32)
    a = tf.reshape(params[:3*(tester.state_dim-1)], [1, 3, tester.state_dim-1])
    b = tf.reshape(params[3*(tester.state_dim-1):3*(tester.state_dim-1) + 3*(tester.state_dim-1)**2], 
                  [1, 3, tester.state_dim-1, tester.state_dim-1])
    c = tf.reshape(params[3*(tester.state_dim-1) + 3*(tester.state_dim-1)**2:], 
                  [1, 3, tester.state_dim-1, tester.state_dim-1, tester.state_dim-1])
    traj = tester.predict_trajectory(initial_state[tf.newaxis, :], a, b, c)
    traj = tf.transpose(traj, [1, 0, 2])
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
        tf.print(f"Error occurred: {e}")
        raise