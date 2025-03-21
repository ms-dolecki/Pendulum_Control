import numpy as np
#from scipy.optimize import minimize
import gpflow
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import tensorflow as tf
import json
from itertools import combinations
import random
import math
from itertools import combinations_with_replacement
import threading
from watchfiles import watch, Change
gpflow.config.set_default_float(tf.float64)
import os
from copy import deepcopy

# Ensure the directory exists
os.makedirs("./trajectories", exist_ok=True)

class Pilco_Learn:
    def __init__(self,model,scaler_x, scaler_y,input_data,output_data):
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.scaler_x_tf = self.TFStandardScaler()
        self.scaler_y_tf = self.TFStandardScaler()
        self.input_data = input_data
        self.output_data = output_data
        self.model = model
        self.sim_iteration = 0
        path_to_watch = "sim_done.txt"  # Replace with your file or directory path
        watcher_thread = threading.Thread(target=self.file_watcher, args=[path_to_watch], daemon=True)
        watcher_thread.start()

    class TFStandardScaler:
        def __init__(self):
            self.mean = None
            self.var = None

        def fit(self, X):
            # Assuming X is a 2D tensor where each row is a sample
            mean = tf.reduce_mean(X, axis=0)
            var = tf.math.reduce_variance(X, axis=0)
            self.mean = tf.Variable(mean, dtype=tf.float32)
            self.var = tf.Variable(var, dtype=tf.float32)

        @tf.function
        def transform(self, X):
            if self.mean is None or self.var is None:
                raise ValueError("Scaler not fitted")
            return (X - self.mean) / tf.sqrt(self.var + 1e-7)  # Adding small epsilon for numerical stability
        
        @tf.function
        def inverse_transform(self, Y):
            if self.mean is None or self.var is None:
                raise ValueError("Scaler not fitted")
            return (Y*tf.sqrt(self.var + 1e-7) + self.mean)  # Adding small epsilon for numerical stability

        @tf.function
        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)
    
    def file_watcher(self, path_to_watch):
        deletion_stop_event = threading.Event()
        while True:
            for changes in watch(path_to_watch, stop_event=deletion_stop_event):
                for change_type, path in changes:
                    if change_type == Change.deleted:
                        deletion_stop_event.set()
                        time.sleep(2)
                    print(f"{Change(change_type).name} {path}")
                    self.run_finish()
            print("exited for loop")
            deletion_stop_event.clear()
                
    
    @tf.function
    def vector_square_tf(self,vector):
        return tf.tensordot(vector, vector, axes=0)
    
    @tf.function
    def vector_cube_tf(self,vector):
        return tf.tensordot(vector, self.vector_square_tf(vector), axes=0)
    
    @tf.function
    def calculate_action_tf(self, state, a, b, c):
        p = state
        p2 = self.vector_square_tf(state)
        p3 = self.vector_cube_tf(state)
        action = tf.reduce_sum(a*p) + tf.reduce_sum(b*p2) + tf.reduce_sum(c*p3)
        return tf.clip_by_value(action,-3,3)
    
    @tf.function
    def calculate_action_2_tf(self, state, policy_vars):
        a,b,c = policy_vars
        p = state
        p2 = self.vector_square_tf(state)
        p3 = self.vector_cube_tf(state)
        action = tf.reduce_sum(tf.nn.leaky_relu(a[0]*p+a[1])*a[2])+tf.reduce_sum(tf.nn.leaky_relu(b[0]*p2+b[1])*b[2])+tf.reduce_sum(tf.nn.leaky_relu(c[0]*p3+c[1])*c[2])
        return tf.clip_by_value(action,-3,3)

    def run_finish(self):
        print("run_finish")
        self.sim_iteration += 1

    # loads time-ordered state data
    def load_data(self,data_file):
        data = open(data_file, "r")
        input_data = []
        output_data = []
        for ln in data:
            #print(ln)
            ln.strip()
            ln = ln.strip("\n").split(",")
            delta_T = float(ln[0].strip())
            x = float(ln[1].strip())
            v = float(ln[2].strip())
            #angle = float(ln[3].strip())
            angle_cos = float(ln[3].strip())
            angle_sin = float(ln[4].strip())
            angle_dot = float(ln[5].strip())
            a_base = float(ln[6].strip())
            #input_data.append([delta_T,x,v,angle,angle_dot,a_base])
            input_data.append([delta_T,x,v,angle_cos,angle_sin,angle_dot,a_base])
            #input_data.append([x,v,angle,angle_dot,a_base])
            #output_data.append([x,v,angle,angle_dot])
            output_data.append([x,v,angle_cos,angle_sin,angle_dot])
        data.close()
        return input_data[:-1],output_data[1:]

    # loads collective input-output data
    def load_data_2(self,data_file):
        data = open(data_file, "r")
        input_data = []
        output_data = []
        for ln in data:
            ln.strip()
            ln = ln.strip("\n").split(",")
            delta_T = float(ln[0].strip())
            x_input = float(ln[1].strip())
            v_input = float(ln[2].strip())
            #angle_input = float(ln[3].strip())
            angle_cos_input = float(ln[3].strip())
            angle_sin_input = float(ln[4].strip())
            angle_dot_input = float(ln[5].strip())
            a_base = float(ln[6].strip())
            x_output = float(ln[7].strip())
            v_output = float(ln[8].strip())
            #angle_output = float(ln[9].strip())
            angle_cos_output = float(ln[9].strip())
            angle_sin_output = float(ln[10].strip())
            angle_dot_output = float(ln[11].strip())
            #input_data.append([delta_T,x_input,v_input,angle_input,angle_dot_input,a_base])
            input_data.append([delta_T,x_input,v_input,angle_cos_input,angle_sin_input,angle_dot_input,a_base])
            #input_data.append([x_input,v_input,angle_input,angle_dot_input,a_base])
            #output_data.append([x_output,v_output,angle_output,angle_dot_output])
            output_data.append([x_output,v_output,angle_cos_output,angle_sin_output,angle_dot_output])
        data.close()
        return input_data,output_data

    # saves collective input-output data
    def save_model_data(self,input_output_data,data_file):
        data = open(data_file, "w")
        for row in input_output_data:
            delta_T = row[0]
            #delta_T= 0.01
            x_input = row[1]
            v_input = row[2]
            #theta_input = row[3]
            angle_cos_input = row[3]
            angle_sin_input = row[4]
            theta_dot_input = row[5]
            a_base = row[6]
            x_output = row[7]
            v_output = row[8]
            #theta_output = row[8]
            angle_cos_output = row[9]
            angle_sin_output = row[10]
            theta_dot_output = row[11]
            #data.write(str(delta_T)+","+str(x_input)+","+str(v_input)+","+str(theta_input)+","+str(theta_dot_input)+","+str(a_base)+","+str(x_output)+","+str(v_output)+","+str(theta_output)+","+str(theta_dot_output)+"\n")
            data.write(str(delta_T)+","+str(x_input)+","+str(v_input)+","+str(angle_cos_input)+","+str(angle_sin_input)+","+str(theta_dot_input)+","+str(a_base)+","+str(x_output)+","+str(v_output)+","+str(angle_cos_output)+","+str(angle_sin_output)+","+str(theta_dot_output)+"\n")
        data.close()


    @tf.function
    def evaluate_policies(self, start_state, a_tensor, b_tensor, c_tensor):
        time_step = 0.01
        total_time = 5
        delta_T = tf.fill([tf.shape(a_tensor)[0]], time_step)
        delta_T = tf.expand_dims(delta_T, axis=1)
        current_state = tf.tile(tf.expand_dims(start_state, 0), [tf.shape(a_tensor)[0], 1])
        action = tf.map_fn(lambda args: self.calculate_action_2_tf(*args), 
                   (current_state, [a_tensor, b_tensor, c_tensor]),dtype=tf.float32)
        action = tf.expand_dims(action, axis=1)
        num_steps = int(total_time/time_step)
        future_state = current_state
        trajectories = tf.expand_dims(current_state, axis=0)

        def body(step, trajectories, future_state, action):
            # Ensure all inputs to concat have the same rank
            scaled_current_state = tf.cast(tf.convert_to_tensor(self.scaler_x_tf.transform(tf.concat([delta_T, future_state, action], axis=1))), tf.float64)
            #scaled_current_state = tf.cast(tf.convert_to_tensor(self.scaler_x_tf.transform(tf.concat([future_state, action], axis=1))), tf.float64)
            scaled_future_state = self.model.predict_f(scaled_current_state)
            future_state = self.scaler_y_tf.inverse_transform(tf.cast(scaled_future_state, tf.float32))[0]
            action = tf.map_fn(lambda args: self.calculate_action_tf(*args), 
                   (future_state, a_tensor, b_tensor, c_tensor),dtype=tf.float32)
            action = tf.expand_dims(action, axis=1)
            trajectories = tf.concat([trajectories, tf.expand_dims(future_state,axis=0)], axis=0)
            tf.print("Step:", step, "Trajectory Shape:", tf.shape(trajectories), "Future State Shape:", tf.shape(future_state))
            return step + 1, trajectories, future_state, action

        def cond(step, *args):
            return step < num_steps

        # Use while loop with adjusted shape invariants
        _, trajectories, _, _ = tf.while_loop(
            cond, 
            body, 
            [tf.constant(0), trajectories, future_state, action],
            shape_invariants=[
                tf.TensorShape([]), 
                tf.TensorShape([None, None, None]),  # Specify exact dimensions where possible
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None])
            ],
            parallel_iterations=1  # This ensures sequential execution of iterations
        )

        # Compute cost
        x = trajectories[:, :, 0]
        v = trajectories[:, :, 1]
        angle_cos = trajectories[:, :, 2]  # Shape: [time_steps, batch_size]
        angle_sin = trajectories[:, :, 3]
        angle_dot = trajectories[:, :, 4]
        angle = tf.atan2(angle_sin, angle_cos)  # Shape: [time_steps, batch_size]
        #costs = time_step * tf.reduce_sum(tf.square(trajectories), axis=[0, 2])
        costs = time_step * (8*tf.reduce_sum(tf.square(x), axis=0)+1000*tf.reduce_sum(tf.sign(tf.abs(x)-2.9)+1, axis=0) + tf.reduce_sum(tf.square(v), axis=0) + 50*tf.reduce_sum(tf.square(angle), axis=0) + tf.reduce_sum(tf.square(angle_dot), axis=0))
        return costs, trajectories

    @tf.function
    def evaluate_policies_2(self, start_state, a_tensor, b_tensor, c_tensor):
        time_step = 0.01
        total_time = 5
        delta_T = tf.fill([tf.shape(a_tensor)[0]], time_step)
        delta_T = tf.expand_dims(delta_T, axis=1)
        current_state = tf.tile(tf.expand_dims(start_state, 0), [tf.shape(a_tensor)[0], 1])
        action = tf.map_fn(lambda args: self.calculate_action_tf(*args), 
                   (current_state, a_tensor, b_tensor, c_tensor),dtype=tf.float32)
        action = tf.expand_dims(action, axis=1)
        num_steps = int(total_time/time_step)
        future_state = current_state
        trajectory = tf.expand_dims(current_state, axis=0)
        for _ in range(num_steps):
            scaled_current_state = tf.cast(tf.convert_to_tensor(self.scaler_x_tf.transform(tf.concat([delta_T, future_state, action], axis=1))), tf.float64)
            scaled_future_state = self.model.predict_f(scaled_current_state)
            future_state = self.scaler_y_tf.inverse_transform(tf.cast(scaled_future_state, tf.float32))[0]
            action = tf.map_fn(lambda args: self.calculate_action_tf(*args), 
                   (future_state, a_tensor, b_tensor, c_tensor),dtype=tf.float32)
            action = tf.expand_dims(action, axis=1)
            trajectory = tf.concat([trajectory, tf.expand_dims(future_state,axis=0)], axis=0)
        cost = time_step * tf.reduce_sum(tf.square(trajectory), axis=[0,2])
        return cost
    

    def write_policy(self,policy):
        policy_file = "policy_config.txt"
        with open(policy_file, 'w') as file:
            json.dump(policy, file)

    def reset_sim(self,num):
        reset_file = "test.txt"
        reset = open(reset_file,"w")
        reset.write(str(num))
        reset.close()
        
    def add_policy_data(self,policy,model_file):
        self.write_policy(policy)
        current_sim_iteration = self.sim_iteration
        self.reset_sim(current_sim_iteration)
        print("simulation_reset")
        print(current_sim_iteration)
        while self.sim_iteration == current_sim_iteration:
            print(self.sim_iteration)
            time.sleep(2)
            pass
        input_data_1, output_data_1 = self.load_data("data.txt")
        if self.input_data is not None:
            #print(self.input_data)
            self.input_data = np.vstack((self.input_data, input_data_1))
            self.output_data = np.vstack((self.output_data, output_data_1))
        else:
            self.input_data = input_data_1
            self.output_data = output_data_1
        
        input_output_data = np.hstack((self.input_data,self.output_data))
        np.random.shuffle(input_output_data)
        self.save_model_data(input_output_data,model_file)
        self.input_data = input_output_data[:,:7]
        self.output_data = input_output_data[:,7:12]
        #self.input_data = input_output_data[:,:6]
        #self.output_data = input_output_data[:,6:10]
        #self.input_data = input_output_data[:,:5]
        #self.output_data = input_output_data[:,5:9]
  
#kernel = gpflow.kernels.Matern12()
#kernel = gpflow.kernels.SquaredExponential()
#kernel = gpflow.kernels.RationalQuadratic()
kernel = gpflow.kernels.Sum([gpflow.kernels.SquaredExponential(lengthscales=0.5),gpflow.kernels.Matern32(lengthscales=0.5), gpflow.kernels.White(variance=0.1)])
# Create a GP model
input_data = None
output_data = None
#model = None
model = gpflow.models.GPR(data=(np.array([[]]), np.array([[]])), kernel=deepcopy(kernel))
scaler_x = StandardScaler()
scaler_y = StandardScaler()
my_pilco_learn = Pilco_Learn(model,scaler_x,scaler_y,input_data,output_data)

for index in range(10):
    a = np.random.uniform(-100, 100, size=(3,5)).tolist()
    b = np.random.uniform(-10, 10, size=(3,5,5)).tolist()
    c = np.random.uniform(-1, 1, size=(3,5,5,5)).tolist()
    policy = {
                    "a":a,
                    "b":b,
                    "c":c
    }
    print(a,b,c)
    my_pilco_learn.add_policy_data(policy,"model.txt")

input_data,output_data = my_pilco_learn.load_data_2("model.txt")
my_pilco_learn.scaler_x_tf.fit(tf.constant(input_data))
my_pilco_learn.scaler_y_tf.fit(tf.constant(output_data))
input_data_scaled = tf.convert_to_tensor(my_pilco_learn.scaler_x.fit_transform(np.array(input_data)), dtype=tf.float64)
output_data_scaled = tf.convert_to_tensor(my_pilco_learn.scaler_y.fit_transform(np.array(output_data)), dtype=tf.float64)
#print(input_data_scaled)
#input_data_scaled = input_data
my_pilco_learn.input_data = input_data
my_pilco_learn.output_data = output_data
#my_pilco_learn.kernel = kernel
my_pilco_learn.model = gpflow.models.GPR(data=(input_data_scaled, output_data_scaled), kernel=deepcopy(kernel))
optimizer = gpflow.optimizers.Scipy()
batches = math.floor(len(input_data_scaled)/100)
for batch_number in range(batches):
    input_batch = np.array(input_data_scaled[batch_number*100:(batch_number+1)*100])
    output_batch = np.array(output_data_scaled[batch_number*100:(batch_number+1)*100])
    my_pilco_learn.model.data = (input_batch, output_batch)
    closure = my_pilco_learn.model.training_loss_closure()
    optimizer.minimize(closure, variables=my_pilco_learn.model.trainable_variables, options=dict(maxiter=10))


print("wait")
time.sleep(2)
print("done_waiting")

def flatten_list(nested_list):
    """
    Recursively flatten a nested list structure into a single list.
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


# Flatten the values, ignoring keys
initial_guess = []
for value in policy.values():
    initial_guess.extend(flatten_list(value))

def objective_function(initial_guess):
    policy = {
    "a" : initial_guess[:4],
    "b" : np.array(initial_guess[4:20]).reshape((4,4)).tolist(),
    "c" : np.array(initial_guess[20:]).reshape((4,4,4)).tolist()
    }
    return my_pilco_learn.evaluate_policy(policy,np.array([1.0,0.0,3.02123817989101573,-0.009558798511813068]))

def calculate_cost(policy):
    flat_policy = []
    for value in policy.values():
        flat_policy.extend(flatten_list(value))
    cost = objective_function(flat_policy)
    return cost

class Genetic_Algorithm:
    def __init__(self, pilco_learn):
        self.pilco_learn = pilco_learn
        self.population = []
        self.individuals_birthed = 0

    class Population:
        def __init__(self, genetic_algorithm):
            self.genetic_algorithm = genetic_algorithm
            self.individuals = []

    class Individual:
        def __init__(self, genetic_algorithm, policy, id):
            self.genetic_algorithm = genetic_algorithm
            self.policy = policy
            self.id = id
            #self.cost = self.genetic_algorithm.calculate_cost(policy)
            self.cost = 9999
            self.trajectory = None

    def calculate_cost(self, policy):
        flat_policy = []
        for value in policy.values():
            flat_policy.extend(self.flatten_list(value))
        cost = self.objective_function(flat_policy)
        return cost

    def flatten_list(self, nested_list):
        """
        Recursively flatten a nested list structure into a single list.
        """
        flat_list = []
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                flat_list.extend(flatten_list(item))
            else:
                flat_list.append(item)
        return flat_list

    def objective_function(self, initial_guess):
        policy = {
            "a" : np.array(initial_guess[:15]).reshape((3,5)),
            "b" : np.array(initial_guess[15:90]).reshape((3,5,5)).tolist(),
            "c" : np.array(initial_guess[90:465]).reshape((3,5,5,5)).tolist()
        }
        return self.pilco_learn.evaluate_policy(policy,np.array([1.0,0.0,math.cos(3.02123817989101573),math.sin(3.02123817989101573),-0.009558798511813068]))
    
    def add_individual(self, policy):
        new_individual = self.Individual(self, policy, self.individuals_birthed + 1)
        #if self.individuals_birthed % 2000 == 0:
        #    self.pilco_learn.add_policy_data(policy,"model.txt")
        self.population.append(new_individual)
        self.individuals_birthed += 1

    def calculate_population_costs(self):
        a_tensor = tf.constant([],shape=(0,3,5))
        b_tensor = tf.constant([],shape=(0,3,5,5))
        c_tensor = tf.constant([],shape=(0,3,5,5,5))
        index = 0
        for individual in self.population:
            #print(index)
            index = index+1
            a_tensor = tf.concat([a_tensor, tf.expand_dims(tf.constant(individual.policy["a"], dtype=a_tensor.dtype), axis=0)], axis=0)
            b_tensor = tf.concat([b_tensor, tf.expand_dims(tf.constant(individual.policy["b"], dtype=a_tensor.dtype), axis=0)], axis=0)
            c_tensor = tf.concat([c_tensor, tf.expand_dims(tf.constant(individual.policy["c"], dtype=a_tensor.dtype), axis=0)], axis=0)
        start_state = tf.constant([1.0,0.0,math.cos(3.02123817989101573),math.sin(3.02123817989101573),-0.009558798511813068])
        costs, trajectories = self.pilco_learn.evaluate_policies(start_state, a_tensor, b_tensor, c_tensor)
        print("costs")
        print(costs)
        index = 0
        for individual in self.population:
            individual.cost = costs[index]
            individual.trajectory = trajectories[:,index,:]
            #print(individual.cost)
            index = index+1
        return costs
    
    def print_population(self):
        for individual in self.population:
            print(individual.cost)
            #print(individual.policy)
            trajectory_np = individual.trajectory.numpy()
            with open(f"./trajectories/trajectory_individual_{individual.id}.txt", "w") as f:
                # Write each time step as a line
                for step in trajectory_np:
                    # Convert each step to a string of comma-separated values
                    line = ",".join(map(str, step))
                    f.write(line + "\n")

    def mate(self, individual_1, individual_2):
        flat_policy_1 = []
        flat_policy_2 = []
        flat_policy_new = []
        for value in individual_1.policy.values():
            flat_policy_1.extend(self.flatten_list(value))
        for value in individual_2.policy.values():
            flat_policy_2.extend(self.flatten_list(value))
        for index in range(len(flat_policy_1)):
            allele = random.randint(1, 2)
            mutation_factor = np.random.normal(loc=1, scale=0.05)
            mutation_flip = np.random.choice([1, -1], p=[0.8,0.2])
            if flat_policy_1[index] != flat_policy_2[index]:
                if allele == 1:
                    flat_policy_new.append(flat_policy_1[index]*mutation_factor*mutation_flip)
                else:
                    flat_policy_new.append(flat_policy_2[index]*mutation_factor*mutation_flip)
            else:
                flat_policy_new.append(flat_policy_1[index])

        policy_new = {
            "a" : np.array(flat_policy_new[:15]).reshape((3,5)).tolist(),
            "b" : np.array(flat_policy_new[15:90]).reshape((3,5,5)).tolist(),
            "c" : np.array(flat_policy_new[90:465]).reshape((3,5,5,5)).tolist()
        }
        self.add_individual(policy_new)
        
    def new_generation(self, survival_threshold):
        #surviving_population = [individual for individual in self.population if individual.cost < survival_threshold]
        surviving_population = sorted(self.population, key=lambda x: x.cost)[:survival_threshold]
        print(surviving_population[0].cost)
        self.pilco_learn.add_policy_data(surviving_population[0].policy,"model.txt")
        print(surviving_population[1].cost)
        self.pilco_learn.add_policy_data(surviving_population[1].policy,"model.txt")
        print(surviving_population[2].cost)
        self.pilco_learn.add_policy_data(surviving_population[2].policy,"model.txt")
        #self.population = surviving_population
        self.population = []
        pairs = [comb for comb in combinations_with_replacement(surviving_population, 2)]
        for pair in pairs:
            self.mate(pair[0],pair[1])
        return self.calculate_population_costs()
    
    
my_genetic_algorithm = Genetic_Algorithm(my_pilco_learn)

#population = []
for index in range(250):
    a = np.random.uniform(-100, 100, size=(3,5)).tolist()
    b = np.random.uniform(-10, 10, size=(3,5,5)).tolist()
    c = np.random.uniform(-1, 1, size=(3,5,5,5)).tolist()
    policy = {
                    "a":a,
                    "b":b,
                    "c":c
    }
    my_genetic_algorithm.add_individual(policy)


costs = my_genetic_algorithm.calculate_population_costs()
print(costs)
time.sleep(2)
costs = my_genetic_algorithm.calculate_population_costs()
print(costs)
time.sleep(2)
costs = my_genetic_algorithm.calculate_population_costs()
print(costs)
time.sleep(2)

print("gen1")
my_genetic_algorithm.print_population()
for index in range(25):
    input_data,output_data = my_pilco_learn.load_data_2("model.txt")
    input_data_scaled = tf.convert_to_tensor(my_pilco_learn.scaler_x.fit_transform(np.array(input_data)), dtype=tf.float64)
    output_data_scaled = tf.convert_to_tensor(my_pilco_learn.scaler_y.fit_transform(np.array(output_data)), dtype=tf.float64)
    my_pilco_learn.scaler_x_tf.fit(tf.constant(input_data))
    my_pilco_learn.scaler_y_tf.fit(tf.constant(output_data))
    my_pilco_learn.input_data = input_data
    my_pilco_learn.output_data = output_data
    my_pilco_learn.model = gpflow.models.GPR(data=(input_data_scaled, output_data_scaled), kernel=deepcopy(kernel))
    #optimizer = gpflow.optimizers.Scipy()
    batches = math.floor(len(input_data_scaled)/100)
    for batch_number in range(batches):
        input_batch = np.array(input_data_scaled[batch_number*100:(batch_number+1)*100])
        output_batch = np.array(output_data_scaled[batch_number*100:(batch_number+1)*100])
        my_pilco_learn.model.data = (input_batch, output_batch)
        closure = my_pilco_learn.model.training_loss_closure()
        optimizer.minimize(closure, variables=my_pilco_learn.model.trainable_variables, options=dict(maxiter=10))

    costs = my_genetic_algorithm.new_generation(150)
    print(costs)
    print("gen: "+str(index))
    my_genetic_algorithm.print_population()

