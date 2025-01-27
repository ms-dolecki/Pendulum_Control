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
#gpflow.config.set_default_float(tf.float32)
#tf.config.run_functions_eagerly(True)


class Pilco_Learn:
    #class Watcher(FileSystemEventHandler):
    #    def __init__(self,pilco_learn):
    #        self.pilco_learn = pilco_learn
    #    def on_modified(self, event):
    #        if event.is_directory:
    #            return
    #        self.pilco_learn.run_finish()

    def __init__(self,model,scaler_x, scaler_y,input_data,output_data):
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.input_data = input_data
        self.output_data = output_data
        self.model = model
        self.sim_iteration = 0
        path_to_watch = "sim_done.txt"  # Replace with your file or directory path
        watcher_thread = threading.Thread(target=self.file_watcher, args=[path_to_watch], daemon=True)
        watcher_thread.start()
        #event_handler = self.Watcher(self)
        #observer = Observer()
        #observer.schedule(event_handler, path=path_to_watch, recursive=False)  # Set recursive=True to monitor subdirectories
        #observer.start()

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
                

    def vector_square(self,vector):
        output = []
        for element in vector:
            output.append([element*element1 for element1 in vector])
        return output
    
    @tf.functon
    def vector_square_tf(self,vector):
        return tf.tensordot(vector, vector, axes=0)
    
    def vector_cube(self,vector):
        output = []
        for element in vector:
            output.append([[element*element1 for element1 in row] for row in self.vector_square(vector)])
        return output
    
    @tf.function
    def vector_cube_tf(self,vector):
        return tf.tensordot(vector, self.vector_square_tf(vector), axes=0)
    
    def calculate_action(self, state, policy):
        p = np.array(state)
        p2 = np.array(self.vector_square(state))
        p3 = np.array(self.vector_cube(state))
        a = np.array(policy["a"])
        b = np.array(policy["b"])
        c = np.array(policy["c"])
        action = max(min(np.sum(a*p) + np.sum(b*p2) + np.sum(c*p3),3),-3)
        return action
    
    @tf.function
    def calculate_action_tf(self, state, a, b, c):
        p = state
        p2 = self.vector_square(state)
        p3 = self.vector_cube(state)
        action = tf.reduce_sum(a*p) + tf.reduce_sum(b*p2) + tf.reduce_sum(c*p3)
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
            print(ln)
            ln.strip()
            ln = ln.strip("\n").split(",")
            delta_T = float(ln[0].strip())
            x = float(ln[1].strip())
            v = float(ln[2].strip())
            angle = float(ln[3].strip())
            angle_dot = float(ln[4].strip())
            a_base = float(ln[5].strip())
            input_data.append([delta_T,x,v,angle,angle_dot,a_base])
            output_data.append([x,v,angle,angle_dot])
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
            angle_input = float(ln[3].strip())
            angle_dot_input = float(ln[4].strip())
            a_base = float(ln[5].strip())
            x_output = float(ln[6].strip())
            v_output = float(ln[7].strip())
            angle_output = float(ln[8].strip())
            angle_dot_output = float(ln[9].strip())
            input_data.append([delta_T,x_input,v_input,angle_input,angle_dot_input,a_base])
            output_data.append([x_output,v_output,angle_output,angle_dot_output])
        data.close()
        return input_data,output_data

    # saves collective input-output data
    def save_model_data(self,input_output_data,data_file):
        data = open(data_file, "w")
        for row in input_output_data:
            delta_T = row[0]
            x_input = row[1]
            v_input = row[2]
            theta_input = row[3]
            theta_dot_input = row[4]
            a_base = row[5]
            x_output = row[6]
            v_output = row[7]
            theta_output = row[8]
            theta_dot_output = row[9]
            data.write(str(delta_T)+","+str(x_input)+","+str(v_input)+","+str(theta_input)+","+str(theta_dot_input)+","+str(a_base)+","+str(x_output)+","+str(v_output)+","+str(theta_output)+","+str(theta_dot_output)+"\n")
        data.close()


    def evaluate_policy(self,policy,start_state):
        time_step = 0.01
        total_time = 5
        delta_T = np.array([time_step])
        current_state = start_state
        #current_state = np.array([1.0,0.0,0.02123817989101573,-0.009558798511813068])
        #current_state = current_state.reshape(-1,1)
        #action = np.array([a*current_state[0]+b*current_state[1]+c*current_state[2]+d*current_state[3]])
        state_vector = [current_state[0],current_state[1],current_state[2],current_state[3]]
        action = np.array([self.calculate_action(state_vector,policy)])
        #print("action")
        #print(action)
        #print(np.concatenate((delta_T,current_state,action), axis=0))
        num_steps = int(total_time/time_step)
        #predicted_states = []
        future_state = current_state
        trajectory = []
        output_data_file = "Pilco_trajectory.txt"
        #output_data = open(output_data_file, "w")
        for _ in range(num_steps):
            #output_data.write(str(time_step)+","+str(float(future_state[0]))+","+str(float(future_state[1]))+","+str(float(future_state[2]))+","+str(float(future_state[3]))+","+str(action[0])+"\n")
            #print(self.scaler.transform([np.concatenate((delta_T,future_state,action), axis=0)]))
            #tf.config.run_functions_eagerly(True)
            scaled_current_state = tf.convert_to_tensor(self.scaler_x.transform([np.concatenate((delta_T,future_state,action), axis=0)]))
            #tf.config.run_functions_eagerly(False)
            #print(scaled_current_state)
            scaled_future_state = self.model.predict_f(scaled_current_state)[0][0]
            future_state = self.scaler_y.inverse_transform(np.array([scaled_future_state]))[0]
            #print(future_state)
            #print(np.array([np.concatenate((delta_T,future_state,action), axis=0)]))
            #future_state = self.model.predict_f(np.array([np.concatenate((delta_T,future_state,action), axis=0)]))[0][0]
            #print(future_state[0][0,1])
            #print("action")
            #action = [a*future_state[0]+b*future_state[1]+c*future_state[2]+d*future_state[3]]
            state_vector = [future_state[0],future_state[1],future_state[2],future_state[3]]
            action = np.array([self.calculate_action(state_vector,policy)])
            #print("action")
            #print(action)
            trajectory.append(future_state)
        #output_data.close()
        #predicted_states.append(trajectory)
            
            # Here, you would compute a cost function based on the predicted trajectory
            # For simplicity, let's assume a simple cost function:
        cost = np.array([time_step*np.sum(np.array(trajectory)**2)])
        #print("final_state")
        #print(future_state)
        #print("a:"+str(policy['a'])+"b:"+str(policy['b'])+"c:"+str(policy['c']))
        print("cost")
        print(cost)
        return cost

    @tf.function
    def evaluate_policies(self, a_tensor, b_tensor, c_tensor, )
    def write_policy(self,policy):
        policy_file = "policy_config.txt"
        with open(policy_file, 'w') as file:
            json.dump(policy, file)
        #policy = open(policy_file,"w")
        #policy.write("#proportional\n")
        #policy.write(str(a)+"\n")
        #policy.write(str(b)+"\n")
        #policy.write(str(c)+"\n")
        #policy.write(str(d)+"\n")
        #policy.close()

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
        print(self.sim_iteration)

        print("loading_input")
        input_data_1, output_data_1 = self.load_data("data.txt")
        print("loaded_input")
        #scaler = StandardScaler()
        #input_data_scaled = scaler.fit_transform(input_data)
        #input_data_scaled = input_data
        #rows,columns = input_data.shape()
        #print(self.input_data)
        if self.input_data is not None:
            self.input_data = np.vstack((self.input_data, input_data_1))
            self.output_data = np.vstack((self.output_data, output_data_1))
        else:
            self.input_data = input_data_1
            self.output_data = output_data_1
        
        input_output_data = np.hstack((self.input_data,self.output_data))
        np.random.shuffle(input_output_data)
        self.save_model_data(input_output_data,model_file)
        self.input_data = input_output_data[:,:6]
        self.output_data = input_output_data[:,6:10]
        # Set up the GP kernel (RBF kernel + constant kernel)
        #kernel = C(1.0, (1e-6, 15)) * RBF(1.0, (1e-6, 15))
        # Initialize GP regressor
        #print("fitting_model")
        #gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, alpha=10**(-2))
        #print("regress_done")
        # Fit the GP model to the data
        #gp.fit(input_data_scaled, output_data)
        #print("fitted_model")
        # Define a kernel (RBF kernel with a constant factor)
        #kernel = gpflow.kernels.SquaredExponential()
        # Create a GP model
        #model = gpflow.models.GPR(data=(np.array(input_data), np.array(output_data)), kernel=kernel)
        # Optimize the model
        #optimizer = gpflow.optimizers.Scipy()
        #optimizer.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
        #return input_data,output_data


#my_Pilco_learn.write_policy(7.2,5.76,80,-30)
#my_Pilco_learn.reset_sim(1)
#print("simulation_reset")
#time.sleep(12)

#print("loading_input")
#input_data, output_data = my_Pilco_learn.load_data("data.txt")
#print("loaded_input")
#input_data_scaled = scaler.fit_transform(input_data)
#scaler = ""
#input_data_scaled = input_data
# Set up the GP kernel (RBF kernel + constant kernel)
#kernel = C(1.0, (1e-6, 15)) * RBF(1.0, (1e-6, 15))
# Initialize GP regressor
#print("fitting_model")
#gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, alpha=10**(-2))
#print("regress_done")
# Fit the GP model to the data
#gp.fit(input_data_scaled, output_data)
#print("fitted_model")
# Define a kernel (RBF kernel with a constant factor)
#kernel = gpflow.kernels.SquaredExponential()
kernel = gpflow.kernels.Matern12()
# Create a GP model
input_data = None
output_data = None
#model = None
model = gpflow.models.GPR(data=(np.array([[]]), np.array([[]])), kernel=kernel)
scaler_x = StandardScaler()
scaler_y = StandardScaler()
my_pilco_learn = Pilco_Learn(model,scaler_x,scaler_y,input_data,output_data)
#model = gpflow.models.GPR(data=(np.array([input_data_scaled]), np.array(output_data)), kernel=kernel)
# Optimize the model
#optimizer = gpflow.optimizers.Scipy()
#optimizer.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))

a1_values = [7.3,7.9,7.1,8.3]
a2_values = [5.7,5.8,6,6.1]
#a_values = [7.3,7.9]
#b_values = [5.7,6.1]
a3_values = [70,90]
a4_values = [-25,-35]
b = np.zeros((4,4)).tolist()
c = np.zeros((4,4,4)).tolist()
#for a1 in a1_values:
#    for a2 in a2_values:
#        for a3 in a3_values:
#            for a4 in a4_values:
#                a = [a1,a2,a3,a4]
#                policy = {
#                    "a":a,
#                    "b":b,
#                    "c":c
#                }
#                my_pilco_learn.add_policy_data(policy,"model.txt")
#                print(a,b,c)

for index in range(10):
    a = np.random.uniform(-100, 100, size=(4)).tolist()
    b = np.random.uniform(-10, 10, size=(4,4)).tolist()
    c = np.random.uniform(-1, 1, size=(4,4,4)).tolist()
    policy = {
                    "a":a,
                    "b":b,
                    "c":c
    }
    print(a,b,c)
    my_pilco_learn.add_policy_data(policy,"model.txt")
#tf.saved_model.save(model, 'gpflow_model')
#model = tf.saved_model.load('gpflow_model')
#model = gpflow.models.load_model('gpflow_model')
#input_data,output_data = my_pilco_learn.load_data_2("model.txt")
#model = gpflow.models.GPR(data=(np.array(input_data), np.array(output_data)), kernel=kernel)
input_data,output_data = my_pilco_learn.load_data_2("model.txt")
input_data_scaled = tf.convert_to_tensor(my_pilco_learn.scaler_x.fit_transform(np.array(input_data)), dtype=tf.float64)
output_data_scaled = tf.convert_to_tensor(my_pilco_learn.scaler_y.fit_transform(np.array(output_data)), dtype=tf.float64)
print(input_data_scaled)
#input_data_scaled = input_data
my_pilco_learn.input_data = input_data
my_pilco_learn.output_data = output_data
#my_pilco_learn.kernel = kernel
my_pilco_learn.model = gpflow.models.GPR(data=(input_data_scaled, output_data_scaled), kernel=kernel)
optimizer = gpflow.optimizers.Scipy()
batches = math.floor(len(input_data_scaled)/100)
for batch_number in range(batches):
    input_batch = np.array(input_data_scaled[batch_number*100:(batch_number+1)*100])
    output_batch = np.array(output_data_scaled[batch_number*100:(batch_number+1)*100])
    print("test")
    print(batch_number)
        
    # Update the entire dataset for this iteration. 
    # Note: This might not be necessary if you're only updating with the batch
    # model.data = (X_batch, Y_batch)
    my_pilco_learn.model.data = (input_batch, output_batch)
    # Instead, use a closure for the current batch:
    closure = my_pilco_learn.model.training_loss_closure()
        
    # Optimize using the batch
    optimizer.minimize(closure, variables=my_pilco_learn.model.trainable_variables, options=dict(maxiter=10))
    #time.sleep(5)
    
#a,b,c,d = 7.2,5.76,80,-30
#policy = {
#                    "a":[7.2,5.76,80,-30],
#                    "b":b,
#                    "c":c
#        }
#my_Pilco_learn.evaluate_policy(policy,scaler,model)
##a,b,c,d = 7.225,5.76,80,-30
#policy = {
#                    "a":[7.225,5.76,80,-30],
#                    "b":b,
#                    "c":c
#        }
#my_Pilco_learn.evaluate_policy(policy,scaler,model)
#a,b,c,d = 7.25,5.76,80,-30
#my_Pilco_learn.evaluate_policy(policy,scaler,model)
#a,b,c,d = 7.275,5.76,80,-30
#my_Pilco_learn.evaluate_policy(policy,scaler,model)
#a,b,c,d = 7.9,8,80,-30
#my_Pilco_learn.evaluate_policy(policy,scaler,model)
#a,b,c,d = 8.3, 6.1, 90, -35
#my_Pilco_learn.evaluate_policy(policy,scaler,model)
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
    return my_pilco_learn.evaluate_policy(policy,np.array([1.0,0.0,0.02123817989101573,-0.009558798511813068]))

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
            self.cost = self.genetic_algorithm.calculate_cost(policy)

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
            "a" : initial_guess[:4],
            "b" : np.array(initial_guess[4:20]).reshape((4,4)).tolist(),
            "c" : np.array(initial_guess[20:]).reshape((4,4,4)).tolist()
        }
        return self.pilco_learn.evaluate_policy(policy,np.array([1.0,0.0,0.02123817989101573,-0.009558798511813068]))
    
    def add_individual(self, policy):
        new_individual = self.Individual(self, policy, self.individuals_birthed + 1)
        if self.individuals_birthed % 10 == 0:
            self.pilco_learn.add_policy_data(policy,"model.txt")
        self.population.append(new_individual)
        self.individuals_birthed += 1

    def print_population(self):
        for individual in self.population:
            print(individual.cost)
            print(individual.policy)

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
            if allele == 1:
                flat_policy_new.append(flat_policy_1[index]*mutation_factor)
            else:
                flat_policy_new.append(flat_policy_2[index]*mutation_factor)
        policy_new = {
            "a" : flat_policy_new[:4],
            "b" : np.array(flat_policy_new[4:20]).reshape((4,4)).tolist(),
            "c" : np.array(flat_policy_new[20:]).reshape((4,4,4)).tolist()
        }
        self.add_individual(policy_new)
        
    def new_generation(self, survival_threshold):
        #surviving_population = [individual for individual in self.population if individual.cost < survival_threshold]
        surviving_population = sorted(self.population, key=lambda x: x.cost)[:survival_threshold]
        #self.population = surviving_population
        self.population = []
        pairs = [comb for comb in combinations_with_replacement(surviving_population, 2)]
        for pair in pairs:
            self.mate(pair[0],pair[1])
    
my_genetic_algorithm = Genetic_Algorithm(my_pilco_learn)

#population = []
for index in range(20):
    a = np.random.uniform(-100, 100, size=(4)).tolist()
    b = np.random.uniform(-10, 10, size=(4,4)).tolist()
    c = np.random.uniform(-1, 1, size=(4,4,4)).tolist()
    policy = {
                    "a":a,
                    "b":b,
                    "c":c
    }
    my_genetic_algorithm.add_individual(policy)
    #population.append(policy)
    #print(a,b,c)
    #input_data_scaled,output_data = my_Pilco_learn.add_policy_data(policy,input_data_scaled,output_data)

print("gen1")
my_genetic_algorithm.print_population()
for index in range(25):
    input_data,output_data = my_pilco_learn.load_data_2("model.txt")
    input_data_scaled = tf.convert_to_tensor(my_pilco_learn.scaler_x.fit_transform(np.array(input_data)), dtype=tf.float64)
    output_data_scaled = tf.convert_to_tensor(my_pilco_learn.scaler_y.fit_transform(np.array(output_data)), dtype=tf.float64)
    #my_pilco_learn.model.data = (input_data_scaled, output_data)
    #my_pilco_learn.input_data = input_data
    #my_pilco_learn.output_data = output_data
    my_pilco_learn.input_data = input_data
    my_pilco_learn.output_data = output_data
    #my_pilco_learn.kernel = kernel
    my_pilco_learn.model = gpflow.models.GPR(data=(input_data_scaled, output_data_scaled), kernel=kernel)
    #my_pilco_learn.kernel = kernel
    #my_pilco_learn.model = gpflow.models.GPR(data=(np.array(input_data[:100]), np.array(output_data[:100])), kernel=my_pilco_learn.kernel)
    optimizer = gpflow.optimizers.Scipy()
    batches = math.floor(len(input_data_scaled)/100)
    for batch_number in range(batches):
        input_batch = np.array(input_data_scaled[batch_number*100:(batch_number+1)*100])
        output_batch = np.array(output_data_scaled[batch_number*100:(batch_number+1)*100])
        print("test")
        print(batch_number)
            
        # Update the entire dataset for this iteration. 
        # Note: This might not be necessary if you're only updating with the batch
        # model.data = (X_batch, Y_batch)
        my_pilco_learn.model.data = (input_batch, output_batch)
        # Instead, use a closure for the current batch:
        closure = my_pilco_learn.model.training_loss_closure()
            
        # Optimize using the batch
        optimizer.minimize(closure, variables=my_pilco_learn.model.trainable_variables, options=dict(maxiter=10))

    my_genetic_algorithm.new_generation(10)
    print("gen: "+str(index))
    my_genetic_algorithm.print_population()

#best_policy = minimize(objective_function,initial_guess, method='Nelder-Mead',options={
#                      'xtol': 100,  # More lenient tolerance for x
#                      'ftol': 100})
#print(best_policy)