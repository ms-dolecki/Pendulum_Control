import numpy as np
from scipy.optimize import minimize
import gpflow
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
#from sklearn.preprocessing import StandardScaler
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import tensorflow as tf
import json


class Pilco_learn:
    class Watcher(FileSystemEventHandler):
        def __init__(self,pilco_learn):
            self.pilco_learn = pilco_learn
        def on_modified(self, event):
            if event.is_directory:
                return
            self.pilco_learn.run_finish()

    def __init__(self):
        self.sim_iteration = 0
        path_to_watch = "sim_done.txt"  # Replace with your file or directory path
        event_handler = self.Watcher(self)
        observer = Observer()
        observer.schedule(event_handler, path=path_to_watch, recursive=False)  # Set recursive=True to monitor subdirectories
        observer.start()

    def vector_square(self,vector):
        output = []
        for element in vector:
            output.append([element*element1 for element1 in vector])
        return output
    
    def vector_cube(self,vector):
        output = []
        for element in vector:
            output.append([[element*element1 for element1 in row] for row in self.vector_square(vector)])
        return output
    
    def calculate_action(self, state, policy):
        p = np.array(state)
        p2 = np.array(self.vector_square(state))
        p3 = np.array(self.vector_cube(state))
        a = np.array(policy["a"])
        b = np.array(policy["b"])
        c = np.array(policy["c"])
        action = np.sum(a*p) + np.sum(b*p2) + np.sum(c*p3)
        return action
    
    def run_finish(self):
        print("run_finish")
        self.sim_iteration += 1

    # loads time-ordered state data
    def load_data(self,data_file):
        data = open(data_file, "r")
        input_data = []
        output_data = []
        for ln in data:
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

    def evaluate_policy(self,policy,scaler,gp):
        time_step = 0.1
        total_time = 10
        delta_T = np.array([time_step])
        current_state = np.array([1.0,0.0,0.02123817989101573,-0.009558798511813068])
        #current_state = current_state.reshape(-1,1)
        #action = np.array([a*current_state[0]+b*current_state[1]+c*current_state[2]+d*current_state[3]])
        state_vector = [current_state[0],current_state[1],current_state[2],current_state[3]]
        action = np.array([self.calculate_action(state_vector,policy)])
        #print(np.concatenate((delta_T,current_state,action), axis=0))
        num_steps = int(total_time/time_step)
        predicted_states = []
        future_state = current_state
        trajectory = []
        output_data_file = "Pilco_trajectory.txt"
        output_data = open(output_data_file, "w")
        for _ in range(num_steps):
            #output_data.write(str(time_step)+","+str(float(future_state[0]))+","+str(float(future_state[1]))+","+str(float(future_state[2]))+","+str(float(future_state[3]))+","+str(action)+"\n")
            #future_state = gp.predict_f(scaler.transform([np.concatenate((delta_T,future_state,action), axis=0)]))[0][0]
            #print(np.array([np.concatenate((delta_T,future_state,action), axis=0)]))
            future_state = gp.predict_f(np.array([np.concatenate((delta_T,future_state,action), axis=0)]))[0][0]
            #print(future_state[0][0,1])
            #print("action")
            #action = [a*future_state[0]+b*future_state[1]+c*future_state[2]+d*future_state[3]]
            state_vector = [future_state[0],future_state[1],future_state[2],future_state[3]]
            action = np.array([self.calculate_action(state_vector,policy)])
            trajectory.append(future_state)
        output_data.close()
        predicted_states.append(trajectory)
            
            # Here, you would compute a cost function based on the predicted trajectory
            # For simplicity, let's assume a simple cost function:
        cost = np.array([time_step*np.sum(np.array(trajectory)**2) for trajectory in predicted_states])
        print("final_state")
        print(future_state)
        print("a:"+str(policy['a'])+"b:"+str(policy['b'])+"c:"+str(policy['c']))
        print("cost")
        print(cost)
        return cost

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
        
    def add_policy_data(self,policy,input_data,output_data):
        self.write_policy(policy)
        current_sim_iteration = self.sim_iteration
        self.reset_sim(1)
        #print("simulation_reset")
        while self.sim_iteration == current_sim_iteration:
            #print(self.sim_iteration)
            pass
        print(self.sim_iteration)

        print("loading_input")
        input_data_1, output_data_1 = self.load_data("data.txt")
        print("loaded_input")
        #scaler = StandardScaler()
        #input_data_scaled = scaler.fit_transform(input_data)
        #input_data_scaled = input_data
        #rows,columns = input_data.shape()
        if input_data is not None:
            input_data = np.vstack((input_data, input_data_1))
            output_data = np.vstack((output_data, output_data_1))
        else:
            input_data = input_data_1
            output_data = output_data_1
        
        input_output_data = np.hstack((input_data,output_data))
        np.random.shuffle(input_output_data)
        self.save_model_data(input_output_data,"model.txt")
        input_data = input_output_data[:,:6]
        output_data = input_output_data[:,6:10]
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
        return input_data,output_data

my_Pilco_learn = Pilco_learn()
#my_Pilco_learn.write_policy(7.2,5.76,80,-30)
#my_Pilco_learn.reset_sim(1)
#print("simulation_reset")
#time.sleep(12)

#print("loading_input")
#input_data, output_data = my_Pilco_learn.load_data("data.txt")
#print("loaded_input")
# = StandardScaler()
#input_data_scaled = scaler.fit_transform(input_data)
scaler = ""
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
kernel = gpflow.kernels.SquaredExponential()
# Create a GP model
input_data_scaled = None
output_data = None
model = None
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
#                #input_data_scaled,output_data = my_Pilco_learn.add_policy_data(policy,input_data_scaled,output_data)
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
    input_data_scaled,output_data = my_Pilco_learn.add_policy_data(policy,input_data_scaled,output_data)
#tf.saved_model.save(model, 'gpflow_model')
#model = tf.saved_model.load('gpflow_model')
#model = gpflow.models.load_model('gpflow_model')
#input_data,output_data = my_Pilco_learn.load_data_2("model.txt")
#model = gpflow.models.GPR(data=(np.array(input_data), np.array(output_data)), kernel=kernel)
input_data,output_data = my_Pilco_learn.load_data_2("model.txt")
model = gpflow.models.GPR(data=(np.array(input_data[:100]), np.array(output_data[:100])), kernel=kernel)
optimizer = gpflow.optimizers.Scipy()
for batch_number in range(9):
    input_batch = np.array(input_data[batch_number*100:(batch_number+1)*100])
    output_batch = np.array(output_data[batch_number*100:(batch_number+1)*100])
    print("test")
    print(batch_number)
        
    # Update the entire dataset for this iteration. 
    # Note: This might not be necessary if you're only updating with the batch
    # model.data = (X_batch, Y_batch)
    model.data = (input_batch, output_batch)
    # Instead, use a closure for the current batch:
    closure = model.training_loss_closure()
        
    # Optimize using the batch
    optimizer.minimize(closure, variables=model.trainable_variables, options=dict(maxiter=10))
    #time.sleep(5)
    
#a,b,c,d = 7.2,5.76,80,-30
policy = {
                    "a":[7.2,5.76,80,-30],
                    "b":b,
                    "c":c
        }
my_Pilco_learn.evaluate_policy(policy,scaler,model)
#a,b,c,d = 7.225,5.76,80,-30
policy = {
                    "a":[7.225,5.76,80,-30],
                    "b":b,
                    "c":c
        }
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
    return my_Pilco_learn.evaluate_policy(policy,scaler,model)

best_policy = minimize(objective_function,initial_guess, method='Nelder-Mead')
print(best_policy)