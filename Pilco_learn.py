import numpy as np
import gpflow
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
#from sklearn.preprocessing import StandardScaler
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import tensorflow as tf


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

    def run_finish(self):
        print("run_finish")
        self.sim_iteration += 1

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

    def evaluate_policy(self,a,b,c,d,scaler,gp):
        time_step = 0.1
        total_time = 10
        delta_T = np.array([time_step])
        current_state = np.array([1.0,0.0,0.02123817989101573,-0.009558798511813068])
        #current_state = current_state.reshape(-1,1)
        action = np.array([a*current_state[0]+b*current_state[1]+c*current_state[2]+d*current_state[3]])
        #print(np.concatenate((delta_T,current_state,action), axis=0))
        num_steps = int(total_time/time_step)
        predicted_states = []
        future_state = current_state
        trajectory = []
        output_data_file = "Pilco_trajectory.txt"
        output_data = open(output_data_file, "w")
        for _ in range(num_steps):
            output_data.write(str(time_step)+","+str(float(future_state[0]))+","+str(float(future_state[1]))+","+str(float(future_state[2]))+","+str(float(future_state[3]))+","+str(action)+"\n")
            #future_state = gp.predict_f(scaler.transform([np.concatenate((delta_T,future_state,action), axis=0)]))[0][0]
            #print(np.array([np.concatenate((delta_T,future_state,action), axis=0)]))
            future_state = gp.predict_f(np.array([np.concatenate((delta_T,future_state,action), axis=0)]))[0][0]
            #print(future_state[0][0,1])
            #print("action")
            action = [a*future_state[0]+b*future_state[1]+c*future_state[2]+d*future_state[3]]
            trajectory.append(future_state)
        output_data.close()
        predicted_states.append(trajectory)
            
            # Here, you would compute a cost function based on the predicted trajectory
            # For simplicity, let's assume a simple cost function:
        cost = np.array([time_step*np.sum(np.array(trajectory)**2) for trajectory in predicted_states])
        print("final_state")
        print(future_state)
        print("a:"+str(a)+"b:"+str(b)+"c:"+str(c)+"d:"+str(d))
        print("cost")
        print(cost)

    def write_policy(self,a,b,c,d):
        policy_file = "policy_config.txt"
        policy = open(policy_file,"w")
        policy.write("#proportional\n")
        policy.write(str(a)+"\n")
        policy.write(str(b)+"\n")
        policy.write(str(c)+"\n")
        policy.write(str(d)+"\n")
        policy.close()

    def reset_sim(self,num):
        reset_file = "test.txt"
        reset = open(reset_file,"w")
        reset.write(str(num))
        reset.close()
        

    def add_policy_data(self,a,b,c,d,input_data,output_data,kernel):
        self.write_policy(a,b,c,d)
        current_sim_iteration = self.sim_iteration
        self.reset_sim(1)
        #print("simulation_reset")
        while self.sim_iteration == current_sim_iteration:
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
        self.save_model_data(input_output_data,"model.txt")
        np.random.shuffle(input_output_data)
        input_data = input_output_data[:20000,:6]
        output_data = input_output_data[:20000,6:10]
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
        model = gpflow.models.GPR(data=(np.array(input_data), np.array(output_data)), kernel=kernel)
        # Optimize the model
        optimizer = gpflow.optimizers.Scipy()
        optimizer.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
        return model,input_data,output_data

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

a_values = [7.3,7.9,7.1,8.3]
b_values = [5.7,5.8,6,6.1]
a_values = [7.3,7.9]
b_values = [5.7,6.1]
c_values = [70,90]
d_values = [-25,-35]
for a in a_values:
    for b in b_values:
        for c in c_values:
            for d in d_values:
                model,input_data_scaled,output_data = my_Pilco_learn.add_policy_data(a,b,c,d,input_data_scaled,output_data,kernel)

#tf.saved_model.save(model, 'gpflow_model')
#model = tf.saved_model.load('gpflow_model')
#model = gpflow.models.load_model('gpflow_model')
input_data,output_data = my_Pilco_learn.load_data_2("model.txt")
model = gpflow.models.GPR(data=(np.array(input_data), np.array(output_data)), kernel=kernel)
a,b,c,d = 7.2,5.76,80,-30
my_Pilco_learn.evaluate_policy(a,b,c,d,scaler,model)
a,b,c,d = 7.225,5.76,80,-30
my_Pilco_learn.evaluate_policy(a,b,c,d,scaler,model)
a,b,c,d = 7.25,5.76,80,-30
my_Pilco_learn.evaluate_policy(a,b,c,d,scaler,model)
a,b,c,d = 7.275,5.76,80,-30
my_Pilco_learn.evaluate_policy(a,b,c,d,scaler,model)
a,b,c,d = 7.9,8,80,-30
my_Pilco_learn.evaluate_policy(a,b,c,d,scaler,model)