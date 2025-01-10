import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def load_data(data_file):
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

input_data, output_data = load_data("data.txt")
# Set up the GP kernel (RBF kernel + constant kernel)
kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
# Initialize GP regressor
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
# Fit the GP model to the data
gp.fit(input_data, output_data)
a,b,c,d = 7.2,5.76,80,-30
time_step = 0.1
total_time = 10
delta_T = np.array([time_step])
current_state = np.array([1.0,0.0,0.02123817989101573,-0.009558798511813068])
#current_state = current_state.reshape(-1,1)
action = np.array([a*current_state[0]+b*current_state[1]+c*current_state[2]+d*current_state[3]])
print(np.concatenate((delta_T,current_state,action), axis=0))
num_steps = int(total_time/time_step)
predicted_states = []
future_state = current_state
trajectory = []
for _ in range(num_steps):
    future_state = gp.predict([np.concatenate((delta_T,future_state,action), axis=0)])[0]
    print(future_state)
    action = [a*future_state[0]+b*future_state[1]+c*future_state[2]+d*future_state[3]]
    trajectory.append(future_state)
predicted_states.append(trajectory)
    
    # Here, you would compute a cost function based on the predicted trajectory
    # For simplicity, let's assume a simple cost function:
cost = np.array([time_step*np.sum(np.array(trajectory)**2) for trajectory in predicted_states])
print(cost)
