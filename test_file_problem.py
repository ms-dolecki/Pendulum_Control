import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def find_max(alpha):
    # constants
    m = 5
    b = 3
    k = 10
    K_P = 10
    K_D = 20
    #alpha = 100  # large positive constant

    # function that returns dy/dt
    def model(y, t):
        x, dxdt = y
        u = 0.5 * (1 + np.tanh(alpha * (t-0.01)))
        du_dt = 0.5 * alpha * (1 - np.tanh(alpha * (t-0.01))**2)
        e = u - x
        de_dt = du_dt - dxdt
        d2xdt2 = (K_P * e + K_D * de_dt - b * dxdt - k * x) / m
        return [dxdt, d2xdt2]
    
    # initial condition
    y0 = [0, 0]
    
    # time points
    t = np.linspace(0, 10, 1000)
    
    # solve ODE
    y = odeint(model, y0, t)
    x = y[:, 0]
    
    # find maximum value of x
    max_x = np.max(x)
    
    # round to nearest hundredth
    max_x = round(max_x, 2)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t, x, label='Position (x)')
    #plt.plot(t, dxdt, label='Velocity (dx/dt)')
    plt.title(f'System Response with alpha = {alpha}')
    plt.xlabel('Time (t)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    return max_x


alphas = [0.1,1,10,100,1000,10000,100000,1000000,100000000,100000000000]
max_values = [find_max(alpha) for alpha in alphas]
converged_max = max_values[-1]
print(converged_max)