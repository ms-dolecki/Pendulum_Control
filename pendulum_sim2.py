import random
import matplotlib.pyplot as plt
import math
import time
import numpy
from numpy.linalg import inv
from numpy import dot
from numpy import absolute
from numpy import sign
import argparse
from math import sin
from math import cos
import threading

# get options from command line
parser = argparse.ArgumentParser(
                    prog='Pendulum_Sim',
                    description='Simulate an n-pendulum')                  
parser.add_argument('--config_file', type=str)
args = parser.parse_args()

config_file = args.config_file
pendulum_config = open(config_file, "r")
masses = []
radii = []
angles = []
angle_dots = []
for ln in pendulum_config:
    ln.strip()
    if ln[0] != "#":
        ln = ln.strip("\n").split(",")
        mass = float(ln[0].strip())
        radius = float(ln[1].strip())
        angle = float(ln[2].strip())
        angle_dot = float(ln[3].strip())
        masses.append([mass])
        radii.append([radius])
        angles.append([angle])
        angle_dots.append([angle_dot])
        
masses_vector = numpy.array(masses)
radii_vector = numpy.array(radii)
angles_vector = numpy.array(angles)
angle_dots_vector = numpy.array(angle_dots)

def K_matrix(n, masses_v, radii_v, angles_v):
    K = numpy.zeros((n,n))
    for i in range(n):
        for j in range(n):
            mass_above_ij = 0
            for mass_index in range(max(i,j),n):
                mass_above_ij += masses_v[mass_index]
            angle_i = angles_v[i]
            angle_j = angles_v[j]
            radius_j = radii_v[j]
            K[i][j] = mass_above_ij*radius_j*(sin(angle_i)*cos(angle_j)-cos(angle_i)*sin(angle_j))
    return K

def L_matrix(n, masses_v, radii_v, angles_v):
    L = numpy.zeros((n,n))
    for i in range(n):
        for j in range(n):
            mass_above_ij = 0
            for mass_index in range(max(i,j),n):
                mass_above_ij += masses_v[mass_index]
            angle_i = angles_v[i]
            angle_j = angles_v[j]
            radius_j = radii_v[j]
            L[i][j] = mass_above_ij*radius_j*(sin(angle_i)*sin(angle_j)+cos(angle_i)*cos(angle_j))
    return L

def T_vector(n, masses_v, angles_v, external_acceleration_v):
    T = numpy.zeros((n,1))
    for i in range(n):
        mass_above_i = 0
        for mass_index in range(i,n):
            mass_above_i += masses_v[mass_index]
        angle_i = angles_v[i]
        T[i] = mass_above_i*(-sin(angle_i)*external_acceleration_v[0]+cos(angle_i)*external_acceleration_v[1])
    return T
                          
class Pendulum_Variables:
    def __init__(self, n, masses_v, radii_v, angles_v, angle_dots_v):
        self.n = n
        self.masses_vector = masses_v
        self.radii_vector = radii_v
        self.angles_vector = angles_v
        self.angle_dots_vector = angle_dots_v
        self.old_time = time.time()
    
    def calculate_KLT(self, angles_v, external_acceleration_v):
        K = K_matrix(self.n, self.masses_vector, self.radii_vector, angles_v)
        L = L_matrix(self.n, self.masses_vector, self.radii_vector, angles_v)
        Linv = inv(L)
        T = T_vector(self.n, self.masses_vector, angles_v, external_acceleration_v)
        return K, Linv, T
        
    def update(self):
        self.external_acceleration_v = [0,0*sin(13.3*time.time())-10]
        new_time = time.time()
        t = new_time - self.old_time
        print(t)
        # initial guess
        K, Linv, T = self.calculate_KLT(self.angles_vector, self.external_acceleration_v)
        angle_dot_squareds_vector = self.angle_dots_vector**2
        angle_dotdots_vector1 = dot(Linv,T) - dot(dot(Linv,K),angle_dot_squareds_vector)
        delta_angles_vector = self.angle_dots_vector*t + 0.5*angle_dotdots_vector1*(t**2)
        angles_vector2 = self.angles_vector + delta_angles_vector
        angle_dots_vector2 = self.angle_dots_vector + angle_dotdots_vector1*t
        # correction
        K, Linv, T = self.calculate_KLT(angles_vector2, self.external_acceleration_v)
        angle_dot_squareds_vector2 = angle_dots_vector2**2
        angle_dotdots_vector2 = dot(Linv,T) - dot(dot(Linv,K),angle_dot_squareds_vector2)
        corrected_delta_angles_vector = self.angle_dots_vector*t + 0.25*(angle_dotdots_vector1+angle_dotdots_vector2)*(t**2)
        self.angles_vector += corrected_delta_angles_vector
        #angle_dot_squareds_vector += 2*delta_angles_vector*angle_dotdots_vector
        #self.angle_dots_vector = sign(delta_angles_vector)*(absolute(angle_dot_squareds_vector)**0.5)
        self.angle_dots_vector += 0.5*(angle_dotdots_vector1+angle_dotdots_vector2)*t
        self.old_time = new_time

def update_pendulum_variables(pendulum_vars):
    while True:
        pendulum_vars.update()
        
pendulum_variables = Pendulum_Variables(3, masses_vector, radii_vector, angles_vector, angle_dots_vector)
print(pendulum_variables.angles_vector)
t1 = threading.Thread(target=update_pendulum_variables, args=[pendulum_variables])
t1.start()

fig, axs = plt.subplots()
plt.xlim(-4, 4)
plt.ylim(-4, 4)
axs.set_title('Pendulum Sim')
old_time = time.time()
x0 = pendulum_variables.radii_vector[0]*cos(pendulum_variables.angles_vector[0])
y0 = pendulum_variables.radii_vector[0]*sin(pendulum_variables.angles_vector[0])
x1 = x0 + pendulum_variables.radii_vector[1]*cos(pendulum_variables.angles_vector[1])
y1 = y0 + pendulum_variables.radii_vector[1]*sin(pendulum_variables.angles_vector[1])
x2 = x1 + pendulum_variables.radii_vector[2]*cos(pendulum_variables.angles_vector[2])
y2 = y1 + pendulum_variables.radii_vector[2]*sin(pendulum_variables.angles_vector[2])
#point0 = axs.scatter(x0,y0,color='blue')
#point1 = axs.scatter(x1,y1,color='blue')
#point2 = axs.scatter(x2,y2,color='blue')
for i in range(50000):
    new_time = time.time()
    t = new_time - old_time
    #pendulum_variables.update(t,9000*[0,sin(1000*new_time)-10])
    #print(new_time - old_time)
    old_time = new_time
    #time.sleep(0.4)
    #point0.remove()
    #point1.remove()
    #point2.remove()
    plt.cla()
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    x0 = pendulum_variables.radii_vector[0]*cos(pendulum_variables.angles_vector[0])
    y0 = pendulum_variables.radii_vector[0]*sin(pendulum_variables.angles_vector[0])
    x1 = x0 + pendulum_variables.radii_vector[1]*cos(pendulum_variables.angles_vector[1])
    y1 = y0 + pendulum_variables.radii_vector[1]*sin(pendulum_variables.angles_vector[1])
    x2 = x1 + pendulum_variables.radii_vector[2]*cos(pendulum_variables.angles_vector[2])
    y2 = y1 + pendulum_variables.radii_vector[2]*sin(pendulum_variables.angles_vector[2])
    #point0 = axs.scatter(x0,y0,color='blue')
    #point1 = axs.scatter(x1,y1,color='blue')
    #point2 = axs.scatter(x2,y2,color='blue')
    plt.plot([0, x0, x1, x2],[0+0*sin(13.3*time.time())/(13.3**2), y0+0*sin(13.3*time.time())/(13.3**2), y1+0*sin(13.3*time.time())/(13.3**2), y2], '-o')
    
    plt.draw()
    plt.pause(0.001)
    #print(t, pendulum_variables.angles_vector)
  