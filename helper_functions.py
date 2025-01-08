import time
from math import sin, cos, trunc
import numpy
from numpy.linalg import inv
from numpy import dot
import sys
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication
from copy import deepcopy

app = QApplication(sys.argv)

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
    def __init__(self, n, masses_v, radii_v, angles_v, angle_dots_v, policy_type, policy_v):
        # pendulum config
        self.n = n
        self.masses_vector_0 = masses_v
        self.radii_vector_0 = radii_v
        self.angles_vector_0 = angles_v
        self.angle_dots_vector_0 = angle_dots_v
        self.masses_vector = deepcopy(masses_v)
        self.radii_vector = deepcopy(radii_v)
        self.angles_vector = deepcopy(angles_v)
        self.angle_dots_vector = deepcopy(angle_dots_v)
        # window config
        self.x_axis_left = -2
        self.x_axis_range = 4
        self.x_pixel_range = 1200
        self.window_x = 600
        # cursor postion in simulation units
        self.mouse_x_convert = 0
        # base motion
        self.x = 0
        self.v = 0
        self.a_base = 0
        self.policy_type = policy_type
        self.policy_vector = policy_v
        print(policy_v)

        # pendulum acceleration due to base motion
        self.horizontal_acceleration = 0
        # running
        self.running = False

        self.data_file = open('data.txt', 'w')
        self.data_file.write(str(0)+","+str(self.x)+","+str(self.v)+","+str((3.14159/2 - (6.28318+(self.angles_vector[0][0] % 6.28318)) % 6.28318))+","+str(self.angle_dots_vector[0][0])+","+str(self.a_base)+"\n")
        # time
        self.time_0 = time.time()
        self.initial_time = time.time()
        self.sample_time = time.time()
    
    def calculate_KLT(self, angles_v, external_acceleration_v):
        K = K_matrix(self.n, self.masses_vector, self.radii_vector, angles_v)
        L = L_matrix(self.n, self.masses_vector, self.radii_vector, angles_v)
        Linv = inv(L)
        T = T_vector(self.n, self.masses_vector, angles_v, external_acceleration_v)
        return K, Linv, T
    
    def open_file(self,file_name):
        self.data_file = open(file_name, 'w')

    def update(self):
        time_1 = time.time()
        t = time_1 - self.time_0
        
        # accelerate base to follow cursor
        mouse_x = float(QCursor.pos().x())
        self.mouse_x_convert = self.x_axis_left+(mouse_x-self.window_x)*self.x_axis_range/self.x_pixel_range
        #print(mouse_x, self.mouse_x_convert)
        #a_base = 500*(self.mouse_x_convert - self.x) - 40*self.v
        #a_base = 3*(0.5 - self.x)
        #self.a_base = 1*(80*(0.09*(self.x+0.8*self.v) + 3.14159/2 - (6.28318+(self.angles_vector[0][0] % 6.28318)) % 6.28318) - 30*self.angle_dots_vector[0][0])
        #print(str(self.policy_type)+"pid")
        #print(str(self.policy_type) == "pid")
        #print(self.policy_vector)
        #print("yo2")
        #print(self.policy_vector[4])
        if str(self.policy_type) == "proportional":
            #print("in")
            #self.a_base = self.policy_vector[0]*(self.policy_vector[1]*(self.policy_vector[2]*(self.x+self.policy_vector[3]*self.v) + 3.14159/2 - (6.28318+(self.angles_vector[0][0] % 6.28318)) % 6.28318) - self.policy_vector[4]*self.angle_dots_vector[0][0])
            #print(self.policy_vector[1])
            #self.a_base = self.policy_vector[0]*(self.policy_vector[1]*(0.09*(self.x+0.8*self.v) + 3.14159/2 - (6.28318+(self.angles_vector[0][0] % 6.28318)) % 6.28318) - 30*self.angle_dots_vector[0][0])
            #a = self.policy_vector[0]*self.policy_vector[1]*self.policy_vector[2]
            #b = self.policy_vector[0]*self.policy_vector[1]*self.policy_vector[2]*self.policy_vector[3]
            #c = self.policy_vector[0]*self.policy_vector[1]
            #d = -self.policy_vector[0]*self.policy_vector[4]
            #print("a")
            #print(a)
            #print("b")
            #print(b)
            #print("c")
            #print(c)
            #print("d")
            #print(d)
            if time.time() - self.sample_time > 0.1: 
                a = self.policy_vector[0]
                b = self.policy_vector[1]
                c = self.policy_vector[2]
                d = self.policy_vector[3]
                self.a_base = a*self.x+b*self.v+c*(3.14159/2 - (6.28318+(self.angles_vector[0][0] % 6.28318)) % 6.28318)+d*self.angle_dots_vector[0][0]
                self.data_file.write(str(time.time()-self.sample_time)+","+str(self.x)+","+str(self.v)+","+str((3.14159/2 - (6.28318+(self.angles_vector[0][0] % 6.28318)) % 6.28318))+","+str(self.angle_dots_vector[0][0])+","+str(self.a_base)+"\n")
                self.sample_time = time.time()
        #print(a_base)
        self.x += self.v*t+0.5*self.a_base*(t**2)
        self.v += self.a_base*t
        self.horizontal_acceleration = -self.a_base
        
        # set external acceleration as if base were stationary
        self.external_acceleration_v = [self.horizontal_acceleration, -9.81]
        
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
        self.angle_dots_vector += 0.5*(angle_dotdots_vector1+angle_dotdots_vector2)*t
        
        self.time_0 = time_1
        #print(time_1 - self.initial_time)
        #print("time_1")
        #print(time_1)
        #print("self.initial_time")
        #print(self.initial_time)
        if time_1 - self.initial_time > 10:
            self.running = False
            self.data_file.close()
        time.sleep(0.00001)

