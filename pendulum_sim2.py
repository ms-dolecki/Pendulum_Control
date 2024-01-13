import numpy
import argparse
from helper_functions import *
from pendulum_plotter import *
from data_output import *
import threading
import serial
from copy import deepcopy

# get options from command line
parser = argparse.ArgumentParser(
                    prog='Pendulum_Sim',
                    description='Simulate an n-pendulum')                  
parser.add_argument('--config_file', type=str)
args = parser.parse_args()

# get pendulum variables from config file
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
        
masses_vector_0 = numpy.array(masses)
radii_vector_0 = numpy.array(radii)
angles_vector_0 = numpy.array(angles)
angle_dots_vector_0 = numpy.array(angle_dots)

# updates pendulum variables
def update_pendulum_variables(pendulum_vars):
    while True:
        if pendulum_vars.running:
            pendulum_vars.update()


# outputs pendulum data
def output_data(data_out):
    while True:
        data_out.output()
        time.sleep(0.002)
        

# initialize pendulum variables and update in separate thread
pendulum_variables = Pendulum_Variables(len(masses_vector_0), deepcopy(masses_vector_0), deepcopy(radii_vector_0), deepcopy(angles_vector_0), deepcopy(angle_dots_vector_0))
t1 = threading.Thread(target=update_pendulum_variables, args=[pendulum_variables])
t1.start()

# output pendulum data in separate thread
data_output = Data_Output(pendulum_variables, "/dev/cu.usbmodemHIDGD1")
t2 = threading.Thread(target=output_data, args=[data_output])
t2.start()

# plot the pendulum
pendulum_plot = Pendulum_Plotter(pendulum_variables, 1200, 600, 1.2)



       


