import argparse

# get options from command line
parser = argparse.ArgumentParser(
                    prog='cost_function',
                    description='calculate the cost of sampled data')                  
parser.add_argument('--data_file', type=str)
parser.add_argument('--cost_file', type=str)
args = parser.parse_args()
data_file = args.data_file
cost_file = args.cost_file
pendulum_data = open(data_file, "r")
cost_output = open(cost_file, "w")


x_array = []
v_array = []
angle_array = []
angle_dot_array = []
a_base_array = []
cost = 0
               

for ln in pendulum_data:
    ln.strip()
    ln = ln.strip("\n").split(",")
    delta_T = float(ln[0].strip())
    x = float(ln[1].strip())
    v = float(ln[2].strip())
    angle = float(ln[3].strip())
    angle_dot = float(ln[4].strip())
    a_base = float(ln[5])
    x_array.append([x])
    v_array.append([v])
    angle_array.append([angle])
    angle_dot_array.append([angle_dot])
    a_base_array.append([a_base])
    cost+= (x**2 + v**2 + angle**2 + angle_dot**2)*delta_T

cost_output.write(str(cost))


