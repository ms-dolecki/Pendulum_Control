import serial
from math import trunc

class Data_Output:
    def __init__(self, pendulum_vars, port_name):
        self.pendulum_variables = pendulum_vars
        self.port_name = port_name
        self.ser = ""
        self.connected = False
        try:
            self.ser = serial.Serial(self.port_name, 115200)
            self.connected = True
            print("connected")
        except:
            print("failed to connect")
            
    def output(self):    
        # get data to two decimal places
        message = ""
        m_message = 'm'+str(trunc(self.pendulum_variables.mouse_x_convert*100)/100)
        message += m_message
        x_message = 'x'+str(trunc(self.pendulum_variables.x*100)/100)
        message += x_message
        v_message = 'v'+str(trunc(self.pendulum_variables.v*100)/100)
        message += v_message
        a_message = 'a'+str(trunc(self.pendulum_variables.angles_vector[0][0]*100)/100)
        message += a_message
        w_message = 'w'+str(trunc(self.pendulum_variables.angle_dots_vector[0][0]*100)/100)
        message += w_message
        s_message = 's'+str(trunc(self.pendulum_variables.x_pixel_range/self.pendulum_variables.x_axis_range))
        message += s_message
        
        
        if self.connected:
            try:
                self.ser.write(bytes(message+'\n','utf-8'))
            except:
                self.connected = False
                print("failed to connect")
        else:
            try:
                self.ser = serial.Serial(self.port_name, 115200)
                self.connected = True
            except:
                print("failed to connect")

