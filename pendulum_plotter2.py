import time
import sys
from math import sin, cos
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt5.QtWidgets import *
from copy import deepcopy
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PyQt5.QtCore import QThread, pyqtSignal

# Worker thread class
class Worker(QThread):
    # Define the signal with an argument (a string in this case)
    update_signal = pyqtSignal(float,float)

    def run(self):
        data_file = "Pilco_trajectory.txt"
        data = open(data_file, "r")
        input_data = []
        output_data = []
        for ln in data:
            ln.strip()
            ln = ln.strip("\n").split(",")
            delta_T = float(ln[0].strip())
            x = float(ln[1].strip())
            #v = float(ln[2].strip())
            angle = float(ln[3].strip())
            #angle_dot = float(ln[4].strip())
            #a_base = float(ln[5].strip())
            #input_data.append([delta_T,x,v,angle,angle_dot,a_base])
            #output_data.append([x,v,angle,angle_dot])
            self.update_signal.emit(x,angle)
            time.sleep(delta_T)
        data.close()

class Pendulum_Plotter:
    def __init__(self, x_pixel_range, y_pixel_range, padding_factor):
        #self.pendulum_variables = pendulum_vars

        self.pendulum_length = 2
        #for i in range(len(self.pendulum_variables.radii_vector)):
         #   self.pendulum_length += self.pendulum_variables.radii_vector[i][0]
        self.win = QWidget()
        self.grid = QGridLayout()
        self.plot_win = pg.GraphicsLayoutWidget(show=True, title="Pendulum Sim")
        self.plot_win.resize(x_pixel_range,y_pixel_range)
        self.plot = self.plot_win.addPlot(row=0, col=0, title="Real-Time Plot", padding=0)
        self.plot.setAspectLocked()
        self.plot.enableAutoRange('xy', False)
        self.plot.setXRange(-self.pendulum_length*(x_pixel_range/y_pixel_range)*padding_factor, self.pendulum_length*(x_pixel_range/y_pixel_range)*padding_factor, padding=0)
        self.plot.setYRange(-self.pendulum_length*padding_factor, self.pendulum_length*padding_factor, padding=0)
        self.pendulum_plot = self.plot.plot([0],[0], pen='y', symbol='o', symbolBrush='r')
        self.grid.addWidget(self.plot_win, 0,0)
        self.reset_button = QPushButton("reset")
        self.reset_button.clicked.connect(self.reset)
        path_to_watch = "test.txt"  # Replace with your file or directory path
        event_handler = self.Watcher(self)
        observer = Observer()
        observer.schedule(event_handler, path=path_to_watch, recursive=False)  # Set recursive=True to monitor subdirectories
        observer.start()
        self.grid.addWidget(self.reset_button,1,0)
        self.win.setLayout(self.grid)
        #self.plot.hideAxis("left")
        #self.winBig.width = 1000
        #self.winBig.height = 500
        self.win.show()
        #self.button_box = self.win.addViewBox(row=1, col=0)
        #self.win.setRowHeight(0,600)
        #self.win.setRowHeight(1,100)
        #timer = QtCore.QTimer()
        #timer.timeout.connect(self.update_plot)
        #timer.start(100)
        self.start = True
         # Create an instance of the worker thread
        self.worker = Worker()
        # Connect the signal to the slot with an argument
        self.worker.update_signal.connect(self.update_plot)
        self.worker.start()  # Start the worker thread
        pg.exec()

    class Watcher(FileSystemEventHandler):
        def __init__(self,pendulum_plotter):
            self.pendulum_plotter = pendulum_plotter
        def on_modified(self, event):
            if event.is_directory:
                return
            self.pendulum_plotter.reset()

    def reset(self):
        #self.pendulum_variables.running = False
        #self.pendulum_variables.x = 1
        #self.pendulum_variables.angles_vector = deepcopy(self.pendulum_variables.angles_vector_0)
        #self.pendulum_variables.angle_dots_vector = deepcopy(self.pendulum_variables.angle_dots_vector_0)
        #self.pendulum_variables.time_0 = time.time()
        #self.pendulum_variables.initial_time = time.time()
        #self.pendulum_variables.sample_time = time.time()
        #self.pendulum_variables.open_file('data.txt')
        #self.pendulum_variables.load_policy()
        print("reset")
        #print("self.pendulum_variables.initial_time")
        #print(self.pendulum_variables.initial_time)
        #self.pendulum_variables.old_time = time.time()
        #self.pendulum_variables.running = True
    
    def update_plot(self,x,theta):
        #self.pendulum_variables.window_x = self.plot.getViewBox().screenGeometry().x()-20
        #self.pendulum_variables.x_pixel_range = self.plot.getViewBox().screenGeometry().width()
        #self.pendulum_variables.x_axis_left = self.plot.getViewBox().viewRect().x()
        #self.pendulum_variables.x_axis_range = self.plot.getViewBox().viewRect().width()   
        #print(self.plot.viewGeometry().x())
        
        x = [x]
        y = [0]
        radius = 2
        for i in range(1):
            x.append(x[-1]+radius*sin(theta))
            y.append(y[-1]+radius*cos(theta))
        self.pendulum_plot.setData(x,y)
        
        if self.start:
            #self.pendulum_variables.time_0 = time.time()
            #self.pendulum_variables.running = True
            self.start = False

app = QApplication(sys.argv)
pendulum_plot = Pendulum_Plotter(1200, 600, 1.2)