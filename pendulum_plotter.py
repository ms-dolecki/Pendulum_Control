import time
from math import sin, cos
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt5.QtWidgets import *
from copy import deepcopy


class Pendulum_Plotter:
    def __init__(self, pendulum_vars, x_pixel_range, y_pixel_range, padding_factor):
        self.pendulum_variables = pendulum_vars

        self.pendulum_length = 0
        for i in range(len(self.pendulum_variables.radii_vector)):
            self.pendulum_length += self.pendulum_variables.radii_vector[i][0]
        
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
        self.grid.addWidget(self.reset_button,1,0)
        self.win.setLayout(self.grid)
        #self.plot.hideAxis("left")
        #self.winBig.width = 1000
        #self.winBig.height = 500
        self.win.show()
        #self.button_box = self.win.addViewBox(row=1, col=0)
        #self.win.setRowHeight(0,600)
        #self.win.setRowHeight(1,100)
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update_plot)
        timer.start(0.0001)
        self.start = True
        pg.exec()

    def reset(self):
        print("reset")
        self.pendulum_variables.running = False
        self.pendulum_variables.angles_vector = deepcopy(self.pendulum_variables.angles_vector_0)
        self.pendulum_variables.angle_dots_vector = deepcopy(self.pendulum_variables.angle_dots_vector_0)
        self.pendulum_variables.time_0 = time.time()
        self.pendulum_variables.old_time = time.time()
        self.pendulum_variables.running = True
    
    def update_plot(self):
        self.pendulum_variables.window_x = self.plot.getViewBox().screenGeometry().x()-20
        self.pendulum_variables.x_pixel_range = self.plot.getViewBox().screenGeometry().width()
        self.pendulum_variables.x_axis_left = self.plot.getViewBox().viewRect().x()
        self.pendulum_variables.x_axis_range = self.plot.getViewBox().viewRect().width()   
        #print(self.plot.viewGeometry().x())
        
        x = [self.pendulum_variables.x]
        y = [0]
        for i in range(len(self.pendulum_variables.radii_vector)):
            x.append(x[-1]+self.pendulum_variables.radii_vector[i][0]*cos(self.pendulum_variables.angles_vector[i]))
            y.append(y[-1]+self.pendulum_variables.radii_vector[i][0]*sin(self.pendulum_variables.angles_vector[i]))
        self.pendulum_plot.setData(x,y)
        
        if self.start:
            self.pendulum_variables.time_0 = time.time()
            self.pendulum_variables.running = True
            self.start = False

