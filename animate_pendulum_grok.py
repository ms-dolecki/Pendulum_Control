import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import os
import argparse

class PendulumAnimator:
    def __init__(self, trajectory_file, lengths=[1.0]):
        self.lengths = lengths
        self.num_links = len(lengths)
        self.times = []
        self.x_positions = []
        self.x_dots = []
        self.angle_cos = [[] for _ in range(self.num_links)]
        self.angle_sin = [[] for _ in range(self.num_links)]
        self.theta_dots = [[] for _ in range(self.num_links)]
        self.load_trajectory(trajectory_file)
        
        x_min, x_max = min(self.x_positions), max(self.x_positions)
        total_length = sum(self.lengths)
        x_range = max(abs(x_min), abs(x_max), total_length) + 0.5
        y_range = total_length + 0.5
        
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-x_range, x_range)
        self.ax.set_ylim(-y_range, y_range)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title(f'Multi-Link Pendulum Animation: {os.path.basename(trajectory_file)}')
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        self.base_line, = self.ax.plot([], [], 'k-', lw=2)
        self.pendulum_lines = [self.ax.plot([], [], 'b-', lw=2)[0] for _ in range(self.num_links)]
        self.bobs = [self.ax.plot([], [], 'ro', ms=5)[0] for _ in range(self.num_links)]  # Smaller dots
        
    def load_trajectory(self, trajectory_file):
        if not os.path.exists(trajectory_file):
            raise FileNotFoundError(f"Trajectory file {trajectory_file} not found")
        
        with open(trajectory_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                time, x, x_dot = map(float, row[:3])
                self.times.append(time)
                self.x_positions.append(x)
                self.x_dots.append(x_dot)
                for i in range(self.num_links):
                    angle_cos, angle_sin, theta_dot = map(float, row[3 + i*3 : 6 + i*3])
                    self.angle_cos[i].append(angle_cos)
                    self.angle_sin[i].append(angle_sin)
                    self.theta_dots[i].append(theta_dot)
        print(f"Loaded trajectory with {len(self.times)} frames from {trajectory_file}")
    
    def init_animation(self):
        self.base_line.set_data([], [])
        for line, bob in zip(self.pendulum_lines, self.bobs):
            line.set_data([], [])
            bob.set_data([], [])
        return [self.base_line] + self.pendulum_lines + self.bobs
    
    def update(self, frame):
        base_x = self.x_positions[frame]
        self.base_line.set_data([base_x, base_x], [0, 0])
        
        x_prev = base_x
        y_prev = 0.0
        for i in range(self.num_links):
            bob_x = x_prev - self.lengths[i] * self.angle_sin[i][frame]
            bob_y = y_prev + self.lengths[i] * self.angle_cos[i][frame]
            self.pendulum_lines[i].set_data([x_prev, bob_x], [y_prev, bob_y])
            self.bobs[i].set_data([bob_x], [bob_y])
            print(f"Link {i}: Start ({x_prev:.2f}, {y_prev:.2f}), End ({bob_x:.2f}, {bob_y:.2f}), Length {self.lengths[i]}")  # Debug
            x_prev, y_prev = bob_x, bob_y
        
        return [self.base_line] + self.pendulum_lines + self.bobs
    
    def animate(self):
        anim = FuncAnimation(self.fig, self.update, frames=len(self.times),
                            init_func=self.init_animation, blit=True, interval=5)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate a multi-link pendulum trajectory from a CSV file")
    parser.add_argument('--trajectory_file', type=str, required=True, 
                        help="Path to the trajectory CSV file (e.g., policies/best_trajectory_gen_0.csv)")
    parser.add_argument('--lengths', type=float, nargs='+', default=[1.0], 
                        help="Lengths of pendulum links (e.g., --lengths 1.0 0.5)")
    args = parser.parse_args()
    
    animator = PendulumAnimator(args.trajectory_file, lengths=args.lengths)
    animator.animate()