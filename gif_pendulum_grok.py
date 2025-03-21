import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import os
import argparse

class DualPendulumAnimator:
    def __init__(self, pred_file, sim_file, length=0.5):
        self.length = length
        self.num_links = 1
        self.dt = 0.0125  # Hardcoded timestep from your sim
        
        # Load trajectories
        self.pred_times, self.pred_x, self.pred_cos_theta, self.pred_sin_theta = self.load_trajectory(pred_file, 'Predicted')
        self.sim_times, self.sim_x, self.sim_cos_theta, self.sim_sin_theta = self.load_trajectory(sim_file, 'Simulated')
        
        # Trim to shortest length
        self.min_length = min(len(self.pred_times), len(self.sim_times))
        self.pred_times = self.pred_times[:self.min_length]
        self.pred_x = self.pred_x[:self.min_length]
        self.pred_cos_theta = self.pred_cos_theta[:self.min_length]
        self.pred_sin_theta = self.pred_sin_theta[:self.min_length]
        self.sim_times = self.sim_times[:self.min_length]
        self.sim_x = self.sim_x[:self.min_length]
        self.sim_cos_theta = self.sim_cos_theta[:self.min_length]
        self.sim_sin_theta = self.sim_sin_theta[:self.min_length]
        
        # Debug: Print first few values
        print(f"Predicted first frame: x={self.pred_x[0]}, cos_theta={self.pred_cos_theta[0]}, sin_theta={self.pred_sin_theta[0]}")
        print(f"Simulated first frame: x={self.sim_x[0]}, cos_theta={self.sim_cos_theta[0]}, sin_theta={self.sim_sin_theta[0]}")
        
        # Set up figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        x_min = min(np.min(self.pred_x), np.min(self.sim_x))
        x_max = max(np.max(self.pred_x), np.max(self.sim_x))
        x_range = max(abs(x_min), abs(x_max), self.length) + 0.5
        y_range = self.length + 0.5
        self.ax.set_xlim(-x_range, x_range)
        self.ax.set_ylim(-y_range, y_range)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title('Pendulum Swing-Up: Predicted vs Simulated')
        self.ax.grid(True)
        self.ax.set_aspect('equal')

    def load_trajectory(self, trajectory_file, label):
        if not os.path.exists(trajectory_file):
            raise FileNotFoundError(f"Trajectory file {trajectory_file} not found")
        
        data = pd.read_csv(trajectory_file)
        times = data['time'].values
        x_positions = data['x'].values
        angle_cos = data['cos_theta'].values
        angle_sin = data['sin_theta'].values
        
        print(f"Loaded {label} trajectory with {len(times)} frames from {trajectory_file}")
        return times, x_positions, angle_cos, angle_sin

    def draw_frame(self, frame):
        # Clear the axes to force a full redraw
        self.ax.clear()
        self.ax.set_xlim(-3.5, 3.5)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title('Pendulum Swing-Up: Predicted vs Simulated')
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        # Predicted (blue)
        pred_base_x = self.pred_x[frame]
        pred_x_tip = pred_base_x - self.length * self.pred_sin_theta[frame]  # Flipped sin_theta sign
        pred_y_tip = self.length * self.pred_cos_theta[frame]
        self.ax.plot([pred_base_x, pred_base_x], [0, 0], 'k-', lw=1)
        self.ax.plot([pred_base_x, pred_x_tip], [0, pred_y_tip], 'b-', lw=3, label='Predicted')
        self.ax.plot([pred_x_tip], [pred_y_tip], 'bo', ms=10)
        print(f"Frame {frame} Predicted: Cart x={pred_base_x:.2f}, Tip x={pred_x_tip:.2f}, y={pred_y_tip:.2f}")
        
        # Simulated (red)
        sim_base_x = self.sim_x[frame]
        sim_x_tip = sim_base_x - self.length * self.sim_sin_theta[frame]  # Flipped sin_theta sign
        sim_y_tip = self.length * self.sim_cos_theta[frame]
        self.ax.plot([sim_base_x, sim_base_x], [0, 0], 'k-', lw=1)
        self.ax.plot([sim_base_x, sim_x_tip], [0, sim_y_tip], 'r-', lw=3, label='Simulated')
        self.ax.plot([sim_x_tip], [sim_y_tip], 'ro', ms=10)
        print(f"Frame {frame} Simulated: Cart x={sim_base_x:.2f}, Tip x={sim_x_tip:.2f}, y={sim_y_tip:.2f}")
        
        # Update time
        elapsed_time = frame * self.dt
        self.ax.text(0.05, 0.95, f'Time: {elapsed_time:.2f}s', transform=self.ax.transAxes)
        
        # Add legend
        self.ax.legend(loc='upper right')

    def create_gif(self, output_gif='pendulum_comparison.gif', fps=30, speedup=2):
        dt = 0.0125  # Hardcoded from your sim
        frame_skip = max(1, int(speedup / (fps * dt)))
        frames = range(0, self.min_length, frame_skip)
        total_frames = len(list(frames))
        print(f"Creating GIF with {total_frames} frames, frame_skip={frame_skip}")
        
        writer = imageio.get_writer(output_gif, fps=fps)
        for i, frame in enumerate(frames):
            self.draw_frame(frame)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            image = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype='uint8')
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]  # Convert RGBA to RGB
            writer.append_data(image)
            if i % 50 == 0:  # Progress update
                elapsed_time = frame * self.dt
                print(f"Processed frame {i}/{total_frames}, Time set to: {elapsed_time:.2f}s")
        writer.close()
        plt.close(self.fig)
        print(f"GIF saved as {output_gif}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate predicted vs simulated pendulum trajectories as a GIF")
    parser.add_argument('--pred_file', type=str, required=True, help="Path to predicted trajectory CSV")
    parser.add_argument('--sim_file', type=str, required=True, help="Path to simulated trajectory CSV")
    parser.add_argument('--output_gif', type=str, default='pendulum_comparison.gif', help="Output GIF filename")
    parser.add_argument('--fps', type=int, default=30, help="Frames per second")
    parser.add_argument('--speedup', type=float, default=2, help="Speedup factor")
    args = parser.parse_args()
    
    animator = DualPendulumAnimator(args.pred_file, args.sim_file, length=0.5)
    animator.create_gif(output_gif=args.output_gif, fps=args.fps, speedup=args.speedup)