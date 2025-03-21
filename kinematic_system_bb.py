import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')  # Ensure interactive backend

class KinematicSystem:
    def __init__(self, a=1, b=0.5, c=0.5, d=np.sqrt(3), max_frames=1000):
        """Initialize the kinematic system with constants and plot setup."""
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        # Animation data
        self.alphas_frames = []  # List of all alphas over time
        self.frame_rate = 50  # Frames per second
        self.total_frames = 0  # Frames currently in alphas_frames
        self.max_frames = max_frames  # Maximum frames to pre-allocate
        
        # Setup plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Plot elements
        self.points_main, = self.ax.plot([], [], [], 'ro', label='Main Points')
        self.points_001, = self.ax.plot([], [], [], 'bo', label='001 Points')
        self.points_01, = self.ax.plot([], [], [], 'go', label='01 Points')
        self.triangle_lines = [self.ax.plot([], [], [], 'b-')[0] for _ in range(3)]
        self.link_lines = [self.ax.plot([], [], [], 'orange')[0] for _ in range(6)]
        
        # Plot settings
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(0, 2)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Animated 3D Kinematic System')
        self.ax.legend()
        self.ax.set_box_aspect([1, 1, 1])
        
        # Animation setup with a fixed large frame range
        self.ani = FuncAnimation(self.fig, self.update, frames=range(self.max_frames), 
                                init_func=self.init, blit=True, interval=1000/self.frame_rate)
        self.ani.pause()  # Start paused
        
    def calculate_betas(self, alphas):
        """Numerically solve for betas given alphas."""
        def beta_objective(betas):
            beta1, beta2, beta3 = betas
            x1 = self.a + self.b * np.cos(alphas[0]) + self.c * np.cos(beta1)
            y1 = 0.0
            z1 = self.b * np.sin(alphas[0]) + self.c * np.sin(beta1)
            x2 = np.cos(2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[1]) + self.c * np.cos(beta2))
            y2 = np.sin(2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[1]) + self.c * np.cos(beta2))
            z2 = self.b * np.sin(alphas[1]) + self.c * np.sin(beta2)
            x3 = np.cos(-2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[2]) + self.c * np.cos(beta3))
            y3 = np.sin(-2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[2]) + self.c * np.cos(beta3))
            z3 = self.b * np.sin(alphas[2]) + self.c * np.sin(beta3)
            d12 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
            d23 = np.sqrt((x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2)
            d31 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2 + (z3 - z1)**2)
            return [d12 - self.d, d23 - self.d, d31 - self.d]
        
        beta0 = np.array([np.pi/2, np.pi/2, np.pi/2])
        betas, info, ier, msg = fsolve(beta_objective, beta0, full_output=True, xtol=1e-10)
        if ier != 1:
            return np.full(3, np.nan)
        return betas

    def get_all_positions(self, alphas):
        """Return all positions including preceding points."""
        betas = self.calculate_betas(alphas)
        if np.any(np.isnan(betas)):
            return np.full((9, 3), np.nan)

        x001 = self.a + self.b
        y001 = 0.0
        z001 = 0.0
        x01 = self.a + self.b * np.cos(alphas[0])
        y01 = 0.0
        z01 = self.b * np.sin(alphas[0])
        x1 = self.a + self.b * np.cos(alphas[0]) + self.c * np.cos(betas[0])
        y1 = 0.0
        z1 = self.b * np.sin(alphas[0]) + self.c * np.sin(betas[0])
        
        x002 = np.cos(2 * np.pi / 3) * self.a
        y002 = np.sin(2 * np.pi / 3) * self.a
        z002 = 0.0 
        x02 = np.cos(2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[1]))
        y02 = np.sin(2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[1]))
        z02 = self.b * np.sin(alphas[1])   
        x2 = np.cos(2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[1]) + self.c * np.cos(betas[1]))
        y2 = np.sin(2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[1]) + self.c * np.cos(betas[1]))
        z2 = self.b * np.sin(alphas[1]) + self.c * np.sin(betas[1])

        x003 = np.cos(-2 * np.pi / 3) * self.a
        y003 = np.sin(-2 * np.pi / 3) * self.a
        z003 = 0.0
        x03 = np.cos(-2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[2]))
        y03 = np.sin(-2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[2]))
        z03 = self.b * np.sin(alphas[2])    
        x3 = np.cos(-2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[2]) + self.c * np.cos(betas[2]))
        y3 = np.sin(-2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[2]) + self.c * np.cos(betas[2]))
        z3 = self.b * np.sin(alphas[2]) + self.c * np.sin(betas[2])

        return np.array([
            [x001, y001, z001], [x01, y01, z01], [x1, y1, z1],
            [x002, y002, z002], [x02, y02, z02], [x2, y2, z2],
            [x003, y003, z003], [x03, y03, z03], [x3, y3, z3]
        ])

    def init(self):
        """Initialize animation with empty data."""
        self.points_main.set_data_3d([], [], [])
        self.points_001.set_data_3d([], [], [])
        self.points_01.set_data_3d([], [], [])
        for line in self.triangle_lines + self.link_lines:
            line.set_data_3d([], [], [])
        return [self.points_main, self.points_001, self.points_01] + self.triangle_lines + self.link_lines

    def update(self, frame):
        """Update animation for given frame."""
        #print(f"Frame: {frame}, Total: {self.total_frames}")
        
        if frame >= self.total_frames:
            self.ani.pause()
            #print("Animation paused at frame", frame)
            return [self.points_main, self.points_001, self.points_01] + self.triangle_lines + self.link_lines
        
        alphas = self.alphas_frames[frame]
        positions = self.get_all_positions(alphas)
        
        main_positions = positions[2::3]
        self.points_main.set_data_3d(main_positions[:, 0], main_positions[:, 1], main_positions[:, 2])
        
        self.points_001.set_data_3d(positions[0::3, 0], positions[0::3, 1], positions[0::3, 2])
        self.points_01.set_data_3d(positions[1::3, 0], positions[1::3, 1], positions[1::3, 2])
        
        self.triangle_lines[0].set_data_3d([positions[2, 0], positions[5, 0]], [positions[2, 1], positions[5, 1]], [positions[2, 2], positions[5, 2]])
        self.triangle_lines[1].set_data_3d([positions[5, 0], positions[8, 0]], [positions[5, 1], positions[8, 1]], [positions[5, 2], positions[8, 2]])
        self.triangle_lines[2].set_data_3d([positions[8, 0], positions[2, 0]], [positions[8, 1], positions[2, 1]], [positions[8, 2], positions[2, 2]])
        
        for i in range(3):
            self.link_lines[2*i].set_data_3d([positions[3*i, 0], positions[3*i+1, 0]], [positions[3*i, 1], positions[3*i+1, 1]], [positions[3*i, 2], positions[3*i+1, 2]])
            self.link_lines[2*i+1].set_data_3d([positions[3*i+1, 0], positions[3*i+2, 0]], [positions[3*i+1, 1], positions[3*i+2, 1]], [positions[3*i+1, 2], positions[3*i+2, 2]])
        
        return [self.points_main, self.points_001, self.points_01] + self.triangle_lines + self.link_lines

    def add_frames(self, start_alphas, end_alphas, delta_T):
        """Add new frames transitioning from start_alphas to end_alphas over delta_T seconds."""
        n_new_frames = int(self.frame_rate * delta_T)
        new_alphas = np.linspace(start_alphas, end_alphas, n_new_frames)
        self.alphas_frames.extend(new_alphas)
        self.total_frames = len(self.alphas_frames)
        if self.total_frames > self.max_frames:
            raise ValueError(f"Total frames ({self.total_frames}) exceeds max_frames ({self.max_frames})")
        self.ani.resume()  # Resume animation
        #print(f"Added {n_new_frames} frames, Total now: {self.total_frames}")

    def show(self):
        """Display the plot non-blocking."""
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)  # Non-blocking show
        plt.pause(0.1)  # Brief pause to ensure display updates
