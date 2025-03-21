import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Constants
a = 1.0
b = 0.5
c = 0.5
d = np.sqrt(3)

def calculate_betas(alphas, a, b, c, d):
    def beta_objective(betas):
        beta1, beta2, beta3 = betas
        x1 = a + b * np.cos(alphas[0]) + c * np.cos(beta1)
        y1 = 0.0
        z1 = b * np.sin(alphas[0]) + c * np.sin(beta1)
        
        x2 = np.cos(2 * np.pi / 3) * (a + b * np.cos(alphas[1]) + c * np.cos(beta2))
        y2 = np.sin(2 * np.pi / 3) * (a + b * np.cos(alphas[1]) + c * np.cos(beta2))
        z2 = b * np.sin(alphas[1]) + c * np.sin(beta2)
        
        x3 = np.cos(-2 * np.pi / 3) * (a + b * np.cos(alphas[2]) + c * np.cos(beta3))
        y3 = np.sin(-2 * np.pi / 3) * (a + b * np.cos(alphas[2]) + c * np.cos(beta3))
        z3 = b * np.sin(alphas[2]) + c * np.sin(beta3)
        
        d12 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        d23 = np.sqrt((x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2)
        d31 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2 + (z3 - z1)**2)
        
        return [d12 - d, d23 - d, d31 - d]
    
    beta0 = np.array([np.pi/2, np.pi/2, np.pi/2])
    betas, info, ier, msg = fsolve(beta_objective, beta0, full_output=True, xtol=1e-10)
    if ier != 1:
        return np.full(3, np.nan)
    return betas

def get_all_positions(alphas, a, b, c, d):
    betas = calculate_betas(alphas, a, b, c, d)
    if np.any(np.isnan(betas)):
        return np.full((9, 3), np.nan)

    # Point 1 chain
    x001 = a + b
    y001 = 0.0
    z001 = 0.0
    x01 = a + b * np.cos(alphas[0])
    y01 = 0.0
    z01 = b * np.sin(alphas[0])
    x1 = a + b * np.cos(alphas[0]) + c * np.cos(betas[0])
    y1 = 0.0
    z1 = b * np.sin(alphas[0]) + c * np.sin(betas[0])
    
    # Point 2 chain
    x002 = np.cos(2 * np.pi / 3) * a
    y002 = np.sin(2 * np.pi / 3) * a
    z002 = 0.0 
    x02 = np.cos(2 * np.pi / 3) * (a + b * np.cos(alphas[1]))
    y02 = np.sin(2 * np.pi / 3) * (a + b * np.cos(alphas[1]))
    z02 = b * np.sin(alphas[1])   
    x2 = np.cos(2 * np.pi / 3) * (a + b * np.cos(alphas[1]) + c * np.cos(betas[1]))
    y2 = np.sin(2 * np.pi / 3) * (a + b * np.cos(alphas[1]) + c * np.cos(betas[1]))
    z2 = b * np.sin(alphas[1]) + c * np.sin(betas[1])

    # Point 3 chain
    x003 = np.cos(-2 * np.pi / 3) * a
    y003 = np.sin(-2 * np.pi / 3) * a
    z003 = 0.0
    x03 = np.cos(-2 * np.pi / 3) * (a + b * np.cos(alphas[2]))
    y03 = np.sin(-2 * np.pi / 3) * (a + b * np.cos(alphas[2]))
    z03 = b * np.sin(alphas[2])    
    x3 = np.cos(-2 * np.pi / 3) * (a + b * np.cos(alphas[2]) + c * np.cos(betas[2]))
    y3 = np.sin(-2 * np.pi / 3) * (a + b * np.cos(alphas[2]) + c * np.cos(betas[2]))
    z3 = b * np.sin(alphas[2]) + c * np.sin(betas[2])

    return np.array([
        [x001, y001, z001], [x01, y01, z01], [x1, y1, z1],
        [x002, y002, z002], [x02, y02, z02], [x2, y2, z2],
        [x003, y003, z003], [x03, y03, z03], [x3, y3, z3]
    ])

# Static plot to debug positions
alphas_test = np.array([np.pi / 3, np.pi / 3, np.pi / 3])
positions = get_all_positions(alphas_test, a, b, c, d)
print("Positions:\n", positions)  # Debug output

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot points
main_positions = positions[2::3]
preceding_positions = positions[0::3]  # 001 points
intermediate_positions = positions[1::3]  # 01 points

ax.scatter(main_positions[:, 0], main_positions[:, 1], main_positions[:, 2], c='r', marker='o', label='Main Points')
ax.scatter(preceding_positions[:, 0], preceding_positions[:, 1], preceding_positions[:, 2], c='b', marker='o', label='001 Points')
ax.scatter(intermediate_positions[:, 0], intermediate_positions[:, 1], intermediate_positions[:, 2], c='g', marker='o', label='01 Points')

# Plot triangle lines
ax.plot([positions[2, 0], positions[5, 0]], [positions[2, 1], positions[5, 1]], [positions[2, 2], positions[5, 2]], 'b-', label='Triangle 1-2')
ax.plot([positions[5, 0], positions[8, 0]], [positions[5, 1], positions[8, 1]], [positions[5, 2], positions[8, 2]], 'b-', label='Triangle 2-3')
ax.plot([positions[8, 0], positions[2, 0]], [positions[8, 1], positions[2, 1]], [positions[8, 2], positions[2, 2]], 'b-', label='Triangle 3-1')

# Plot link lines
for i in range(3):
    ax.plot([positions[3*i, 0], positions[3*i+1, 0]], [positions[3*i, 1], positions[3*i+1, 1]], [positions[3*i, 2], positions[3*i+1, 2]], 'orange', label=f'Link {i+1}a')
    ax.plot([positions[3*i+1, 0], positions[3*i+2, 0]], [positions[3*i+1, 1], positions[3*i+2, 1]], [positions[3*i+1, 2], positions[3*i+2, 2]], 'orange', label=f'Link {i+1}b')

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Static 3D Kinematic System')
ax.legend()
ax.set_box_aspect([1, 1, 1])
plt.show()

# Animation setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

alphas_start = np.array([np.pi / 3, np.pi / 3, np.pi / 3])
alphas_end = np.array([np.pi / 4, np.pi / 2, np.pi / 2])  # Example end point, replace with your solved alphas
n_frames = 50
alphas_frames = np.linspace(alphas_start, alphas_end, n_frames)

points_main, = ax.plot([], [], [], 'ro', label='Main Points')
points_001, = ax.plot([], [], [], 'bo', label='001 Points')
points_01, = ax.plot([], [], [], 'go', label='01 Points')
triangle_lines = [ax.plot([], [], [], 'b-')[0] for _ in range(3)]
link_lines = [ax.plot([], [], [], 'orange')[0] for _ in range(6)]

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Animated 3D Kinematic System')
ax.legend()
ax.set_box_aspect([1, 1, 1])

def init():
    points_main.set_data_3d([], [], [])
    points_001.set_data_3d([], [], [])
    points_01.set_data_3d([], [], [])
    for line in triangle_lines + link_lines:
        line.set_data_3d([], [], [])
    return [points_main, points_001, points_01] + triangle_lines + link_lines

def update(frame):
    alphas = alphas_frames[frame]
    positions = get_all_positions(alphas, a, b, c, d)
    print(f"Frame {frame}, Alphas: {alphas}")  # Debug output
    
    # Main points
    main_positions = positions[2::3]
    points_main.set_data_3d(main_positions[:, 0], main_positions[:, 1], main_positions[:, 2])
    
    # Preceding points
    points_001.set_data_3d(positions[0::3, 0], positions[0::3, 1], positions[0::3, 2])
    points_01.set_data_3d(positions[1::3, 0], positions[1::3, 1], positions[1::3, 2])
    
    # Triangle lines
    triangle_lines[0].set_data_3d([positions[2, 0], positions[5, 0]], [positions[2, 1], positions[5, 1]], [positions[2, 2], positions[5, 2]])
    triangle_lines[1].set_data_3d([positions[5, 0], positions[8, 0]], [positions[5, 1], positions[8, 1]], [positions[5, 2], positions[8, 2]])
    triangle_lines[2].set_data_3d([positions[8, 0], positions[2, 0]], [positions[8, 1], positions[2, 1]], [positions[8, 2], positions[2, 2]])
    
    # Link lines
    for i in range(3):
        link_lines[2*i].set_data_3d([positions[3*i, 0], positions[3*i+1, 0]], [positions[3*i, 1], positions[3*i+1, 1]], [positions[3*i, 2], positions[3*i+1, 2]])
        link_lines[2*i+1].set_data_3d([positions[3*i+1, 0], positions[3*i+2, 0]], [positions[3*i+1, 1], positions[3*i+2, 1]], [positions[3*i+1, 2], positions[3*i+2, 2]])
    
    return [points_main, points_001, points_01] + triangle_lines + link_lines

ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=50)
plt.show()