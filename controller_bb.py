from kinematic_system_bb import KinematicSystem
from inverse_kinematics_bb import InverseKinematics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

def wrap_to_pi(angle):
    """Wrap angle to [-π, π] range."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

if __name__ == "__main__":
    # Initialize systems
    system = KinematicSystem(a=0.75, b=0.35, c=0.3, d=1.0*np.sqrt(3))
    ik = InverseKinematics(a=0.75, b=0.35, c=0.3, d=1.0*np.sqrt(3))
    
    # Initial conditions
    alphas_guess = np.array([np.pi / 5, np.pi / 6, np.pi / 7])
    alphas_start = np.array([np.pi / 3, np.pi / 3, np.pi / 3])
    theta_target = 0
    gamma_target = 0.05
    
    # Start with initial position
    system.add_frames(alphas_start, alphas_start, delta_T=0.1)
    
    # Trajectory loop - reduced for testing
    dt = 0.2  # Fewer frames per step
    theta_step = 2 * np.pi / 10  # Larger step, fewer iterations
    steps = 100  # Much shorter animation
    
    for i in range(steps):
        print(f"Step {i}")
        theta_target = wrap_to_pi(theta_target + theta_step)
        
        alphas_end = ik.get_alphas(theta_target, gamma_target, alphas_guess)
        print(f"Theta: {theta_target:.4f}, Gamma: {gamma_target:.4f}, Alphas: {alphas_end}")
        
        if not np.any(np.isnan(alphas_end)):
            system.add_frames(alphas_start, alphas_end, delta_T=dt)
            alphas_start = alphas_end.copy()
            alphas_guess = alphas_end.copy()
        else:
            print(f"Skipping NaN alphas at step {i}")
        
        time.sleep(0.01)  # Minimal delay
    
    # Save animation as GIF with progress feedback
    writer = animation.PillowWriter(fps=25, bitrate=500)
    system.fig.set_size_inches(6, 6)  # Smaller size
    print("Starting GIF save...")
    start_time = time.time()
    system.ani.save('kinematic_animation.gif', writer=writer, dpi=80, progress_callback=lambda i, n: print(f"Saving frame {i+1}/{n}"))
    save_duration = time.time() - start_time
    print(f"Animation saved as 'kinematic_animation.gif' in {save_duration:.2f} seconds")
    
    # Optional: Display
    system.show()
    plt.ioff()
    plt.show()