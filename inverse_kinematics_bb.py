import numpy as np
from scipy.optimize import fsolve

class InverseKinematics:
    def __init__(self, a=1, b=0.5, c=0.5, d=np.sqrt(3), alphas_baseline=np.array([np.pi/4, np.pi/4, np.pi/4])):
        """Initialize the inverse kinematics solver with constants and baseline."""
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.alphas_baseline = alphas_baseline
        self.z_centroid_target = self.calculate_z_centroid(alphas_baseline)

    def wrap_to_pi(self, angle):
        """Wrap angle to [-π, π] range."""
        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)
        return np.arctan2(sin_angle, cos_angle)

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

    def calculate_theta(self, alphas):
        """Calculate theta from alphas."""
        betas = self.calculate_betas(alphas)
        if np.any(np.isnan(betas)):
            return np.nan
        
        x1 = self.a + self.b * np.cos(alphas[0]) + self.c * np.cos(betas[0])
        y1 = 0.0
        z1 = self.b * np.sin(alphas[0]) + self.c * np.sin(betas[0])
        
        x2 = np.cos(2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[1]) + self.c * np.cos(betas[1]))
        y2 = np.sin(2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[1]) + self.c * np.cos(betas[1]))
        z2 = self.b * np.sin(alphas[1]) + self.c * np.sin(betas[1])
        
        x3 = np.cos(-2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[2]) + self.c * np.cos(betas[2]))
        y3 = np.sin(-2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[2]) + self.c * np.cos(betas[2]))
        z3 = self.b * np.sin(alphas[2]) + self.c * np.sin(betas[2])
        
        v12 = np.array([x2 - x1, y2 - y1, z2 - z1])
        v13 = np.array([x3 - x1, y3 - y1, z3 - z1])
        normal = np.cross(v12, v13)
        return np.arctan2(normal[1], normal[0])

    def calculate_gamma(self, alphas):
        """Calculate gamma from alphas."""
        betas = self.calculate_betas(alphas)
        if np.any(np.isnan(betas)):
            return np.nan
        
        x1 = self.a + self.b * np.cos(alphas[0]) + self.c * np.cos(betas[0])
        y1 = 0.0
        z1 = self.b * np.sin(alphas[0]) + self.c * np.sin(betas[0])
        
        x2 = np.cos(2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[1]) + self.c * np.cos(betas[1]))
        y2 = np.sin(2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[1]) + self.c * np.cos(betas[1]))
        z2 = self.b * np.sin(alphas[1]) + self.c * np.sin(betas[1])
        
        x3 = np.cos(-2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[2]) + self.c * np.cos(betas[2]))
        y3 = np.sin(-2 * np.pi / 3) * (self.a + self.b * np.cos(alphas[2]) + self.c * np.cos(betas[2]))
        z3 = self.b * np.sin(alphas[2]) + self.c * np.sin(betas[2])
        
        v12 = np.array([x2 - x1, y2 - y1, z2 - z1])
        v13 = np.array([x3 - x1, y3 - y1, z3 - z1])
        normal = np.cross(v12, v13)
        horizontal_comp = np.sqrt(normal[0]**2 + normal[1]**2)
        return np.arctan2(horizontal_comp, normal[2])

    def calculate_z_centroid(self, alphas):
        """Calculate z_centroid from alphas."""
        betas = self.calculate_betas(alphas)
        if np.any(np.isnan(betas)):
            return np.nan
        z1 = self.b * np.sin(alphas[0]) + self.c * np.sin(betas[0])
        z2 = self.b * np.sin(alphas[1]) + self.c * np.sin(betas[1])
        z3 = self.b * np.sin(alphas[2]) + self.c * np.sin(betas[2])
        return (z1 + z2 + z3) / 3

    def objective_3d(self, alphas, theta_target, gamma_target):
        """Objective function for inverse kinematics."""
        theta_calc = self.calculate_theta(alphas)
        gamma_calc = self.calculate_gamma(alphas)
        z_centroid_calc = self.calculate_z_centroid(alphas)
        if np.any(np.isnan([theta_calc, gamma_calc, z_centroid_calc])):
            return np.array([1e6, 1e6, 1e6])
        return np.array([
            theta_calc - theta_target,
            gamma_calc - gamma_target,
            z_centroid_calc - self.z_centroid_target
        ])

    def get_alphas(self, theta_target, gamma_target, alphas_guess=None):
        """Compute alphas from theta and gamma targets."""
        if alphas_guess is None:
            alphas_guess = self.alphas_baseline
        result = fsolve(
            self.objective_3d, 
            alphas_guess, 
            args=(self.wrap_to_pi(theta_target), self.wrap_to_pi(gamma_target)),
            xtol=1e-10, 
            maxfev=10000,
            full_output=True
        )
        alphas, infodict, ier, msg = result
        if ier != 1:
            print(f"fsolve failed: {msg}")
            return np.full(3, np.nan)
        return alphas

# Example usage
if __name__ == "__main__":
    ik = InverseKinematics()
    
    theta_target = 1.4
    gamma_target = 0.1
    alphas_guess = np.array([np.pi / 3, np.pi / 4, np.pi / 5])
    
    alphas = ik.get_alphas(theta_target, gamma_target, alphas_guess)
    
    print("Solved alphas:")
    print(alphas)
    
    # Verify
    theta_check = ik.calculate_theta(alphas)
    gamma_check = ik.calculate_gamma(alphas)
    z_centroid_check = ik.calculate_z_centroid(alphas)
    print("Computed theta, gamma, z_centroid:")
    print([theta_check, gamma_check, z_centroid_check])
    print("Target theta, gamma, z_centroid:")
    print([theta_target, gamma_target, ik.z_centroid_target])