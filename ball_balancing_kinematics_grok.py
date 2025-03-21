import numpy as np
from scipy.optimize import fsolve

# Constants
a = 1.0
b = 0.5
c = 0.5
d = np.sqrt(3)

def calculate_betas(alphas, a, b, c, d):
    """Numerically solve for betas given alphas."""
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
        return np.full(3, np.nan)  # Return NaN array on failure
    return betas

def calculate_theta(alphas, a, b, c, d):
    betas = calculate_betas(alphas, a, b, c, d)
    if np.any(np.isnan(betas)):
        return np.nan
    
    x1 = a + b * np.cos(alphas[0]) + c * np.cos(betas[0])
    y1 = 0.0
    z1 = b * np.sin(alphas[0]) + c * np.sin(betas[0])
    
    x2 = np.cos(2 * np.pi / 3) * (a + b * np.cos(alphas[1]) + c * np.cos(betas[1]))
    y2 = np.sin(2 * np.pi / 3) * (a + b * np.cos(alphas[1]) + c * np.cos(betas[1]))
    z2 = b * np.sin(alphas[1]) + c * np.sin(betas[1])
    
    x3 = np.cos(-2 * np.pi / 3) * (a + b * np.cos(alphas[2]) + c * np.cos(betas[2]))
    y3 = np.sin(-2 * np.pi / 3) * (a + b * np.cos(alphas[2]) + c * np.cos(betas[2]))
    z3 = b * np.sin(alphas[2]) + c * np.sin(betas[2])
    
    v12 = np.array([x2 - x1, y2 - y1, z2 - z1])
    v13 = np.array([x3 - x1, y3 - y1, z3 - z1])
    normal = np.cross(v12, v13)
    return np.arctan2(normal[1], normal[0])

def calculate_gamma(alphas, a, b, c, d):
    betas = calculate_betas(alphas, a, b, c, d)
    if np.any(np.isnan(betas)):
        return np.nan
    
    x1 = a + b * np.cos(alphas[0]) + c * np.cos(betas[0])
    y1 = 0.0
    z1 = b * np.sin(alphas[0]) + c * np.sin(betas[0])
    
    x2 = np.cos(2 * np.pi / 3) * (a + b * np.cos(alphas[1]) + c * np.cos(betas[1]))
    y2 = np.sin(2 * np.pi / 3) * (a + b * np.cos(alphas[1]) + c * np.cos(betas[1]))
    z2 = b * np.sin(alphas[1]) + c * np.sin(betas[1])
    
    x3 = np.cos(-2 * np.pi / 3) * (a + b * np.cos(alphas[2]) + c * np.cos(betas[2]))
    y3 = np.sin(-2 * np.pi / 3) * (a + b * np.cos(alphas[2]) + c * np.cos(betas[2]))
    z3 = b * np.sin(alphas[2]) + c * np.sin(betas[2])
    
    v12 = np.array([x2 - x1, y2 - y1, z2 - z1])
    v13 = np.array([x3 - x1, y3 - y1, z3 - z1])
    normal = np.cross(v12, v13)
    horizontal_comp = np.sqrt(normal[0]**2 + normal[1]**2)
    return np.arctan2(horizontal_comp, normal[2])

def calculate_z_centroid(alphas, a, b, c, d):
    betas = calculate_betas(alphas, a, b, c, d)
    if np.any(np.isnan(betas)):
        return np.nan
    z1 = b * np.sin(alphas[0]) + c * np.sin(betas[0])
    z2 = b * np.sin(alphas[1]) + c * np.sin(betas[1])
    z3 = b * np.sin(alphas[2]) + c * np.sin(betas[2])
    return (z1 + z2 + z3) / 3
        
                      
def objective_3d(alphas, theta_target, gamma_target, z_centroid_target, a, b, c, d):
    theta_calc = calculate_theta(alphas, a, b, c, d)
    gamma_calc = calculate_gamma(alphas, a, b, c, d)
    z_centroid_calc = calculate_z_centroid(alphas, a, b, c, d)
    # Handle NaN by returning a large penalty
    if np.any(np.isnan([theta_calc, gamma_calc, z_centroid_calc])):
        return np.array([1e6, 1e6, 1e6])
    return np.array([
        theta_calc - theta_target,
        gamma_calc - gamma_target,
        z_centroid_calc - z_centroid_target
    ])

def calculate_alphas(theta_target, gamma_target, z_centroid_target, a, b, c, d, alphas_guess):
    # Add bounds to keep alphas reasonable (e.g., [0, pi])
    bounds = [(0, np.pi/2)] * 3
    result = fsolve(
        objective_3d, 
        alphas_guess, 
        args=(theta_target, gamma_target, z_centroid_target, a, b, c, d),
        xtol=1e-10, 
        maxfev=10000,
        full_output=True
    )
    alphas, infodict, ier, msg = result
    if ier != 1:
        print(f"fsolve failed: {msg}")
    return alphas

# Main script
alphas_baseline = np.array([np.pi / 3, np.pi / 3, np.pi / 3])
z_centroid_target = calculate_z_centroid(alphas_baseline, a, b, c, d)

theta_target = 1.4
gamma_target = 0.1
alphas_guess = np.array([np.pi / 3, np.pi / 4, np.pi / 5])  # Good initial guess from MATLAB

alphas = calculate_alphas(theta_target, gamma_target, z_centroid_target, a, b, c, d, alphas_guess)

print("Solved alphas:")
print(alphas)

# Verify the solution
theta_check = calculate_theta(alphas, a, b, c, d)
gamma_check = calculate_gamma(alphas, a, b, c, d)
z_centroid_check = calculate_z_centroid(alphas, a, b, c, d)
print("Computed theta, gamma, z_centroid:")
print([theta_check, gamma_check, z_centroid_check])
print("Target theta, gamma, z_centroid:")
print([theta_target, gamma_target, z_centroid_target])

