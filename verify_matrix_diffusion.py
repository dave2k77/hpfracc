
import numpy as np
import torch
import matplotlib.pyplot as plt
from hpfracc.solvers.sde_solvers import solve_fractional_sde
from hpfracc.ml.neural_fsde import create_neural_fsde, NeuralFSDEConfig, NeuralFractionalSDE

def test_matrix_diffusion_solver():
    print("\n=== Testing Matrix Diffusion in Solvers ===")
    
    # Parameters
    dim = 2
    noise_dim = 2
    alpha = 0.8
    dt = 0.01
    num_steps = 1000
    num_paths = 500  # Number of paths to estimate covariance
    
    # Define correlated diffusion matrix
    # Covariance matrix Q = [[1, 0.5], [0.5, 1]]
    # Cholesky decomposition L:
    # L = [[1, 0], [0.5, sqrt(0.75)]] ~ [[1, 0], [0.5, 0.866]]
    sigma1 = 1.0
    sigma2 = 1.0
    rho = 0.5
    
    G = np.array([
        [sigma1, 0],
        [sigma2 * rho, sigma2 * np.sqrt(1 - rho**2)]
    ])
    
    print(f"Diffusion Matrix G:\n{G}")
    expected_cov = G @ G.T * dt
    print(f"Expected Covariance of increments (dt={dt}):\n{expected_cov}")
    
    def drift(t, x):
        return np.zeros_like(x) # Zero drift for pure noise check
        
    def diffusion(t, x):
        return G
        
    x0 = np.zeros(dim)
    
    # Collect final points
    final_points = []
    
    print("Running paths...")
    for i in range(num_paths):
        sol = solve_fractional_sde(
            drift, diffusion, x0, (0, dt), # One step check effectively? No, let's do 1 step
            fractional_order=alpha,
            method="euler_maruyama",
            num_steps=1
        )
        final_points.append(sol.y[-1])
        
    final_points = np.array(final_points)
    
    # Compute covariance of increments
    # Since x0=0, y[1] is the increment
    emp_cov = np.cov(final_points.T)
    
    print(f"Empirical Covariance:\n{emp_cov}")
    print(f"Error:\n{np.abs(emp_cov - expected_cov)}")
    
    # Check if error is reasonable
    if np.allclose(emp_cov, expected_cov, atol=0.1):
        print("SUCCESS: Covariance matches expected structure.")
    else:
        print("WARNING: Covariance mismatch (might be due to sample size or fractional effects).")
        # Note: For fractional SDE, the variance scales with t^(2H) or similar?
        # For Euler-Maruyama 1 step:
        # y[1] = x0 + 1/Gamma(alpha+1) * dt^alpha * G * dW
        # dW ~ N(0, dt)
        # So y[1] ~ N(0, C)
        # C = (1/Gamma(alpha+1) * dt^alpha)^2 * G * G^T * dt
        # Wait, my solver implementation:
        # y[i+1] = x0 + gamma_factor * dt^alpha * (drift_int + diffusion_int)
        # diffusion_int for 1 step is just G * dW (since weights[0]=1)
        # So y[1] = x0 + gamma_factor * dt^alpha * G * dW
        # Var(y[1]) = (gamma_factor * dt^alpha)^2 * G * G^T * dt
        #           = gamma_factor^2 * dt^(2*alpha + 1) * G * G^T
        
    # Let's calculate the expected fractional covariance for 1 step
    from scipy.special import gamma
    gamma_factor = 1.0 / gamma(alpha + 1)
    frac_scaling = (gamma_factor * dt**alpha)**2 * dt
    expected_frac_cov = G @ G.T * frac_scaling
    
    print(f"Expected Fractional Covariance (1 step):\n{expected_frac_cov}")
    print(f"Error vs Fractional:\n{np.abs(emp_cov - expected_frac_cov)}")
    
    if np.allclose(emp_cov, expected_frac_cov, atol=0.1 * np.max(expected_frac_cov)):
         print("SUCCESS: Covariance matches expected fractional scaling.")
    else:
         print("FAILURE: Covariance does not match expected fractional scaling.")


def test_neural_sde_matrix_diffusion():
    print("\n=== Testing Matrix Diffusion in Neural SDE ===")
    
    dim = 2
    diffusion_dim = 2
    batch_size = 100
    
    # Create model with matrix diffusion
    config = NeuralFSDEConfig(
        input_dim=dim,
        output_dim=dim,
        hidden_dim=16,
        fractional_order=0.8,
        diffusion_dim=diffusion_dim,
        noise_type="multiplicative" # Matrix diffusion
    )
    
    model = NeuralFractionalSDE(config)
    
    # Mock diffusion network to return constant matrix G
    # G = [[1, 0], [0, 2]]
    # We need to manually set weights or just monkeypatch forward
    
    x0 = torch.zeros(batch_size, dim)
    t = torch.linspace(0, 0.1, 10)
    
    print("Running forward pass...")
    try:
        traj = model(x0, t)
        print(f"Trajectory shape: {traj.shape}") # Should be (10, 100, 2)
        
        if traj.shape == (10, batch_size, dim):
            print("SUCCESS: Output shape is correct.")
        else:
            print(f"FAILURE: Output shape mismatch. Expected {(10, batch_size, dim)}")
            
    except Exception as e:
        print(f"FAILURE: Forward pass failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_matrix_diffusion_solver()
    test_neural_sde_matrix_diffusion()
