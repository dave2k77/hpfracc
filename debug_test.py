
import torch
import numpy as np
from hpfracc.ml.neural_fsde import NeuralFractionalSDE
from hpfracc.solvers.sde_solvers import FractionalEulerMaruyama
import sys

def test_differentiability():
    print("Testing differentiability...")
    try:
        input_dim = 1
        hidden_dim = 10
        output_dim = 1
        
        model = NeuralFractionalSDE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            fractional_order=0.5
        )
        
        x0 = torch.tensor([[1.0]], requires_grad=True)
        t = torch.linspace(0, 1, 10)
        
        trajectory = model(x0, t, num_steps=20)
        print(f"Trajectory shape: {trajectory.shape}")
        
        loss = trajectory[-1].sum()
        loss.backward()
        
        has_grad = False
        for name, param in model.drift_net.named_parameters():
            if param.grad is not None and torch.norm(param.grad) > 0:
                has_grad = True
                break
        
        if has_grad and x0.grad is not None:
            print("Differentiability test PASSED")
        else:
            print("Differentiability test FAILED: No gradients")
            
    except Exception as e:
        print(f"Differentiability test ERROR: {e}")
        import traceback
        traceback.print_exc()

def test_convergence():
    print("\nTesting convergence...")
    try:
        lam = 1.0
        def drift(t, x):
            return -lam * x
            
        def diffusion(t, x):
            return 0.0
            
        x0 = np.array([1.0])
        t_span = (0, 1)
        
        solver = FractionalEulerMaruyama(fractional_order=1.0)
        
        errors = []
        steps = [10, 20, 40]
        
        for n in steps:
            sol = solver.solve(drift, diffusion, x0, t_span, num_steps=n)
            final_y = sol.y[-1, 0]
            true_y = x0[0] * np.exp(-lam * 1.0)
            error = abs(final_y - true_y)
            errors.append(error)
            print(f"Steps: {n}, Error: {error}")
            
        ratios = [errors[i]/errors[i+1] for i in range(len(errors)-1)]
        print(f"Convergence ratios: {ratios}")
        
        if np.mean(ratios) > 1.5:
            print("\n>>> CONVERGENCE TEST PASSED <<<")
        else:
            print("\n>>> CONVERGENCE TEST FAILED <<<")
            
    except Exception as e:
        print(f"Convergence test ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_differentiability()
    test_convergence()
