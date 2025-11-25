"""
Rigorous tests for fractional SDE correctness and differentiability.
"""

import pytest
import torch
import numpy as np
from hpfracc.ml.neural_fsde import NeuralFractionalSDE
from hpfracc.solvers.sde_solvers import FractionalEulerMaruyama

class TestFractionalCorrectness:
    
    def test_differentiability(self):
        """
        Test that the NeuralFractionalSDE is differentiable end-to-end.
        """
        from hpfracc.ml.neural_fsde import NeuralFSDEConfig
        
        # Define a simple neural SDE
        input_dim = 1
        hidden_dim = 10
        output_dim = 1
        
        config = NeuralFSDEConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            fractional_order=0.5
        )
        
        model = NeuralFractionalSDE(config)
        
        # Inputs
        x0 = torch.tensor([[1.0]], requires_grad=True)
        t = torch.linspace(0, 1, 10)
        
        # Forward pass
        trajectory = model(x0, t, num_steps=20)
        
        # Loss function (e.g., minimize final value)
        loss = trajectory[-1].sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        # Parameters of the drift network should have gradients
        has_grad = False
        for name, param in model.drift_net.named_parameters():
            if param.grad is not None and torch.norm(param.grad) > 0:
                has_grad = True
                break
        
        assert has_grad, "Drift network parameters have no gradients!"
        assert x0.grad is not None, "Initial condition has no gradient!"
        
    def test_convergence_alpha_1(self):
        """
        Test that for alpha=1, the solver converges to standard exponential decay
        for dX = -lambda X dt.
        """
        # Define drift and diffusion
        lam = 1.0
        def drift(t, x):
            return -lam * x
            
        def diffusion(t, x):
            return 0.0 # Deterministic for analytical comparison
            
        x0 = np.array([1.0])
        t_span = (0, 1)
        
        # Analytical solution for alpha=1: x(t) = x0 * exp(-lambda * t)
        
        errors = []
        steps = [10, 20, 40, 80]
        
        solver = FractionalEulerMaruyama(fractional_order=1.0)
        
        for n in steps:
            sol = solver.solve(drift, diffusion, x0, t_span, num_steps=n)
            final_y = sol.y[-1, 0]
            true_y = x0[0] * np.exp(-lam * 1.0)
            errors.append(abs(final_y - true_y))
            
        # Check convergence order (should be approx 1 for Euler)
        # Ratio of errors should be around 2
        ratios = [errors[i]/errors[i+1] for i in range(len(errors)-1)]
        mean_ratio = np.mean(ratios)
        
        # Allow some slack, but it should be > 1.5
        assert mean_ratio > 1.5, f"Convergence ratio too low: {ratios}"
        
    def test_fractional_decay(self):
        """
        Test fractional decay dX = -X dt^alpha.
        Solution is Mittag-Leffler function E_alpha(-t^alpha).
        We verify that the solution is decreasing and bounded.
        """
        alpha = 0.5
        solver = FractionalEulerMaruyama(fractional_order=alpha)
        
        def drift(t, x):
            return -x
            
        def diffusion(t, x):
            return 0.0
            
        x0 = np.array([1.0])
        sol = solver.solve(drift, diffusion, x0, t_span=(0, 2), num_steps=100)
        
        # Check monotonicity (should be decreasing)
        y = sol.y[:, 0]
        assert np.all(np.diff(y) <= 1e-10), "Solution should be monotonically decreasing"
        
        # Check bounds (0 < y <= 1)
        assert np.all(y > -1e-5) and np.all(y <= 1.0 + 1e-5)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
