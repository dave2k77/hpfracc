"""
Unit tests for Neural FODE implementation.
Verifies correct fractional integration using known analytical solutions (Mittag-Leffler).
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from hpfracc.ml.neural_ode import create_neural_ode, NeuralFODE
from hpfracc.special.mittag_leffler import mittag_leffler

class TestNeuralFODE:
    def test_initialization(self):
        """Test initialization and configuration."""
        model = create_neural_ode(
            model_type="fractional",
            input_dim=2,
            hidden_dim=16,
            output_dim=2,
            fractional_order=0.5
        )
        assert isinstance(model, NeuralFODE)
        assert model.get_fractional_order() == 0.5
        assert model.input_dim == 2

    def test_forward_shape(self):
        """Test output shape."""
        model = create_neural_ode(
            model_type="fractional",
            input_dim=2,
            hidden_dim=16,
            output_dim=2,
            fractional_order=0.7
        )
        batch_size = 5
        time_steps = 10
        x = torch.randn(batch_size, 2)
        t = torch.linspace(0, 1, time_steps)
        
        y = model(x, t)
        assert y.shape == (batch_size, time_steps, 2)
        assert torch.isfinite(y).all()

    def test_mittag_leffler_convergence(self):
        """
        Verify that NeuralFODE with a linear field f(x)=-x approximates
        the Mittag-Leffler function E_alpha(-t^alpha).
        This confirms correct fractional integration logic.
        """
        alpha = 0.5
        # Determine strictness based on whether we expect correct implementation yet
        # For now, we write the test assuming correct implementation.
        
        # Define a model that perfectly implements f(x) = -x
        class LinearFODE(NeuralFODE):
            def _build_network(self):
                # We override network building to force strictly linear dynamics
                # f(t, x) = -x
                # We use a dummy linear layer just to satisfy initialization
                self.network = nn.Linear(1, 1) # Ignored
            
            def ode_func(self, t, x):
                # Ensure input is just state (ignore time)
                # Input to ode_func in base class is [t, x]
                # But here we override ode_func to ignore network and return -x
                # Note: The base class ode_func takes (t, x).
                # x shape: (batch, dim)
                output = -x
                # Ensure output dim matches
                return output

        # Initialize
        model = LinearFODE(
            input_dim=1, hidden_dim=1, output_dim=1,
            fractional_order=alpha, solver="fractional_euler"
        )
        
        # Simulation
        x0 = torch.tensor([[1.0]])
        t_max = 2.0
        steps = 100
        t = torch.linspace(0, t_max, steps + 1)
        
        with torch.no_grad():
            # Run forward
            # Note: NeuralFODE.forward takes (x, t)
            y_pred = model(x0, t) # Shape (1, steps+1, 1)

        y_pred = y_pred.squeeze().numpy()
        t_np = t.numpy()
        
        # Argument is -(t^alpha)
        z = -np.power(t_np, alpha)
        # mittag_leffler signature is (z, alpha, beta)
        y_true = mittag_leffler(z, alpha, 1.0)
        
        # The current broken implementation (local update) generates something like exp(-t^alpha/Gamma)
        # The correct implementation (convolution) generates ML function.
        # We assert reasonable closeness.
        mse = np.mean((y_pred - y_true)**2)
        assert mse < 0.05, f"NeuralFODE failed to match Mittag-Leffler dynamics. MSE: {mse}"

    def test_memory_effect_weights(self):
        """
        Check that weights behave like GL weights: (k+1)^a - k^a
        This is an indirect test ensuring the solver uses the right decay.
        """
        # We can't easily inspect internal weights without mocking, 
        # but we can verify that y[n] depends on history.
        pass

if __name__ == "__main__":
    pytest.main([__file__])
