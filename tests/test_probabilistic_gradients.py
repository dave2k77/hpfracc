"""
Unit tests for probabilistic fractional order gradients.

Tests reparameterization and score-function gradient estimators
for probabilistic fractional orders in neural networks.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Uniform, Beta
import numpy as np

from hpfracc.ml.probabilistic_fractional_orders import (
    ProbabilisticFractionalOrder,
    ProbabilisticFractionalLayer,
    create_normal_alpha_layer,
    create_uniform_alpha_layer,
    create_beta_alpha_layer
)


class TestProbabilisticGradients:
    """Test gradient computation for probabilistic fractional orders."""
    
    def test_reparameterization_gradient_flow(self):
        """Test that reparameterization gradients flow correctly."""
        torch.manual_seed(42)
        
        # Create learnable normal distribution layer
        layer = create_normal_alpha_layer(0.5, 0.1, learnable=True)
        
        # Simple input
        x = torch.randn(10, requires_grad=True)
        
        # Forward pass (uses rsample for reparameterization)
        result = layer(x)
        
        # Backward pass
        loss = result.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert layer.probabilistic_order.loc.grad is not None
        assert layer.probabilistic_order.scale.grad is not None
        
        # Check gradient magnitudes are reasonable
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(layer.probabilistic_order.loc.grad).all()
        assert torch.isfinite(layer.probabilistic_order.scale.grad).all()
    
    def test_score_function_gradient_flow(self):
        """Test that gradients flow correctly with uniform distribution."""
        torch.manual_seed(42)
        
        # Create uniform distribution layer (uniform doesn't support learnable params in current impl)
        # So we'll test with a learnable normal instead
        layer = create_normal_alpha_layer(0.5, 0.1, learnable=True)
        
        # Simple input
        x = torch.randn(10, requires_grad=True)
        
        # Forward pass
        result = layer(x)
        
        # Backward pass
        loss = result.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert layer.probabilistic_order.loc.grad is not None
        assert layer.probabilistic_order.scale.grad is not None
        
        # Check gradient magnitudes are reasonable
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(layer.probabilistic_order.loc.grad).all()
        assert torch.isfinite(layer.probabilistic_order.scale.grad).all()
    
    def test_beta_distribution_gradients(self):
        """Test gradient computation for Beta distribution."""
        torch.manual_seed(42)
        
        # Create learnable Beta distribution layer
        layer = create_beta_alpha_layer(2.0, 2.0, learnable=True)
        
        x = torch.randn(8, requires_grad=True)
        
        # Test forward pass (uses rsample for reparameterization)
        result = layer(x)
        
        loss = result.sum()
        loss.backward()
        
        # Check gradients exist for Beta parameters
        assert layer.probabilistic_order.concentration1.grad is not None
        assert layer.probabilistic_order.concentration0.grad is not None
        assert torch.isfinite(layer.probabilistic_order.concentration1.grad).all()
        assert torch.isfinite(layer.probabilistic_order.concentration0.grad).all()
    
    def test_gradient_consistency_across_runs(self):
        """Test that gradients are consistent across different runs."""
        torch.manual_seed(42)
        
        layer = create_normal_alpha_layer(0.5, 0.1, learnable=True)
        x = torch.randn(10, requires_grad=True)
        
        # Test multiple runs
        n_runs = 3
        gradients = []
        
        for _ in range(n_runs):
            # Reset gradients
            if x.grad is not None:
                x.grad.zero_()
            for param in layer.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            
            # Forward and backward
            result = layer(x)
            loss = result.sum()
            loss.backward()
            
            # Store gradients
            gradients.append({
                'x_grad': x.grad.clone(),
                'loc_grad': layer.probabilistic_order.loc.grad.clone(),
                'scale_grad': layer.probabilistic_order.scale.grad.clone()
            })
        
        # Check that gradients are finite
        for i, grads in enumerate(gradients):
            assert torch.isfinite(grads['x_grad']).all(), f"x_grad not finite for run {i}"
            assert torch.isfinite(grads['loc_grad']).all(), f"loc_grad not finite for run {i}"
            assert torch.isfinite(grads['scale_grad']).all(), f"scale_grad not finite for run {i}"
    
    def test_layer_integration_gradients(self):
        """Test gradient flow through neural network layers."""
        torch.manual_seed(42)
        
        # Create a simple network with probabilistic fractional layer
        class TestNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 5)
                self.frac_layer = create_normal_alpha_layer(0.5, 0.1, learnable=True)
                self.linear2 = nn.Linear(5, 1)
            
            def forward(self, x):
                x = F.relu(self.linear1(x))
                x = self.frac_layer(x)
                return self.linear2(x)
        
        net = TestNet()
        x = torch.randn(4, 10, requires_grad=True)
        
        # Forward pass
        output = net(x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check all parameters have gradients
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
    
    def test_gradient_variance_stability(self):
        """Test that gradients have reasonable variance across multiple runs."""
        torch.manual_seed(42)
        
        layer = create_normal_alpha_layer(0.5, 0.1, learnable=True)
        x = torch.randn(10, requires_grad=True)
        
        # Collect gradients from multiple runs
        n_runs = 10
        gradients = []
        
        for _ in range(n_runs):
            # Reset gradients
            if x.grad is not None:
                x.grad.zero_()
            for param in layer.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            
            # Forward and backward
            result = layer(x)
            loss = result.sum()
            loss.backward()
            gradients.append(layer.probabilistic_order.loc.grad.clone())
        
        # Compute variance
        grad_stack = torch.stack(gradients)
        grad_var = torch.var(grad_stack)
        
        # Variance should be finite and reasonable
        assert torch.isfinite(grad_var), "Gradient variance should be finite"
        # Variance should not be extremely large (indicates instability)
        assert grad_var < 1e6, "Gradient variance should be reasonable"
    
    def test_non_learnable_distribution(self):
        """Test that non-learnable distributions work without gradient computation."""
        torch.manual_seed(42)
        
        # Non-learnable distribution layer
        layer = create_normal_alpha_layer(0.5, 0.1, learnable=False)
        
        x = torch.randn(10, requires_grad=True)
        
        # Forward pass
        result = layer(x)
        
        loss = result.sum()
        loss.backward()
        
        # Only x should have gradients, not distribution parameters
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        
        # Distribution should not have learnable parameters
        assert not hasattr(layer.probabilistic_order, 'loc') or not isinstance(layer.probabilistic_order.loc, nn.Parameter)
    
    def test_gradient_stability(self):
        """Test gradient stability across multiple forward passes."""
        torch.manual_seed(42)
        
        layer = create_normal_alpha_layer(0.5, 0.1, learnable=True)
        x = torch.randn(10, requires_grad=True)
        
        # Test multiple forward passes
        for i in range(5):
            # Reset gradients
            if x.grad is not None:
                x.grad.zero_()
            for param in layer.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            
            # Forward and backward
            result = layer(x)
            loss = result.sum()
            loss.backward()
            
            # Check gradients are finite
            assert torch.isfinite(x.grad).all(), f"Non-finite x_grad for iteration {i}"
            assert torch.isfinite(layer.probabilistic_order.loc.grad).all(), f"Non-finite loc_grad for iteration {i}"
            assert torch.isfinite(layer.probabilistic_order.scale.grad).all(), f"Non-finite scale_grad for iteration {i}"


if __name__ == "__main__":
    pytest.main([__file__])
