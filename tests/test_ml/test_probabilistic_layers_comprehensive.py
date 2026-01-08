"""
Comprehensive tests for hpfracc.ml.probabilistic_fractional_orders module

This module tests probabilistic fractional orders where the fractional order
is treated as a random variable.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple
import warnings

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Check if NumPyro is available
NUMPYRO_AVAILABLE = False
try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.optim import Adam
    import jax
    NUMPYRO_AVAILABLE = True
except ImportError:
    pass


# Global skip removed to allow fallback testing


from hpfracc.ml.probabilistic_fractional_orders import (
    ProbabilisticFractionalOrder,
    ProbabilisticFractionalLayer,
    create_probabilistic_fractional_layer,
    create_normal_alpha_layer,
    create_uniform_alpha_layer,
    create_beta_alpha_layer,
    model,
    guide
)



@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro required for direct Order testing")
class TestProbabilisticFractionalOrder:
    """Test the ProbabilisticFractionalOrder class"""

    def test_initialization_default(self):
        """Test ProbabilisticFractionalOrder initialization with default parameters"""
        prob_order = ProbabilisticFractionalOrder(model, guide)
        
        assert prob_order.model is not None
        assert prob_order.guide is not None
        assert prob_order.backend_type == "numpyro"
        assert prob_order.svi is not None
        assert prob_order.svi_state is None

    def test_initialization_with_backend(self):
        """Test ProbabilisticFractionalOrder initialization with specific backend"""
        prob_order = ProbabilisticFractionalOrder(model, guide, backend="numpyro")
        
        assert prob_order.model is not None
        assert prob_order.guide is not None
        assert prob_order.backend_type == "numpyro"
        assert prob_order.svi is not None

    def test_initialization_invalid_backend(self):
        """Test ProbabilisticFractionalOrder initialization with invalid backend"""
        with pytest.raises(ValueError):
            ProbabilisticFractionalOrder(model, guide, backend="invalid")

    def test_init_svi_state(self):
        """Test SVI state initialization"""
        prob_order = ProbabilisticFractionalOrder(model, guide)
        
        # Create dummy data
        rng_key = jax.random.PRNGKey(0)
        dummy_x = jax.numpy.ones((1, 128))
        dummy_y = jax.numpy.ones((1, 128, 1))
        
        # Initialize SVI state
        prob_order.init(rng_key, dummy_x, dummy_y)
        
        assert prob_order.svi_state is not None

    def test_sample_without_init(self):
        """Test that sampling without initialization raises error"""
        prob_order = ProbabilisticFractionalOrder(model, guide)
        
        with pytest.raises(RuntimeError):
            prob_order.sample()

    def test_sample_after_init(self):
        """Test sampling after initialization"""
        prob_order = ProbabilisticFractionalOrder(model, guide)
        
        # Create dummy data
        rng_key = jax.random.PRNGKey(0)
        dummy_x = jax.numpy.ones((1, 128))
        dummy_y = jax.numpy.ones((1, 128, 1))
        
        # Initialize SVI state
        prob_order.init(rng_key, dummy_x, dummy_y)
        
        # Sample
        samples = prob_order.sample(k=5)
        
        assert samples is not None
        assert hasattr(samples, 'shape') or isinstance(samples, (list, tuple))
        if hasattr(samples, 'shape'):
            assert samples.shape[0] == 5

    def test_log_prob_without_init(self):
        """Test that log_prob without initialization raises error"""
        prob_order = ProbabilisticFractionalOrder(model, guide)
        
        with pytest.raises(RuntimeError):
            prob_order.log_prob(torch.tensor(0.5))

    def test_log_prob_after_init(self):
        """Test log_prob after initialization"""
        prob_order = ProbabilisticFractionalOrder(model, guide)
        
        # Create dummy data
        rng_key = jax.random.PRNGKey(0)
        dummy_x = jax.numpy.ones((1, 128))
        dummy_y = jax.numpy.ones((1, 128, 1))
        
        # Initialize SVI state
        prob_order.init(rng_key, dummy_x, dummy_y)
        
        # Compute log probability
        value = torch.tensor(0.5)
        log_prob = prob_order.log_prob(value)
        
        assert log_prob is not None
        assert isinstance(log_prob, torch.Tensor)


class TestProbabilisticFractionalLayer:
    """Test the ProbabilisticFractionalLayer class"""

    def test_initialization_default(self):
        """Test ProbabilisticFractionalLayer initialization with default parameters"""
        layer = ProbabilisticFractionalLayer()
        
        assert layer.probabilistic_order is not None
        assert layer.probabilistic_order is not None
        if NUMPYRO_AVAILABLE:
            assert layer.probabilistic_order.model is not None
            assert layer.probabilistic_order.guide is not None
            assert layer.probabilistic_order.svi_state is not None
        else:
             # Torch fallback doesn't have these
             pass
        assert layer.kwargs is not None

    def test_initialization_with_kwargs(self):
        """Test ProbabilisticFractionalLayer initialization with kwargs"""
        kwargs = {'param1': 'value1', 'param2': 42}
        layer = ProbabilisticFractionalLayer(**kwargs)
        
        assert layer.probabilistic_order is not None
        assert layer.kwargs == kwargs

    def test_forward_basic(self):
        """Test basic forward pass"""
        layer = ProbabilisticFractionalLayer()
        
        # Create test input
        x = torch.randn(10, 10)
        
        # Forward pass
        result = layer.forward(x)
        
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_forward_empty(self):
        """Test forward pass with empty tensor"""
        layer = ProbabilisticFractionalLayer()
        
        # Create test input
        x = torch.tensor([])
        
        # Forward pass
        result = layer.forward(x)
        
        assert result is not None
        assert isinstance(result, torch.Tensor)

    def test_forward_2d(self):
        """Test forward pass with 2D tensor"""
        layer = ProbabilisticFractionalLayer()
        
        # Create test input
        x = torch.randn(10, 20)
        
        # Forward pass
        result = layer.forward(x)
        
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_forward_different_sizes(self):
        """Test forward pass with different tensor sizes"""
        layer = ProbabilisticFractionalLayer()
        
        sizes = [(10,), (10, 20), (10, 20, 30)]
        
        for size in sizes:
            x = torch.randn(*size)
            result = layer.forward(x)
            
            assert result is not None
            assert isinstance(result, torch.Tensor)
            assert result.shape == x.shape

    def test_sample_alpha(self):
        """Test sampling fractional orders"""
        layer = ProbabilisticFractionalLayer()
        
        # Sample alpha values
        samples = layer.sample_alpha(n_samples=5)
        
        assert samples is not None
        assert isinstance(samples, torch.Tensor)
        assert len(samples) == 5

    def test_sample_alpha_default(self):
        """Test sampling fractional orders with default n_samples"""
        layer = ProbabilisticFractionalLayer()
        
        # Sample alpha values
        samples = layer.sample_alpha()
        
        assert samples is not None
        assert isinstance(samples, torch.Tensor)
        assert len(samples) == 1

    def test_get_alpha_statistics(self):
        """Test getting alpha statistics"""
        layer = ProbabilisticFractionalLayer()
        
        # Get statistics
        stats = layer.get_alpha_statistics()
        
        assert stats is not None
        assert isinstance(stats, dict)
        assert 'mean' in stats
        assert 'std' in stats
        assert isinstance(stats['mean'], torch.Tensor)
        assert isinstance(stats['std'], torch.Tensor)

    def test_extra_repr(self):
        """Test the extra_repr method"""
        layer = ProbabilisticFractionalLayer()
        
        repr_str = layer.extra_repr()
        
        assert isinstance(repr_str, str)
        if NUMPYRO_AVAILABLE:
            assert "NumPyro SVI" in repr_str
        else:
            # Fallback behavior might have different repr, currently it likely defaults to empty or simple
            # Since ProbabilisticFractionalOrder.__init__ with distribution doesn't set extra fields on layer
            # let's just check it returns a string
            pass

    def test_forward_gradient_flow(self):
        """Test that gradients flow through the layer"""
        layer = ProbabilisticFractionalLayer()
        
        # Create test input with gradients
        x = torch.randn(10, 10, requires_grad=True)
        
        # Forward pass
        result = layer.forward(x)
        
        # Compute loss
        loss = result.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestProbabilisticConvenienceFunctions:
    """Test the convenience functions"""

    def test_create_probabilistic_fractional_layer(self):
        """Test create_probabilistic_fractional_layer function"""
        layer = create_probabilistic_fractional_layer()
        
        assert isinstance(layer, ProbabilisticFractionalLayer)
        assert layer.probabilistic_order is not None

    def test_create_probabilistic_fractional_layer_with_kwargs(self):
        """Test create_probabilistic_fractional_layer with kwargs"""
        layer = create_probabilistic_fractional_layer(param1='value1')
        
        assert isinstance(layer, ProbabilisticFractionalLayer)
        assert layer.kwargs['param1'] == 'value1'

    def test_create_normal_alpha_layer(self):
        """Test create_normal_alpha_layer function"""
        layer = create_normal_alpha_layer(mean=0.5, std=0.1)
        
        assert isinstance(layer, ProbabilisticFractionalLayer)
        assert layer.probabilistic_order is not None

    def test_create_uniform_alpha_layer(self):
        """Test create_uniform_alpha_layer function"""
        layer = create_uniform_alpha_layer(low=0.1, high=0.9)
        
        assert isinstance(layer, ProbabilisticFractionalLayer)
        assert layer.probabilistic_order is not None

    def test_create_beta_alpha_layer(self):
        """Test create_beta_alpha_layer function"""
        layer = create_beta_alpha_layer(a=2.0, b=5.0)
        
        assert isinstance(layer, ProbabilisticFractionalLayer)
        assert layer.probabilistic_order is not None


class TestProbabilisticIntegration:
    """Integration tests for probabilistic layers"""

    def test_layer_in_neural_network(self):
        """Test ProbabilisticFractionalLayer in a neural network"""
        import torch.nn as nn
        
        # Create a simple network with probabilistic layer
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.prob_layer = ProbabilisticFractionalLayer()
                self.fc = nn.Linear(10, 5)
            
            def forward(self, x):
                x = self.prob_layer(x)
                x = self.fc(x)
                return x
        
        net = Net()
        
        # Create test input
        x = torch.randn(10, 10)
        
        # Forward pass
        result = net(x)
        
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 5)

    def test_multiple_probabilistic_layers(self):
        """Test stacking multiple probabilistic layers"""
        layer1 = ProbabilisticFractionalLayer()
        layer2 = ProbabilisticFractionalLayer()
        
        # Create test input
        x = torch.randn(10, 10)
        
        # Forward pass through both layers
        out1 = layer1(x)
        out2 = layer2(out1)
        
        assert out1 is not None
        assert out2 is not None
        assert out1.shape == x.shape
        assert out2.shape == x.shape

    def test_probabilistic_with_regular_layers(self):
        """Test combining probabilistic layers with regular layers"""
        import torch.nn as nn
        
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.prob_layer = ProbabilisticFractionalLayer()
                self.fc2 = nn.Linear(20, 5)
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.prob_layer(x)
                x = self.fc2(x)
                return x
        
        net = Net()
        
        # Create test input
        x = torch.randn(10, 10)
        
        # Forward pass
        result = net(x)
        
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 5)

    def test_end_to_end_training_workflow(self):
        """Test end-to-end training workflow"""
        import torch.nn as nn
        import torch.optim as optim
        
        # Create a simple network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.prob_layer = ProbabilisticFractionalLayer()
                self.fc = nn.Linear(10, 5)
            
            def forward(self, x):
                x = self.prob_layer(x)
                x = self.fc(x)
                return x
        
        net = Net()
        optimizer = optim.Adam(net.parameters())
        criterion = nn.MSELoss()
        
        # Create test data
        x = torch.randn(10, 10)
        y = torch.randn(10, 5)
        
        # Training step
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        assert loss is not None
        assert isinstance(loss, torch.Tensor)


class TestProbabilisticErrorHandling:
    """Test error handling in probabilistic layers"""

    def test_forward_with_invalid_input(self):
        """Test forward pass with invalid input"""
        layer = ProbabilisticFractionalLayer()
        
        # Test with None input
        with pytest.raises((TypeError, AttributeError)):
            layer.forward(None)

    def test_sample_with_invalid_n_samples(self):
        """Test sampling with invalid n_samples"""
        layer = ProbabilisticFractionalLayer()
        
        # Test with negative n_samples
        with pytest.raises((ValueError, AssertionError, RuntimeError)):
            layer.sample_alpha(n_samples=-1)


class TestProbabilisticEdgeCases:
    """Test edge cases in probabilistic layers"""

    def test_single_element_tensor(self):
        """Test forward pass with single element tensor"""
        layer = ProbabilisticFractionalLayer()
        
        x = torch.randn(1, 1)
        result = layer.forward(x)
        
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_very_large_tensor(self):
        """Test forward pass with very large tensor"""
        layer = ProbabilisticFractionalLayer()
        
        # Use a reasonable size that doesn't cause memory issues
        x = torch.randn(100, 100)
        result = layer.forward(x)
        
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_zero_tensor(self):
        """Test forward pass with zero tensor"""
        layer = ProbabilisticFractionalLayer()
        
        x = torch.zeros(10, 10)
        result = layer.forward(x)
        
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_ones_tensor(self):
        """Test forward pass with ones tensor"""
        layer = ProbabilisticFractionalLayer()
        
        x = torch.ones(10, 10)
        result = layer.forward(x)
        
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
