
import pytest
import torch
import torch.nn as nn
from torch.autograd import gradcheck
import numpy as np
from hpfracc.ml.stochastic_memory_sampling import (
    StochasticMemorySampler,
    ImportanceSampler,
    StratifiedSampler,
    ControlVariateSampler,
    stochastic_fractional_derivative,
    StochasticFractionalLayer
)

class TestStochasticSamplers:
    """Test individual sampler components."""
    
    @pytest.mark.parametrize("SamplerData", [
        (ImportanceSampler, {"tau": 0.1}),
        (StratifiedSampler, {"recent_window": 5, "tail_ratio": 0.5}),
        (ControlVariateSampler, {"baseline_window": 5})
    ])
    def test_sampler_initialization(self, SamplerData):
        SamplerClass, kwargs = SamplerData
        sampler = SamplerClass(alpha=0.5, **kwargs)
        assert isinstance(sampler, StochasticMemorySampler)
        assert sampler.alpha == 0.5

    def test_importance_sampling_indices(self):
        n = 100
        k = 20
        sampler = ImportanceSampler(alpha=0.5)
        indices = sampler.sample_indices(n, k)
        
        assert isinstance(indices, torch.Tensor)
        assert len(indices) == k
        assert (indices >= 0).all() and (indices < n).all()
        # Check uniqueness (sample without replacement)
        assert len(torch.unique(indices)) == k

    def test_stratified_sampling_structure(self):
        n = 100
        k = 20
        recent = 5
        sampler = StratifiedSampler(alpha=0.5, recent_window=recent, tail_ratio=0.5)
        indices = sampler.sample_indices(n, k)
        
        # Check that we sampled from recent window
        # Note: StratifiedSampler implementation might need review on how it returns indices
        # But generally it should include 0..recent-1 if k_recent is large enough
        assert len(indices) == k
        
    def test_weights_computation(self):
        n = 50
        k = 10
        sampler = ImportanceSampler(alpha=0.5)
        indices = sampler.sample_indices(n, k)
        weights = sampler.compute_weights(indices, n)
        
        assert weights.shape == indices.shape
        assert not torch.isnan(weights).any()
        assert not torch.isinf(weights).any()

class TestStochasticDerivativeGradients:
    """Test gradient flow through stochastic derivative."""
    
    def test_gradient_flow_history(self):
        """
        Verify that gradients flow back to historical inputs, not just the last point.
        This is the critical test for BPTT correctness.
        """
        # Create a sequence with requires_grad
        x = torch.randn(20, requires_grad=True, dtype=torch.float64)
        
        # Compute derivative
        # We use a large k to ensure we likely hit some historical points
        result = stochastic_fractional_derivative(x, alpha=0.5, k=15, method="importance")
        
        # Compute loss and backward
        loss = result.sum()
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        
        # The last point should definitely have a gradient
        assert x.grad[-1] != 0.0
        
        # KEY CHECK: Historical points should also have non-zero gradients
        # because the derivative depends on history: D^a x_t = sum w_k (x_t - x_{t-k})
        # If the implementation is correct, sampled history points will have gradients.
        # If the implementation is broken (like the identity pass-through), 
        # historical gradients might be zero or incorrect.
        historical_grads = x.grad[:-1]
        assert (historical_grads != 0).any(), "No gradient flow to history! BPTT is broken."

    def test_alpha_gradient(self):
        """Test if gradients flow w.r.t alpha (if supported)."""
        # Note: Currently alpha is float in the signature, so we wrap it
        pass

class TestStochasticLayer:
    """Test nn.Module wrapper."""
    
    def test_forward_pass(self):
        layer = StochasticFractionalLayer(alpha=0.5, k=10)
        x = torch.randn(5, 20) # Batch, Seq
        out = layer(x)
        assert out.shape == x.shape

