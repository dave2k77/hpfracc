"""
Stochastic Memory Sampling for Fractional Derivatives

This module implements unbiased estimators for fractional derivatives using
stochastic sampling of the memory history instead of full computation.
"""

import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Tuple, Optional, Union, Callable, Dict, Any
import math


class StochasticMemorySampler:
    """
    Base class for stochastic memory sampling strategies.
    """

    def __init__(self, alpha: float, method: str = "importance", **kwargs):
        self.alpha = alpha
        self.method = method
        self.kwargs = kwargs

    def sample_indices(self, n: int, k: int) -> torch.Tensor:
        """Sample k indices from history of length n."""
        raise NotImplementedError

    def compute_weights(self, indices: torch.Tensor, n: int) -> torch.Tensor:
        """Compute importance weights for sampled indices."""
        raise NotImplementedError

    def estimate_derivative(self, x: torch.Tensor, indices: torch.Tensor,
                            weights: torch.Tensor) -> torch.Tensor:
        """Estimate fractional derivative using sampled indices and weights."""
        raise NotImplementedError


class ImportanceSampler(StochasticMemorySampler):
    """
    Importance sampling for fractional derivative memory.

    Uses power-law distribution p(j) ∝ (n-j)^(-(1+α-τ)) where τ controls
    the tempering of the heavy tail.
    """

    def __init__(self, alpha: float, tau: float = 0.1, **kwargs):
        super().__init__(alpha, "importance", **kwargs)
        self.tau = tau  # Tempering parameter

    def sample_indices(self, n: int, k: int) -> torch.Tensor:
        """Sample indices using importance sampling distribution."""
        if k >= n:
            return torch.arange(n, dtype=torch.long)

        # Power-law distribution: p(j) ∝ (n-j)^(-(1+α-τ))
        j_vals = torch.arange(n, dtype=torch.float32)
        log_probs = -(1 + self.alpha - self.tau) * torch.log(n - j_vals + 1e-8)

        # Normalize probabilities
        probs = torch.exp(log_probs - torch.logsumexp(log_probs, dim=0))

        # Sample without replacement
        indices = torch.multinomial(probs, k, replacement=False)
        return indices

    def compute_weights(self, indices: torch.Tensor, n: int) -> torch.Tensor:
        """Compute importance weights w(j)/p(j)."""
        # True weights: w(j) ∝ (n-j)^(-(1+α))
        j_vals = indices.float()
        true_weights = torch.pow(n - j_vals + 1e-8, -(1 + self.alpha))

        # Sampling probabilities: p(j) ∝ (n-j)^(-(1+α-τ))
        sampling_probs = torch.pow(
            n - j_vals + 1e-8, -(1 + self.alpha - self.tau))

        # Importance weights: w(j)/p(j)
        importance_weights = true_weights / (sampling_probs + 1e-8)

        return importance_weights

    def estimate_derivative(self, x: torch.Tensor, indices: torch.Tensor,
                            weights: torch.Tensor) -> torch.Tensor:
        """Estimate fractional derivative using importance sampling."""
        # Ensure x is 2D: (batch, seq) or (1, seq) for uniform handling
        is_1d = x.dim() == 1
        if is_1d:
            x_2d = x.unsqueeze(0)
        else:
            x_2d = x
            
        batch_size, n = x_2d.shape
        
        if len(indices) == 0:
            output = torch.zeros_like(x_2d)
            return output.squeeze(0) if is_1d else output

        # Current value (x_n)
        current_val = x_2d[:, -1] # Shape: (batch,)
        
        # Sampled values (x_{n-k})
        # indices are relative offsets from current position
        # history indices = n - 1 - indices
        history_indices = n - 1 - indices
        
        # Gather sampled values
        # We use advanced indexing to ensure gradients are preserved
        # x_2d: (batch, n)
        # history_indices: (sampled_k,)
        # We want sampled_vals: (batch, sampled_k)
        sampled_vals = x_2d[:, history_indices]
        
        # Differences: x_n - x_{n-k}
        # (batch, 1) - (batch, sampled_k) -> (batch, sampled_k)
        differences = current_val.unsqueeze(1) - sampled_vals
        
        # Apply weights: (sampled_k,) -> (1, sampled_k)
        weighted_diffs = differences * weights.unsqueeze(0)
        
        # Sum over samples and normalize
        weighted_sum = weighted_diffs.sum(dim=1)
        result_val = weighted_sum / len(indices)
        
        # Construct output tensor preserving gradient graph
        # We create a new tensor where the last element is the computed derivative
        # The other elements (zeros) are independent of x and won't affect grad
        
        # We need the output to carry gradients from result_val to x.
        # output is currently detached zeros.
        # We can't do in-place assignment to a leaf created by zeros_like if we want prop.
        # Actually we want output to *be* result_val at the last index.
        
        # Let's return a tensor where all but last are detached zeros, and last is result_val
        zeros_part = torch.zeros(batch_size, n - 1, device=x.device, dtype=x.dtype)
        # result_val is (batch,) -> (batch, 1)
        final_part = result_val.unsqueeze(1)
        
        output = torch.cat([zeros_part, final_part], dim=1)
        
        if is_1d:
            return output.squeeze(0)
        return output


class StratifiedSampler(StochasticMemorySampler):
    """
    Stratified sampling with recent window and tail sampling.

    Samples densely from recent history and sparsely from tail.
    """

    def __init__(self, alpha: float, recent_window: int = 32,
                 tail_ratio: float = 0.3, **kwargs):
        super().__init__(alpha, "stratified", **kwargs)
        self.recent_window = recent_window
        self.tail_ratio = tail_ratio

    def sample_indices(self, n: int, k: int) -> torch.Tensor:
        """Sample indices using stratified sampling."""
        if k >= n:
            return torch.arange(n, dtype=torch.long)

        # Determine split between recent and tail
        k_recent_target = int(k * (1 - self.tail_ratio))
        recent_available = min(self.recent_window, n)
        
        # We can't sample more than available in recent window
        k_recent = min(k_recent_target, recent_available)
        
        # Shift shortfall to tail
        shortfall = k_recent_target - k_recent
        k_tail = (k - k_recent_target) + shortfall
        
        indices = []

        # Sample from recent window (uniform)
        if k_recent > 0:
            # We want unique indices from 0 to recent_available-1
            # randperm ensures uniqueness
            recent_indices = torch.randperm(recent_available)[:k_recent]
            indices.append(recent_indices)

        # Sample from tail (power-law)
        if k_tail > 0 and n > self.recent_window:
            tail_start = self.recent_window
            tail_n = n - tail_start
            tail_j = torch.arange(tail_n, dtype=torch.float32)

            # Power-law distribution for tail
            log_probs = -(1 + self.alpha) * torch.log(tail_j + 1e-8)
            probs = torch.exp(log_probs - torch.logsumexp(log_probs, dim=0))

            tail_indices = torch.multinomial(probs, k_tail, replacement=False)
            tail_indices += tail_start  # Offset to actual indices
            indices.append(tail_indices)

        if indices:
            return torch.cat(indices)
        else:
            return torch.arange(min(k, n), dtype=torch.long)

    def compute_weights(self, indices: torch.Tensor, n: int) -> torch.Tensor:
        """Compute weights for stratified sampling."""
        weights = torch.ones_like(indices, dtype=torch.float32)

        # Recent window: uniform weights
        recent_mask = indices < self.recent_window
        if recent_mask.any():
            weights[recent_mask] = 1.0

        # Tail: power-law weights
        tail_mask = indices >= self.recent_window
        if tail_mask.any():
            tail_indices = indices[tail_mask]
            j_vals = tail_indices.float()
            # Simple weight based on decay
            tail_weights = torch.pow(n - j_vals + 1e-8, -(1 + self.alpha))
            weights[tail_mask] = tail_weights

        return weights

    def estimate_derivative(self, x: torch.Tensor, indices: torch.Tensor,
                            weights: torch.Tensor) -> torch.Tensor:
        """Estimate fractional derivative using stratified sampling."""
        # Reuse ImportanceSampler implementation
        return ImportanceSampler.estimate_derivative(self, x, indices, weights)


class ControlVariateSampler(StochasticMemorySampler):
    """
    Control variate sampling with deterministic baseline.

    Uses a cheap deterministic approximation (e.g., short memory) as baseline
    and samples only the residual tail.
    """

    def __init__(self, alpha: float, baseline_window: int = 16, **kwargs):
        super().__init__(alpha, "control_variate", **kwargs)
        self.baseline_window = baseline_window
        self.sampler = ImportanceSampler(alpha, **kwargs)

    def compute_baseline(self, x: torch.Tensor) -> torch.Tensor:
        """Compute deterministic baseline using recent window."""
        # Placeholder - actual baseline computed in estimate_derivative
        # to ensure graph connectivity
        return torch.zeros_like(x)

    def sample_indices(self, n: int, k: int) -> torch.Tensor:
        """Sample indices from tail only (excluding baseline window)."""
        if n <= self.baseline_window:
            return torch.arange(min(k, n), dtype=torch.long)

        # Sample only from tail
        tail_n = n - self.baseline_window
        tail_k = min(k, tail_n)

        if tail_k == 0:
            return torch.empty(0, dtype=torch.long)

        # Use importance sampling for tail
        tail_indices = self.sampler.sample_indices(tail_n, tail_k)
        # Offset to actual indices
        return tail_indices + self.baseline_window

    def compute_weights(self, indices: torch.Tensor, n: int) -> torch.Tensor:
        """Compute weights for control variate sampling."""
        if len(indices) == 0:
            return torch.empty(0, dtype=torch.float32)

        # Use importance sampling weights for tail
        tail_indices = indices - self.baseline_window
        tail_n = n - self.baseline_window
        return self.sampler.compute_weights(tail_indices, tail_n)

    def estimate_derivative(self, x: torch.Tensor, indices: torch.Tensor,
                            weights: torch.Tensor) -> torch.Tensor:
        """Estimate derivative using control variate method."""
        # Determine shape
        is_1d = x.dim() == 1
        if is_1d:
            x_2d = x.unsqueeze(0)
        else:
            x_2d = x
            
        batch_size, n = x_2d.shape
        
        # 1. Baseline
        # Sum of (current - history) for recent window
        window = min(n, self.baseline_window)
        if window > 1:
            current_val = x_2d[:, -1].unsqueeze(1)
            recent_vals = x_2d[:, -window:-1]
            # Simple average difference as baseline
            baseline_val = (current_val - recent_vals).mean(dim=1)
        else:
            baseline_val = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
            
        # 2. Tail Residual
        if len(indices) > 0:
             # history indices = n - 1 - indices
            history_indices = n - 1 - indices
            sampled_vals = x_2d[:, history_indices]
            current_val_sq = x_2d[:, -1].unsqueeze(1)
            
            tail_diffs = current_val_sq - sampled_vals
            # Weighted average of residuals
            tail_val = (tail_diffs * weights.unsqueeze(0)).sum(dim=1) / len(indices)
        else:
            tail_val = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
            
        result_val = baseline_val + tail_val
        
        # Pack into output tensor
        zeros_part = torch.zeros(batch_size, n - 1, device=x.device, dtype=x.dtype)
        final_part = result_val.unsqueeze(1)
        output = torch.cat([zeros_part, final_part], dim=1)
        
        if is_1d:
            return output.squeeze(0)
        return output


def stochastic_fractional_derivative(x: torch.Tensor, alpha: float, k: int = 64,
                                     method: str = "importance", **sampler_kwargs) -> torch.Tensor:
    """
    Public interface for stochastic fractional derivative.
    Fully differentiable via PyTorch Autograd.
    """
    # Create appropriate sampler
    if method == "importance":
        sampler = ImportanceSampler(alpha, **sampler_kwargs)
    elif method == "stratified":
        sampler = StratifiedSampler(alpha, **sampler_kwargs)
    elif method == "control_variate":
        sampler = ControlVariateSampler(alpha, **sampler_kwargs)
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    # Sample indices and compute weights
    n = x.dim() == 1 and len(x) or x.shape[-1]
    indices = sampler.sample_indices(n, k)
    
    # ensure indices are on correct device
    if indices.device != x.device:
        indices = indices.to(x.device)
        
    weights = sampler.compute_weights(indices, n)
    if weights.device != x.device:
        weights = weights.to(x.device)
        
    # Estimate derivative
    result = sampler.estimate_derivative(x, indices, weights)

    return result


class StochasticFractionalLayer(nn.Module):
    """
    PyTorch module for stochastic fractional derivatives.
    """

    def __init__(self, alpha: float, k: int = 64, method: str = "importance", **kwargs):
        super().__init__()
        self.alpha = alpha
        self.k = k
        self.method = method
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return stochastic_fractional_derivative(x, self.alpha, self.k, self.method, **self.kwargs)

    def extra_repr(self) -> str:
        return f'alpha={self.alpha}, k={self.k}, method={self.method}'


# Convenience functions
def create_stochastic_fractional_layer(alpha: float, k: int = 64,
                                       method: str = "importance", **kwargs) -> StochasticFractionalLayer:
    """Convenience function for creating stochastic fractional layer."""
    return StochasticFractionalLayer(alpha, k, method, **kwargs)
