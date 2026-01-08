"""
Adjoint Method Optimization for Fractional Derivatives

This module implements memory-efficient "adjoint" methods for fractional derivatives
using PyTorch's Gradient Checkpointing. This allows training deeper fractional 
networks with rigorous fractional calculus (no approximations) while keeping
memory usage constant (O(1) relative to depth) for the intermediate activations.

Author: Davian R. Chin
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Callable, Union, Optional
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint

from hpfracc.core.definitions import FractionalOrder
from .fractional_autograd import fractional_derivative

@dataclass
class AdjointConfig:
    """Configuration for adjoint method optimization"""
    use_adjoint: bool = True
    memory_efficient: bool = True  # Enable gradient checkpointing
    checkpoint_frequency: int = 1  # Checkpoint every N layers (if applicable in a sequence)

def _fractional_derivative_wrapper(x: torch.Tensor, alpha: float, method: str) -> torch.Tensor:
    """
    Wrapper for fractional_derivative that ensures arguments differentially 
    interact with autograd only where appropriate.
    
    This is the function that will be re-run during the backward pass.
    """
    # alpha and method are not trainable parameters for the derivative operator itself
    # (though alpha *could* be trainable in other contexts, here we treat it as a scalar arg)
    return fractional_derivative(x, alpha, method)


class AdjointFractionalLayer(nn.Module):
    """
    Adjoint-optimized fractional layer using Gradient Checkpointing.

    This layer computes the rigorous fractional derivative but uses 
    gradient checkpointing to save memory during training. 
    Ideally suited for deep fractional networks.
    """

    def __init__(
            self,
            alpha: float,
            method: str = "RL",
            config: AdjointConfig = None):
        super().__init__()
        self.alpha = FractionalOrder(alpha)
        self.method = method
        self.config = config or AdjointConfig()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional adjoint (checkpointing) optimization"""
        
        # Determine effective alpha value
        alpha_val = float(self.alpha.alpha)

        if self.config.use_adjoint and x.requires_grad:
            # use_reentrant=False is generally recommended for newer PyTorch versions 
            # to handle backward hooks correctly, but True is default. 
            # We'll use False for stability with complex graphs if available, else default.
            return checkpoint(
                _fractional_derivative_wrapper, 
                x, 
                alpha_val, 
                self.method,
                use_reentrant=False
            )
        else:
            return fractional_derivative(x, alpha_val, self.method)

    def extra_repr(self) -> str:
        return f'alpha={self.alpha}, method={self.method}, adjoint={self.config.use_adjoint}'


class MemoryEfficientFractionalNetwork(nn.Module):
    """
    Memory-efficient fractional neural network using checkpointing.

    This network uses sequential checkpointing to minimize memory usage 
    during training while maintaining rigorous mathematical performance.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        fractional_order: float = 0.5,
        activation: str = "relu",
        dropout: float = 0.1,
        adjoint_config: AdjointConfig = None
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.fractional_order = FractionalOrder(fractional_order)
        self.activation = activation
        self.dropout = dropout
        self.adjoint_config = adjoint_config or AdjointConfig()

        self.layers = nn.ModuleList()
        
        # Build layer dimensions
        dims = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(dims) - 1):
            self.layers.append(self._make_block(dims[i], dims[i+1]))

    def _make_block(self, in_dim: int, out_dim: int) -> nn.Module:
        """Create a block consisting of Linear -> Fractional -> Activation"""
        block = nn.ModuleList()
        block.append(nn.Linear(in_dim, out_dim))
        
        # Add fractional layer
        block.append(AdjointFractionalLayer(
            self.fractional_order.alpha, 
            method="RL", 
            config=self.adjoint_config
        ))
        
        # Activation
        if self.activation == "relu":
            block.append(nn.ReLU())
        elif self.activation == "tanh":
            block.append(nn.Tanh())
        elif self.activation == "sigmoid":
            block.append(nn.Sigmoid())
            
        if self.dropout > 0:
            block.append(nn.Dropout(self.dropout))
            
        # Wrap in Sequential for easier handling, but we will iterate in forward
        return nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with memory-efficient sequential processing.
        """
        for i, block in enumerate(self.layers):
            if self.adjoint_config.use_adjoint and x.requires_grad:
                # Checkpoint the entire block
                # x must be passed as a positional arg
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


# -----------------------------------------------------------------------------
# Aliases for backward compatibility
# -----------------------------------------------------------------------------

def adjoint_fractional_derivative(x: torch.Tensor, alpha: float, method: str = "RL") -> torch.Tensor:
    """Convenience function for checkpointed fractional derivatives"""
    alpha_val = float(alpha)
    if x.requires_grad:
        return checkpoint(_fractional_derivative_wrapper, x, alpha_val, method, use_reentrant=False)
    return fractional_derivative(x, alpha_val, method)

adjoint_rl_derivative = adjoint_fractional_derivative

def adjoint_caputo_derivative(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return adjoint_fractional_derivative(x, alpha, "Caputo")

def adjoint_gl_derivative(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return adjoint_fractional_derivative(x, alpha, "GL")

