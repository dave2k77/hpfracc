"""
Neural Fractional Stochastic Differential Equations

This module provides neural network-based fractional SDEs with adjoint
training methods for efficient gradient-based learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np

from ..core.definitions import FractionalOrder, validate_fractional_order
from ..ml.neural_ode import BaseNeuralODE, NeuralODEConfig
from ..ml.adjoint_optimization import AdjointConfig, adjoint_sde_gradient
from ..solvers.sde_solvers import solve_fractional_sde, FractionalSDESolver


@dataclass
class NeuralFSDEConfig(NeuralODEConfig):
    """Configuration for neural fractional SDE models."""
    diffusion_dim: int = 1  # Dimension of noise
    noise_type: str = "additive"  # "additive" or "multiplicative"
    drift_net: Optional[nn.Module] = None
    diffusion_net: Optional[nn.Module] = None
    use_sde_adjoint: bool = True  # Use SDE-specific adjoint method
    learn_alpha: bool = False  # Whether to learn fractional order


class NeuralFractionalSDE(BaseNeuralODE):
    """
    Neural network-based fractional SDE with adjoint training.
    
    Extends neural ODE framework to fractional stochastic differential equations
    for modeling stochastic dynamics with memory effects.
    
    The model learns:
    - Drift function f(t, x): deterministic dynamics
    - Diffusion function g(t, x): stochastic noise magnitude
    - Fractional order: memory effects in dynamics
    """
    
    def __init__(self, config: NeuralFSDEConfig):
        """
        Initialize neural fractional SDE.
        
        Args:
            config: Configuration for the neural fSDE
        """
        super().__init__(config)
        self.config = config
        self.diffusion_dim = config.diffusion_dim
        self.noise_type = config.noise_type
        
        # Build drift and diffusion networks
        self._build_drift_network()
        self._build_diffusion_network()
        
        # Fractional order parameter
        if isinstance(config.fractional_order, float):
            self.fractional_order_value = config.fractional_order
        else:
            self.fractional_order_value = config.fractional_order.alpha
        
        # Initialize learned fractional order if needed
        self.learn_alpha = getattr(config, 'learn_alpha', False)
        if self.learn_alpha:
            self.alpha_param = nn.Parameter(torch.tensor(self.fractional_order_value))
    
    def _build_drift_network(self):
        """Build neural network for drift function."""
        if self.config.drift_net is not None:
            self.drift_net = self.config.drift_net
        else:
            # Default drift network
            self.drift_net = nn.Sequential(
                nn.Linear(self.input_dim + 1, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
    
    def _build_diffusion_network(self):
        """Build neural network for diffusion function."""
        if self.config.diffusion_net is not None:
            self.diffusion_net = self.config.diffusion_net
        else:
            # Default diffusion network
            # Output shape: (batch, output_dim, diffusion_dim) for multiplicative
            # or (batch, diffusion_dim) for additive
            if self.noise_type == "multiplicative":
                output_dim = self.output_dim * self.diffusion_dim
            else:
                output_dim = self.diffusion_dim
            
            self.diffusion_net = nn.Sequential(
                nn.Linear(self.input_dim + 1, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, output_dim)
            )
            
        # Use softplus to ensure positive diffusion
        self.diffusion_activation = nn.Softplus()
    
    def drift_function(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Alias for drift() for compatibility with tests."""
        return self.drift(t, x)
    
    def diffusion_function(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Alias for diffusion() for compatibility with tests."""
        return self.diffusion(t, x)
    
    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute drift function f(t, x).
        
        Args:
            t: Time tensor
            x: State tensor
            
        Returns:
            Drift vector
        """
        # Ensure x has batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        
        # Ensure t has proper shape (batch, 1)
        if t.dim() == 0:  # Scalar
            t_expanded = t.unsqueeze(0).unsqueeze(0).expand(batch_size, 1)
        elif t.dim() == 1:  # 1D tensor
            if t.shape[0] == 1:  # Single time value
                t_expanded = t.unsqueeze(0).expand(batch_size, 1)
            else:  # Multiple time values (should match batch)
                t_expanded = t.unsqueeze(-1)
        else:  # Already 2D
            t_expanded = t
        
        # Concatenate [t, x]
        input_tensor = torch.cat([t_expanded, x], dim=-1)
        
        # Forward pass through drift network
        drift = self.drift_net(input_tensor)
        
        return drift
    
    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion function g(t, x).
        
        Args:
            t: Time tensor
            x: State tensor
            
        Returns:
            Diffusion matrix
        """
        # Ensure x has batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        
        # Ensure t has proper shape (batch, 1)
        if t.dim() == 0:  # Scalar
            t_expanded = t.unsqueeze(0).unsqueeze(0).expand(batch_size, 1)
        elif t.dim() == 1:  # 1D tensor
            if t.shape[0] == 1:  # Single time value
                t_expanded = t.unsqueeze(0).expand(batch_size, 1)
            else:  # Multiple time values (should match batch)
                t_expanded = t.unsqueeze(-1)
        else:  # Already 2D
            t_expanded = t
        
        # Concatenate [t, x]
        input_tensor = torch.cat([t_expanded, x], dim=-1)
        
        # Forward pass through diffusion network
        diffusion = self.diffusion_net(input_tensor)
        diffusion = self.diffusion_activation(diffusion)
        
        # Reshape for multiplicative noise
        if self.noise_type == "multiplicative":
            diffusion = diffusion.view(batch_size, self.output_dim, self.diffusion_dim)
        
        return diffusion
    
    def fractional_order(self) -> float:
        """Get current fractional order."""
        if self.learn_alpha:
            # Clamp to valid range (0, 2)
            return torch.clamp(self.alpha_param, 0.1, 1.9).item()
        return self.fractional_order_value
    
    def forward(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        method: str = "euler_maruyama",
        num_steps: int = 100,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass through neural fractional SDE.
        
        Args:
            x0: Initial condition
            t: Time points (1D tensor or 2D batch)
            method: Solver method
            num_steps: Number of integration steps
            seed: Random seed for reproducibility
            
        Returns:
            Trajectory solution
        """
        # Ensure inputs are tensors
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, dtype=torch.float32)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
            
        # Handle batch dimensions
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)  # (1, dim)
        
        batch_size, dim = x0.shape
        
        # Get time span
        if t.dim() > 1:
            t_flat = t.flatten()
        else:
            t_flat = t
            
        t_start = t_flat[0]
        t_end = t_flat[-1]
        
        # Use the PyTorch solver directly
        return self._solve_fractional_sde_torch(
            x0, t_start, t_end, num_steps, seed
        )

    def _solve_fractional_sde_torch(
        self,
        x0: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        num_steps: int,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        PyTorch-native fractional SDE solver (Euler-Maruyama).
        """
        device = x0.device
        dtype = x0.dtype
        batch_size, dim = x0.shape
        
        dt = (t_end - t_start) / num_steps
        alpha = self.fractional_order()
        
        # Gamma factor: 1 / Gamma(alpha + 1)
        # Use torch.lgamma for stability: exp(lgamma(x)) = Gamma(x)
        if isinstance(alpha, torch.Tensor):
            gamma_val = torch.exp(torch.lgamma(alpha + 1))
        else:
            gamma_val = torch.exp(torch.lgamma(torch.tensor(alpha + 1.0, device=device, dtype=dtype)))
            
        gamma_factor = 1.0 / gamma_val
        
        # Precompute weights
        # weights[k] = (k+1)^alpha - k^alpha
        # We need to handle alpha as tensor or float
        k_vals = torch.arange(num_steps + 1, device=device, dtype=dtype)
        
        if isinstance(alpha, torch.Tensor):
            weights = (k_vals + 1).pow(alpha) - k_vals.pow(alpha)
        else:
            weights = (k_vals + 1).pow(alpha) - k_vals.pow(alpha)
            
        # Initialize history
        # We need to store history for convolution
        # Shape: (num_steps, batch_size, dim)
        drift_history = []
        diffusion_history = []
        
        # Initialize trajectory
        # Shape: (num_steps + 1, batch_size, dim)
        trajectory = [x0]
        
        curr_x = x0
        curr_t = t_start
        
        if seed is not None:
            torch.manual_seed(seed)
            
        for i in range(num_steps):
            # Compute drift and diffusion
            # drift shape: (batch_size, dim)
            drift_val = self.drift(curr_t, curr_x)
            diffusion_val = self.diffusion(curr_t, curr_x)
            
            # Generate noise
            # Shape: (batch_size, diffusion_dim)
            dW = torch.randn(batch_size, self.diffusion_dim, device=device, dtype=dtype) * torch.sqrt(dt)
            
            # Store history
            drift_history.append(drift_val)
            
            # Handle noise term
            if self.noise_type == "multiplicative":
                # diffusion_val is (batch, dim, diffusion_dim)
                # dW is (batch, diffusion_dim)
                # Result should be (batch, dim)
                noise_term = torch.bmm(diffusion_val, dW.unsqueeze(-1)).squeeze(-1)
            else:
                # Additive or diagonal
                # diffusion_val is (batch, diffusion_dim)
                # dW is (batch, diffusion_dim)
                if dim == self.diffusion_dim:
                    # Diagonal noise
                    noise_term = diffusion_val * dW
                elif self.diffusion_dim == 1:
                    # Scalar noise broadcasted
                    noise_term = diffusion_val * dW
                    # If result is (batch, 1), it will broadcast during addition
                else:
                    # Mismatched dimensions for additive noise without matrix
                    # This might be an issue if not handled, but for now assume broadcasting or error
                    # If diffusion_val is (batch, m) and dW is (batch, m), we get (batch, m).
                    # If m != d, we can't add to drift (batch, d) unless m=1.
                    if diffusion_val.shape[-1] != dim:
                         # Try to treat as diagonal if possible or raise error?
                         # For now, let's assume if it's not multiplicative, it's element-wise compatible
                         noise_term = diffusion_val * dW
                    else:
                         noise_term = diffusion_val * dW
                
            diffusion_history.append(noise_term)
            
            # Compute memory terms via convolution
            # We need to sum weights[i-j] * history[j] for j=0..i
            # Efficient way using torch operations
            
            # Stack history so far
            # drift_hist_stack: (i+1, batch, dim)
            drift_hist_stack = torch.stack(drift_history)
            diff_hist_stack = torch.stack(diffusion_history)
            
            # Get weights for this step: w_i, w_{i-1}, ..., w_0
            # weights is 1D: (num_steps+1,)
            # We need weights[0]...weights[i]
            # And we need to reverse them to match history:
            # sum_{j=0}^i w_{i-j} h_j
            # w_{i} * h_0 + w_{i-1} * h_1 + ... + w_0 * h_i
            
            current_weights = weights[:i+1].flip(0) # (i+1,)
            
            # Reshape weights for broadcasting: (i+1, 1, 1)
            w_reshaped = current_weights.view(-1, 1, 1)
            
            # Weighted sum
            drift_integral = (w_reshaped * drift_hist_stack).sum(dim=0)
            diffusion_integral = (w_reshaped * diff_hist_stack).sum(dim=0)
            
            # Update step
            # X_{i+1} = X_0 + h^alpha / Gamma(alpha+1) * (drift_int + diff_int)
            
            next_x = x0 + gamma_factor * dt.pow(alpha) * (drift_integral + diffusion_integral)
            
            trajectory.append(next_x)
            curr_x = next_x
            curr_t = curr_t + dt
            
        # Stack trajectory
        # (num_steps + 1, batch_size, dim)
        trajectory_tensor = torch.stack(trajectory)
        
        # Return as (time_steps, batch_size, dim)
        return trajectory_tensor
    
    def get_fractional_order(self) -> Union[float, torch.Tensor]:
        """Get the fractional order parameter."""
        if self.learn_alpha:
            return torch.clamp(self.alpha_param, 0.1, 1.9)
        return self.fractional_order_value
    
    def adjoint_forward(self, x0: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Adjoint-compatible forward pass."""
        return self.forward(x0, t, **kwargs)


def create_neural_fsde(
    input_dim: int = None,
    output_dim: int = None,
    hidden_dim: int = 64,
    num_layers: int = 3,
    fractional_order: float = 0.5,
    diffusion_dim: int = 1,
    noise_type: str = "additive",
    learn_alpha: bool = False,
    use_adjoint: bool = True,
    drift_net: Optional[nn.Module] = None,
    diffusion_net: Optional[nn.Module] = None,
    config: Optional[NeuralFSDEConfig] = None
) -> NeuralFractionalSDE:
    """
    Factory function to create a neural fractional SDE.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        fractional_order: Initial fractional order
        diffusion_dim: Dimension of noise
        noise_type: Type of noise ("additive" or "multiplicative")
        learn_alpha: Whether to learn fractional order
        use_adjoint: Use adjoint method for backpropagation
        drift_net: Custom drift network
        diffusion_net: Custom diffusion network
        
    Returns:
        NeuralFractionalSDE instance
    """
    if config is not None:
        # Use provided config
        pass
    else:
        # Create config from parameters
        if input_dim is None or output_dim is None:
            raise ValueError("input_dim and output_dim must be provided when config is None")
        config = NeuralFSDEConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            fractional_order=fractional_order,
            use_adjoint=use_adjoint,
            diffusion_dim=diffusion_dim,
            noise_type=noise_type,
            learn_alpha=learn_alpha,
            drift_net=drift_net,
            diffusion_net=diffusion_net
        )
    
    model = NeuralFractionalSDE(config)
    model.learn_alpha = learn_alpha
    
    return model
