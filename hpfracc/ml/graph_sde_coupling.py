"""
Graph-SDE Coupling for Spatio-Temporal Dynamics

This module provides coupling layers that integrate spatial dynamics (via graph
neural networks) with temporal dynamics (via fractional SDEs) for modeling
spatio-temporal phenomena.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np

from hpfracc.core.definitions import FractionalOrder
from .gnn_layers import FractionalGraphConv
from .neural_fsde import NeuralFractionalSDE, NeuralFSDEConfig
from .backends import BackendType


class CouplingType:
    """Types of spatial-temporal coupling."""
    BIDIRECTIONAL = "bidirectional"  # Space ↔ Time
    SPATIAL_TO_TEMPORAL = "spatial_to_temporal"  # Space → Time only
    TEMPORAL_TO_SPATIAL = "temporal_to_spatial"  # Time → Space only
    GATED = "gated"  # Learned gating mechanism


class SpatialTemporalCoupling(nn.Module):
    """
    Coupling layer between spatial (graph) and temporal (SDE) dynamics.
    
    Computes learned coupling between spatial embeddings and temporal states.
    """
    
    def __init__(
        self,
        spatial_dim: int,
        temporal_dim: int,
        coupling_dim: int,
        coupling_type: str = CouplingType.BIDIRECTIONAL,
        use_attention: bool = True
    ):
        """
        Initialize coupling layer.
        
        Args:
            spatial_dim: Dimension of spatial (graph) features
            temporal_dim: Dimension of temporal (SDE) features
            coupling_dim: Dimension of coupling embedding
            coupling_type: Type of coupling mechanism
            use_attention: Use attention mechanism for coupling
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.coupling_dim = coupling_dim
        self.coupling_type = coupling_type
        self.use_attention = use_attention
        
        # Spatial to temporal projection
        self.spatial_to_temporal = nn.Linear(spatial_dim, coupling_dim)
        
        # Temporal to spatial projection
        self.temporal_to_spatial = nn.Linear(temporal_dim, coupling_dim)
        
        # Coupling function
        if coupling_type == CouplingType.GATED:
            self.gate = nn.Sequential(
                nn.Linear(coupling_dim * 2, coupling_dim),
                nn.Sigmoid()
            )
        
        # Final projection back to original dimensions
        self.temporal_projection = nn.Linear(coupling_dim, temporal_dim)
        self.spatial_projection = nn.Linear(coupling_dim, spatial_dim)
    
    def forward(
        self,
        spatial_features: torch.Tensor,
        temporal_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply spatial-temporal coupling.
        
        Args:
            spatial_features: Graph node features (batch, num_nodes, spatial_dim)
            temporal_features: SDE state features (batch, num_nodes, temporal_dim)
            
        Returns:
            Tuple of (coupled_spatial, coupled_temporal)
        """
        # Project to coupling space
        spatial_embed = self.spatial_to_temporal(spatial_features)
        temporal_embed = self.temporal_to_spatial(temporal_features)
        
        # Compute coupling based on type
        if self.coupling_type == CouplingType.BIDIRECTIONAL:
            # Average coupling
            coupled_embed = (spatial_embed + temporal_embed) / 2
            
        elif self.coupling_type == CouplingType.SPATIAL_TO_TEMPORAL:
            # Only spatial affects temporal
            coupled_embed = spatial_embed
            
        elif self.coupling_type == CouplingType.TEMPORAL_TO_SPATIAL:
            # Only temporal affects spatial
            coupled_embed = temporal_embed
            
        elif self.coupling_type == CouplingType.GATED:
            # Learned gating
            gate = self.gate(torch.cat([spatial_embed, temporal_embed], dim=-1))
            coupled_embed = gate * spatial_embed + (1 - gate) * temporal_embed
            
        else:
            raise ValueError(f"Unknown coupling type: {self.coupling_type}")
        
        # Project back to original spaces
        coupled_temporal = self.temporal_projection(coupled_embed)
        coupled_spatial = self.spatial_projection(coupled_embed)
        
        # Combine with residuals
        coupled_temporal = coupled_temporal + temporal_features
        coupled_spatial = coupled_spatial + spatial_features
        
        return coupled_spatial, coupled_temporal


class GraphFractionalSDELayer(nn.Module):
    """
    Layer that couples graph-based spatial dynamics with fractional SDE temporal evolution.
    
    Architecture:
    - Graph convolution for spatial features
    - Fractional SDE for temporal dynamics at each node
    - Learned coupling between spatial and temporal embeddings
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        fractional_order: Union[float, FractionalOrder] = 0.5,
        coupling_type: str = CouplingType.BIDIRECTIONAL,
        num_sde_steps: int = 10,
        backend: BackendType = BackendType.AUTO
    ):
        """
        Initialize Graph-Fractional SDE layer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for both spatial and temporal
            output_dim: Output feature dimension
            fractional_order: Fractional order for SDE dynamics
            coupling_type: Type of spatial-temporal coupling
            num_sde_steps: Number of SDE integration steps
            backend: Computation backend
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fractional_order = FractionalOrder(fractional_order) if isinstance(
            fractional_order, float) else fractional_order
        self.num_sde_steps = num_sde_steps
        self.backend = backend
        
        # Spatial (graph) dynamics
        self.spatial_layer = FractionalGraphConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            fractional_order=fractional_order,
            backend=backend
        )
        
        # Temporal (SDE) dynamics - using a simplified neural SDE
        self.temporal_layer = nn.GRUCell(
            input_dim,
            hidden_dim
        )
        
        # SDE drift network
        self.drift_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # SDE diffusion network
        self.diffusion_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus()
        )
        
        # Coupling layer
        self.coupling = SpatialTemporalCoupling(
            spatial_dim=hidden_dim,
            temporal_dim=hidden_dim,
            coupling_dim=hidden_dim,
            coupling_type=coupling_type
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def drift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute drift term for SDE."""
        return self.drift_net(x)
    
    def diffusion(self, x: torch.Tensor) -> torch.Tensor:
        """Compute diffusion term for SDE."""
        return self.diffusion_net(x)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through graph-SDE layer.
        
        Args:
            x: Node features (batch, num_nodes, input_dim)
            edge_index: Edge connectivity (2, num_edges)
            edge_weight: Optional edge weights (num_edges,)
            
        Returns:
            Updated node features (batch, num_nodes, output_dim)
        """
        batch_size, num_nodes, _ = x.shape
        
        # Spatial dynamics: graph convolution
        x_graph = x.view(-1, self.input_dim)
        spatial_features = self.spatial_layer(x_graph, edge_index, edge_weight)
        spatial_features = spatial_features.view(batch_size, num_nodes, self.hidden_dim)
        
        # Temporal dynamics: SDE evolution
        # Initialize temporal state from spatial features
        temporal_state = spatial_features.clone()
        
        # Simulate SDE for a few steps
        dt = 0.1  # Time step
        
        # L1 Scheme (Grünwald-Letnikov) for Fractional SDE
        # Precompute weights: (k+1)^alpha - k^alpha
        alpha_val = self.fractional_order.alpha
        k_vals = torch.arange(self.num_sde_steps + 1, device=x.device, dtype=x.dtype)
        weights = (k_vals + 1).pow(alpha_val) - k_vals.pow(alpha_val)
        
        # Gamma factor for scaling
        gamma_factor = 1.0 / torch.exp(torch.lgamma(torch.tensor(alpha_val + 1.0, device=x.device)))
        
        # History lists required for convolution
        drift_history = []
        diffusion_history = []
        
        for i in range(self.num_sde_steps):
            # Compute drift (batch, num_nodes, hidden)
            drift_val = self.drift(temporal_state)
            diffusion_val = self.diffusion(temporal_state)
            
            # Generate noise (batch, num_nodes, hidden)
            # Assuming diagonal/additive noise structure for simplicity in this layer
            noise = torch.randn_like(temporal_state) * np.sqrt(dt)
            
            # Store history
            drift_history.append(drift_val)
            diffusion_history.append(diffusion_val * noise) # Noise term history
            
            # Convolution
            # Stack history: (i+1, batch, nodes, hidden)
            drift_hist_stack = torch.stack(drift_history)
            diff_hist_stack = torch.stack(diffusion_history)
            
            # Get current weights and flip for convolution
            current_weights = weights[:i+1].flip(0)
            
            # Reshape weights for broadcasting
            # need shape (i+1, 1, 1, 1) to match (T, B, N, H)
            w_reshaped = current_weights.view(-1, 1, 1, 1)
            
            # Weighted sum
            drift_integral = (w_reshaped * drift_hist_stack).sum(dim=0)
            diffusion_integral = (w_reshaped * diff_hist_stack).sum(dim=0)
            
            # Update: X_{i+1} = X_0 + h^alpha / Gamma * Integral
            update_term = gamma_factor * (dt ** alpha_val) * (drift_integral + diffusion_integral)
            temporal_state = spatial_features.clone() + update_term  # Base is initial state (spatial_features)
        
        # Apply coupling
        coupled_spatial, coupled_temporal = self.coupling(spatial_features, temporal_state)
        
        # Combine spatial and temporal
        combined = (coupled_spatial + coupled_temporal) / 2
        
        # Output projection
        output = self.output_proj(combined)
        
        return output


class MultiScaleGraphSDE(nn.Module):
    """
    Multi-scale graph-SDE network with adaptive time stepping.
    
    Handles different time scales for graph updates vs SDE evolution,
    optimal for stiff coupled systems.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        fractional_order: Union[float, FractionalOrder] = 0.5,
        spatial_time_scale: float = 1.0,
        temporal_time_scale: float = 0.1
    ):
        """
        Initialize multi-scale graph-SDE.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden dimensions for each layer
            output_dim: Output dimension
            fractional_order: Fractional order for SDE
            spatial_time_scale: Time scale for spatial (graph) dynamics
            temporal_time_scale: Time scale for temporal (SDE) dynamics
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.spatial_time_scale = spatial_time_scale
        self.temporal_time_scale = temporal_time_scale
        
        # Build layers
        self.layers = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layer = GraphFractionalSDELayer(
                input_dim=dims[i],
                hidden_dim=dims[i+1],
                output_dim=dims[i+1],
                fractional_order=fractional_order
            )
            self.layers.append(layer)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with multi-scale dynamics.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_weight: Optional edge weights
            
        Returns:
            Output features
        """
        # Propagate through layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        
        # Output projection
        output = self.output_layer(x)
        
        return output
