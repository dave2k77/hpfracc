
import pytest
import torch
import numpy as np
from hpfracc.ml.hybrid_gnn_layers import HybridFractionalGraphConv, HybridFractionalGraphAttention, GraphConfig
from hpfracc.ml.gnn_layers.chebyshev_approx import chebyshev_spectral_fractional

class TestHybridGNN:
    """Test Hybrid GNN layers and Chebyshev approximations."""
    
    def test_chebyshev_approximation_identity(self):
        """Test that for alpha=0, Chebyshev approx returns identity."""
        x = torch.randn(10, 5)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # L^0 = I
        out = chebyshev_spectral_fractional(x, edge_index, alpha=0.0, k=5)
        
        assert torch.allclose(out, x, atol=1e-5), "L^0 should be I"

    def test_chebyshev_approximation_laplacian(self):
        """Test that for alpha=1, Chebyshev approx approximates L * x."""
        x = torch.randn(2, 5) # 2 nodes, 5 features
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long) # Simple 2-node graph
        
        # Compute exact Laplacian
        # D = [[1, 0], [0, 1]]
        # A = [[0, 1], [1, 0]]
        # L_sym = I - D^-0.5 A D^-0.5 = I - A
        # L = [[1, -1], [-1, 1]]
        L_exact = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        y_exact = L_exact @ x
        
        # Approx L^1
        # Need high K for exact reconstruction if L is not scaled perfectly
        # But L has eigenvalues in [0, 2], so it should work well
        y_approx = chebyshev_spectral_fractional(x, edge_index, alpha=1.0, k=5)
        
        # Should be reasonably close
        assert torch.allclose(y_approx, y_exact, atol=1e-3), "L^1 approx failed"

    def test_hybrid_conv_forward(self):
        """Test HybridFractionalGraphConv forward pass (Torch backend)."""
        layer = HybridFractionalGraphConv(
            in_channels=10, 
            out_channels=20, 
            fractional_order=0.5,
            config=GraphConfig(backend='pytorch', use_advanced_fractional=True)
        )
        
        x = torch.randn(10, 10)
        # Adjacency matrix for Hybrid layer
        adj = torch.tensor([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=torch.float32)
        # Pad to match x shape (10 nodes)
        adj_full = torch.zeros(10, 10)
        adj_full[:3, :3] = adj
        
        out = layer(x, adj_full)
        assert out.shape == (10, 20)

    def test_hybrid_attention_forward(self):
        """Test HybridFractionalGraphAttention forward pass."""
        layer = HybridFractionalGraphAttention(
            in_channels=10,
            out_channels=16,
            num_heads=4,
            config=GraphConfig(backend='pytorch')
        )
        
        x = torch.randn(10, 10)
        adj = torch.eye(10) # Self loops
        
        out = layer(x, adj)
        assert out.shape == (10, 16)
