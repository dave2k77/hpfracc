
import pytest
import torch
import numpy as np
from hpfracc.ml.graph_sde_coupling import GraphFractionalSDELayer, SpatialTemporalCoupling

class TestGraphSDE:
    """Test Graph SDE Coupling and Layers."""

    def test_coupling_layer_bidirectional(self):
        """Test SpatialTemporalCoupling bidirectional mode."""
        coupling = SpatialTemporalCoupling(
            spatial_dim=10,
            temporal_dim=10,
            coupling_dim=5,
            coupling_type="bidirectional"
        )
        
        s_feat = torch.randn(2, 5, 10) # B, N, D
        t_feat = torch.randn(2, 5, 10)
        
        s_out, t_out = coupling(s_feat, t_feat)
        
        assert s_out.shape == s_feat.shape
        assert t_out.shape == t_feat.shape
        # Ensure values changed (coupling happened)
        assert not torch.allclose(s_out, s_feat)

    def test_graph_sde_layer_forward(self):
        """Test GraphFractionalSDELayer forward pass with L1 scheme."""
        layer = GraphFractionalSDELayer(
            input_dim=10,
            hidden_dim=20,
            output_dim=5,
            num_sde_steps=5,
            coupling_type="bidirectional"
        )
        
        x = torch.randn(2, 10, 10) # B, N, D
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # This will run the newly implemented L1 scheme
        out = layer(x, edge_index)
        
        assert out.shape == (2, 10, 5) # B, N, Out
        assert not torch.isnan(out).any()

    def test_sde_time_evolution(self):
        """Test that SDE actually evolves over time steps."""
        # We can't easily inspect internal state without mocking, 
        # but we can check that increasing steps changes output (due to drift/noise accumulation)
        torch.manual_seed(42)
        layer1 = GraphFractionalSDELayer(input_dim=10, hidden_dim=10, output_dim=10, num_sde_steps=1)
        
        torch.manual_seed(42)
        layer2 = GraphFractionalSDELayer(input_dim=10, hidden_dim=10, output_dim=10, num_sde_steps=5)
        # Copy weights
        layer2.load_state_dict(layer1.state_dict())
        
        x = torch.randn(1, 10, 10)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        torch.manual_seed(0)
        out1 = layer1(x, edge_index)
        
        torch.manual_seed(0)
        out2 = layer2(x, edge_index)
        
        # More steps = more drift/diff integration = different result
        assert not torch.allclose(out1, out2)
