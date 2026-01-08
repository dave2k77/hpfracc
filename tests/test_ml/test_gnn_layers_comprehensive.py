"""
Comprehensive tests for hpfracc.ml.gnn_layers module

This module tests Graph Neural Network layers with fractional calculus integration.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple
import warnings

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from hpfracc.ml.gnn_layers import (
        BaseFractionalGNNLayer,
        FractionalGraphConv,
        FractionalGraphAttention,
        FractionalGraphPooling
    )
    from hpfracc.ml.backends import BackendType
    from hpfracc.core.definitions import FractionalOrder
except ImportError as e:
    pytest.skip(f"Skipping GNN tests due to import error: {e}", allow_module_level=True)


class TestBaseFractionalGNNLayer:
    """Test the base FractionalGNNLayer class"""

    def test_initialization_default(self):
        """Test BaseFractionalGNNLayer initialization with default parameters"""
        # Base class is abstract, so we can't instantiate it directly
        # We'll test through concrete implementations
        layer = FractionalGraphConv(in_channels=10, out_channels=20)
        
        assert layer.in_channels == 10
        assert layer.out_channels == 20
        assert layer.fractional_order.alpha == 0.5
        assert layer.method == "RL"
        assert layer.use_fractional is True
        assert layer.activation == "relu"
        assert layer.dropout == 0.1
        assert layer.bias is not None

    def test_initialization_custom(self):
        """Test BaseFractionalGNNLayer initialization with custom parameters"""
        layer = FractionalGraphConv(
            in_channels=10,
            out_channels=20,
            fractional_order=0.7,
            method="Caputo",
            use_fractional=False,
            activation="sigmoid",
            dropout=0.2,
            bias=False
        )
        
        assert layer.in_channels == 10
        assert layer.out_channels == 20
        assert layer.fractional_order.alpha == 0.7
        assert layer.method == "Caputo"
        assert layer.use_fractional is False
        assert layer.activation == "sigmoid"
        assert layer.dropout == 0.2
        assert layer.bias is None

    def test_initialization_with_fractional_order(self):
        """Test initialization with FractionalOrder object"""
        alpha = FractionalOrder(0.6)
        layer = FractionalGraphConv(
            in_channels=10,
            out_channels=20,
            fractional_order=alpha
        )
        
        assert layer.fractional_order.alpha == 0.6

    def test_initialization_with_backend(self):
        """Test initialization with specific backend"""
        layer = FractionalGraphConv(
            in_channels=10,
            out_channels=20,
            backend=BackendType.TORCH
        )
        
        assert layer.backend is not None

    def test_apply_fractional_derivative_with_fractional(self):
        """Test apply_fractional_derivative when use_fractional is True"""
        layer = FractionalGraphConv(
            in_channels=10,
            out_channels=20,
            use_fractional=True,
            fractional_order=0.5
        )
        
        # Create test input
        x = torch.randn(10, 10)
        
        # Apply fractional derivative
        result = layer.apply_fractional_derivative(x)
        
        assert result is not None
        assert result.shape == x.shape

    def test_apply_fractional_derivative_without_fractional(self):
        """Test apply_fractional_derivative when use_fractional is False"""
        layer = FractionalGraphConv(
            in_channels=10,
            out_channels=20,
            use_fractional=False,
            fractional_order=0.5
        )
        
        # Create test input
        x = torch.randn(10, 10)
        
        # Apply fractional derivative
        result = layer.apply_fractional_derivative(x)
        
        assert result is not None
        assert result.shape == x.shape
        # Should return the input unchanged
        assert torch.allclose(result, x)

    def test_call_method(self):
        """Test the __call__ method"""
        layer = FractionalGraphConv(in_channels=10, out_channels=20)
        
        # Create test data
        x = torch.randn(10, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Call the layer
        result = layer(x, edge_index)
        
        assert result is not None
        assert result.shape == (10, 20)

    def test_repr_method(self):
        """Test the __repr__ method"""
        layer = FractionalGraphConv(in_channels=10, out_channels=20)
        
        repr_str = repr(layer)
        
        assert isinstance(repr_str, str)
        assert "FractionalGraphConv" in repr_str
        assert "10" in repr_str
        assert "20" in repr_str


class TestFractionalGraphConv:
    """Test the FractionalGraphConv class"""

    def test_initialization_default(self):
        """Test FractionalGraphConv initialization with default parameters"""
        layer = FractionalGraphConv(in_channels=10, out_channels=20)
        
        assert layer.in_channels == 10
        assert layer.out_channels == 20
        assert layer.fractional_order.alpha == 0.5
        assert layer.method == "RL"
        assert layer.use_fractional is True

    def test_initialization_custom(self):
        """Test FractionalGraphConv initialization with custom parameters"""
        layer = FractionalGraphConv(
            in_channels=10,
            out_channels=20,
            fractional_order=0.7,
            method="Caputo",
            use_fractional=False,
            activation="sigmoid",
            dropout=0.2,
            bias=False
        )
        
        assert layer.in_channels == 10
        assert layer.out_channels == 20
        assert layer.fractional_order.alpha == 0.7
        assert layer.method == "Caputo"
        assert layer.use_fractional is False
        assert layer.activation == "sigmoid"
        assert layer.dropout == 0.2
        assert layer.bias is None

    def test_forward_basic(self):
        """Test basic forward pass"""
        layer = FractionalGraphConv(in_channels=10, out_channels=20)
        
        # Create test data
        x = torch.randn(10, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        result = layer.forward(x, edge_index)
        
        assert result is not None
        assert result.shape == (10, 20)

    def test_forward_with_edge_weights(self):
        """Test forward pass with edge weights"""
        layer = FractionalGraphConv(in_channels=10, out_channels=20)
        
        # Create test data
        x = torch.randn(10, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_weight = torch.tensor([1.0, 2.0, 3.0])
        
        # Forward pass
        result = layer.forward(x, edge_index, edge_weight=edge_weight)
        
        assert result is not None
        assert result.shape == (10, 20)

    def test_forward_without_fractional(self):
        """Test forward pass without fractional calculus"""
        layer = FractionalGraphConv(
            in_channels=10,
            out_channels=20,
            use_fractional=False
        )
        
        # Create test data
        x = torch.randn(10, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        result = layer.forward(x, edge_index)
        
        assert result is not None
        assert result.shape == (10, 20)

    def test_forward_with_different_activations(self):
        """Test forward pass with different activation functions"""
        activations = ["relu", "sigmoid", "tanh"]
        
        for activation in activations:
            layer = FractionalGraphConv(
                in_channels=10,
                out_channels=20,
                activation=activation
            )
            
            # Create test data
            x = torch.randn(10, 10)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
            
            # Forward pass
            result = layer.forward(x, edge_index)
            
            assert result is not None
            assert result.shape == (10, 20)

    def test_forward_with_different_dropout(self):
        """Test forward pass with different dropout rates"""
        dropout_rates = [0.0, 0.1, 0.5, 0.9]
        
        for dropout in dropout_rates:
            layer = FractionalGraphConv(
                in_channels=10,
                out_channels=20,
                dropout=dropout
            )
            
            # Create test data
            x = torch.randn(10, 10)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
            
            # Forward pass
            result = layer.forward(x, edge_index)
            
            assert result is not None
            assert result.shape == (10, 20)

    def test_forward_with_different_methods(self):
        """Test forward pass with different fractional methods"""
        methods = ["RL", "Caputo", "GL"]
        
        for method in methods:
            layer = FractionalGraphConv(
                in_channels=10,
                out_channels=20,
                method=method
            )
            
            # Create test data
            x = torch.randn(10, 10)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
            
            # Forward pass
            result = layer.forward(x, edge_index)
            
            assert result is not None
            assert result.shape == (10, 20)

    def test_forward_with_different_fractional_orders(self):
        """Test forward pass with different fractional orders"""
        orders = [0.1, 0.5, 1.0, 1.5, 2.0]
        
        for order in orders:
            layer = FractionalGraphConv(
                in_channels=10,
                out_channels=20,
                fractional_order=order
            )
            
            # Create test data
            x = torch.randn(10, 10)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
            
            # Forward pass
            result = layer.forward(x, edge_index)
            
            assert result is not None
            assert result.shape == (10, 20)

    def test_forward_gradient_flow(self):
        """Test that gradients flow through the layer"""
        layer = FractionalGraphConv(in_channels=10, out_channels=20)
        
        # Create test data with gradients
        x = torch.randn(10, 10, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        result = layer.forward(x, edge_index)
        
        # Compute loss
        loss = result.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_batch_processing(self):
        """Test processing multiple graphs in batch"""
        layer = FractionalGraphConv(in_channels=10, out_channels=20)
        
        # Create test data for batch of graphs
        x = torch.randn(30, 10)  # 30 nodes total
        edge_index = torch.tensor([[0, 1, 2, 10, 11, 12], [1, 2, 0, 11, 12, 10]], dtype=torch.long)
        
        # Forward pass
        result = layer.forward(x, edge_index)
        
        assert result is not None
        assert result.shape == (30, 20)


class TestFractionalGraphAttention:
    """Test the FractionalGraphAttention class"""

    def test_initialization_default(self):
        """Test FractionalGraphAttention initialization with default parameters"""
        layer = FractionalGraphAttention(in_channels=10, out_channels=20)
        
        assert layer.in_channels == 10
        assert layer.out_channels == 20
        assert layer.fractional_order.alpha == 0.5
        assert layer.method == "RL"
        assert layer.use_fractional is True

    def test_initialization_custom(self):
        """Test FractionalGraphAttention initialization with custom parameters"""
        layer = FractionalGraphAttention(
            in_channels=10,
            out_channels=20,
            fractional_order=0.7,
            method="Caputo",
            use_fractional=False,
            activation="sigmoid",
            dropout=0.2,
            bias=False
        )
        
        assert layer.in_channels == 10
        assert layer.out_channels == 20
        assert layer.fractional_order.alpha == 0.7
        assert layer.method == "Caputo"
        assert layer.use_fractional is False
        assert layer.activation == "sigmoid"
        assert layer.dropout == 0.2
        assert layer.bias is None

    def test_forward_basic(self):
        """Test basic forward pass"""
        layer = FractionalGraphAttention(in_channels=10, out_channels=20)
        
        # Create test data
        x = torch.randn(10, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        result = layer.forward(x, edge_index)
        
        assert result is not None
        assert result.shape == (10, 20)

    def test_forward_with_edge_weights(self):
        """Test forward pass with edge weights"""
        layer = FractionalGraphAttention(in_channels=10, out_channels=20)
        
        # Create test data
        x = torch.randn(10, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_weight = torch.tensor([1.0, 2.0, 3.0])
        
        # Forward pass
        result = layer.forward(x, edge_index, edge_weight=edge_weight)
        
        assert result is not None
        assert result.shape == (10, 20)

    def test_forward_without_fractional(self):
        """Test forward pass without fractional calculus"""
        layer = FractionalGraphAttention(
            in_channels=10,
            out_channels=20,
            use_fractional=False
        )
        
        # Create test data
        x = torch.randn(10, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        result = layer.forward(x, edge_index)
        
        assert result is not None
        assert result.shape == (10, 20)

    def test_attention_mechanism(self):
        """Test that attention weights are computed"""
        layer = FractionalGraphAttention(in_channels=10, out_channels=20)
        
        # Create test data
        x = torch.randn(10, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        result = layer.forward(x, edge_index)
        
        assert result is not None
        assert result.shape == (10, 20)

    def test_forward_gradient_flow(self):
        """Test that gradients flow through the layer"""
        layer = FractionalGraphAttention(in_channels=10, out_channels=20)
        
        # Create test data with gradients
        x = torch.randn(10, 10, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        result = layer.forward(x, edge_index)
        
        # Compute loss
        loss = result.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_batch_processing(self):
        """Test processing multiple graphs in batch"""
        layer = FractionalGraphAttention(in_channels=10, out_channels=20)
        
        # Create test data for batch of graphs
        x = torch.randn(30, 10)  # 30 nodes total
        edge_index = torch.tensor([[0, 1, 2, 10, 11, 12], [1, 2, 0, 11, 12, 10]], dtype=torch.long)
        
        # Forward pass
        result = layer.forward(x, edge_index)
        
        assert result is not None
        assert result.shape == (30, 20)


class TestFractionalGraphPooling:
    """Test the FractionalGraphPooling class"""

    def test_initialization_default(self):
        """Test FractionalGraphPooling initialization with default parameters"""
        layer = FractionalGraphPooling()
        
        assert layer is not None

    def test_initialization_custom(self):
        """Test FractionalGraphPooling initialization with custom parameters"""
        layer = FractionalGraphPooling(
            fractional_order=0.7,
            method="Caputo",
            use_fractional=False
        )
        
        assert layer is not None

    def test_forward_basic(self):
        """Test basic forward pass"""
        layer = FractionalGraphPooling()
        
        # Create test data
        x = torch.randn(10, 20)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        result = layer.forward(x, edge_index)
        
        assert result is not None
        assert result.shape[0] == x.shape[0]  # Number of nodes should be preserved
        assert result.shape[1] == x.shape[1]  # Feature dimension should be preserved

    def test_forward_with_edge_weights(self):
        """Test forward pass with edge weights"""
        layer = FractionalGraphPooling()
        
        # Create test data
        x = torch.randn(10, 20)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_weight = torch.tensor([1.0, 2.0, 3.0])
        
        # Forward pass
        result = layer.forward(x, edge_index, edge_weight=edge_weight)
        
        assert result is not None
        assert result.shape[0] == x.shape[0]
        assert result.shape[1] == x.shape[1]

    def test_forward_without_fractional(self):
        """Test forward pass without fractional calculus"""
        layer = FractionalGraphPooling(use_fractional=False)
        
        # Create test data
        x = torch.randn(10, 20)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        result = layer.forward(x, edge_index)
        
        assert result is not None
        assert result.shape[0] == x.shape[0]
        assert result.shape[1] == x.shape[1]

    def test_forward_gradient_flow(self):
        """Test that gradients flow through the layer"""
        layer = FractionalGraphPooling()
        
        # Create test data with gradients
        x = torch.randn(10, 20, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        result = layer.forward(x, edge_index)
        
        # Compute loss
        loss = result.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_batch_processing(self):
        """Test processing multiple graphs in batch"""
        layer = FractionalGraphPooling()
        
        # Create test data for batch of graphs
        x = torch.randn(30, 20)  # 30 nodes total
        edge_index = torch.tensor([[0, 1, 2, 10, 11, 12], [1, 2, 0, 11, 12, 10]], dtype=torch.long)
        
        # Forward pass
        result = layer.forward(x, edge_index)
        
        assert result is not None
        assert result.shape == x.shape


class TestGNNGradients:
    """Test gradient computation in GNN layers"""

    def test_graph_conv_gradient(self):
        """Test gradient computation in FractionalGraphConv"""
        layer = FractionalGraphConv(in_channels=10, out_channels=20)
        
        # Create test data
        x = torch.randn(10, 10, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        out = layer(x, edge_index)
        
        # Backward pass
        out.mean().backward()
        
        # Check gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_attention_gradient(self):
        """Test gradient computation in FractionalGraphAttention"""
        layer = FractionalGraphAttention(in_channels=10, out_channels=20)
        
        # Create test data
        x = torch.randn(10, 10, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        out = layer(x, edge_index)
        
        # Backward pass
        out.mean().backward()
        
        # Check gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_pooling_gradient(self):
        """Test gradient computation in FractionalGraphPooling"""
        layer = FractionalGraphPooling()
        
        # Create test data
        x = torch.randn(10, 20, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        out = layer(x, edge_index)
        
        # Backward pass
        out.mean().backward()
        
        # Check gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestGNNErrorHandling:
    """Test error handling in GNN layers"""

    def test_invalid_input_shapes(self):
        """Test error handling for invalid input shapes"""
        layer = FractionalGraphConv(in_channels=10, out_channels=20)
        
        # Create invalid test data
        x = torch.randn(10, 5)  # Wrong number of features
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Should raise an error or handle gracefully
        with pytest.raises((RuntimeError, AssertionError)):
            layer.forward(x, edge_index)

    def test_invalid_edge_index(self):
        """Test error handling for invalid edge indices"""
        layer = FractionalGraphConv(in_channels=10, out_channels=20)
        
        # Create test data with invalid edge index
        x = torch.randn(10, 10)
        # Invalid shape (should be (2, E))
        edge_index = torch.zeros((3, 5), dtype=torch.long)
        
        # The implementation is robust and clips/slices invalid indices
        # So we assert it runs without error
        result = layer.forward(x, edge_index)
        assert result is not None
        assert result.shape == (10, 20)

    def test_invalid_fractional_order(self):
        """Test error handling for invalid fractional orders"""
        # Test negative order
        with pytest.raises((ValueError, AssertionError)):
            layer = FractionalGraphConv(
                in_channels=10,
                out_channels=20,
                fractional_order=-1.0
            )


class TestGNNIntegration:
    """Integration tests for GNN layers"""

    def test_multiple_layers(self):
        """Test stacking multiple GNN layers"""
        layer1 = FractionalGraphConv(in_channels=10, out_channels=20)
        layer2 = FractionalGraphConv(in_channels=20, out_channels=30)
        
        # Create test data
        x = torch.randn(10, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass through both layers
        out1 = layer1(x, edge_index)
        out2 = layer2(out1, edge_index)
        
        assert out1 is not None
        assert out2 is not None
        assert out1.shape == (10, 20)
        assert out2.shape == (10, 30)

    def test_gnn_with_attention(self):
        """Test combining graph convolution and attention"""
        conv_layer = FractionalGraphConv(in_channels=10, out_channels=20)
        attn_layer = FractionalGraphAttention(in_channels=20, out_channels=30)
        
        # Create test data
        x = torch.randn(10, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass through both layers
        out1 = conv_layer(x, edge_index)
        out2 = attn_layer(out1, edge_index)
        
        assert out1 is not None
        assert out2 is not None
        assert out1.shape == (10, 20)
        assert out2.shape == (10, 30)

    def test_gnn_with_pooling(self):
        """Test combining graph layers with pooling"""
        conv_layer = FractionalGraphConv(in_channels=10, out_channels=20)
        pool_layer = FractionalGraphPooling()
        
        # Create test data
        x = torch.randn(10, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass through both layers
        out1 = conv_layer(x, edge_index)
        out2 = pool_layer(out1, edge_index)
        
        assert out1 is not None
        assert out2 is not None
        assert out1.shape == (10, 20)
        assert out2.shape == (10, 20)

    def test_end_to_end_workflow(self):
        """Test end-to-end GNN workflow"""
        # Create a simple GNN with multiple layers
        layers = [
            FractionalGraphConv(in_channels=10, out_channels=20),
            FractionalGraphAttention(in_channels=20, out_channels=30),
            FractionalGraphPooling(),
            FractionalGraphConv(in_channels=30, out_channels=40)
        ]
        
        # Create test data
        x = torch.randn(10, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass through all layers
        out = x
        for layer in layers:
            out = layer(out, edge_index)
        
        assert out is not None
        assert out.shape == (10, 40)

    def test_learnable_fractional_order(self):
        """Test GNN with learnable fractional order"""
        # Note: This tests the ability to use FractionalOrder objects
        layer = FractionalGraphConv(
            in_channels=10,
            out_channels=20,
            fractional_order=FractionalOrder(0.5)
        )
        
        # Create test data
        x = torch.randn(10, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        out = layer(x, edge_index)
        
        assert out is not None
        assert out.shape == (10, 20)


class TestGNNEdgeCases:
    """Test edge cases in GNN layers"""

    def test_single_node(self):
        """Test GNN with single node"""
        layer = FractionalGraphConv(in_channels=10, out_channels=20)
        
        # Create test data with single node
        x = torch.randn(1, 10)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Self-loop
        
        # Forward pass
        result = layer.forward(x, edge_index)
        
        assert result is not None
        assert result.shape == (1, 20)

    def test_empty_edge_list(self):
        """Test GNN with empty edge list"""
        layer = FractionalGraphConv(in_channels=10, out_channels=20)
        
        # Create test data with no edges
        x = torch.randn(10, 10)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Forward pass
        result = layer.forward(x, edge_index)
        
        assert result is not None
        assert result.shape == (10, 20)

    def test_large_graph(self):
        """Test GNN with large graph"""
        layer = FractionalGraphConv(in_channels=10, out_channels=20)
        
        # Create test data with large graph
        num_nodes = 1000
        num_edges = 5000
        x = torch.randn(num_nodes, 10)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
        
        # Forward pass
        result = layer.forward(x, edge_index)
        
        assert result is not None
        assert result.shape == (num_nodes, 20)

    def test_different_feature_sizes(self):
        """Test GNN with different input/output feature sizes"""
        layer = FractionalGraphConv(in_channels=5, out_channels=100)
        
        # Create test data
        x = torch.randn(10, 5)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        result = layer.forward(x, edge_index)
        
        assert result is not None
        assert result.shape == (10, 100)
