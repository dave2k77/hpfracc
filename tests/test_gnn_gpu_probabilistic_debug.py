"""
Test script to reproduce issues with GNN layers, GPU optimization, and probabilistic layers
"""
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from hpfracc.ml.gnn_layers import FractionalGraphConv, FractionalGraphAttention
from hpfracc.ml.gpu_optimization import GPUOptimizedSpectralEngine
from hpfracc.ml.probabilistic_fractional_orders import create_normal_alpha_layer

def test_gnn_layers():
    """Test GNN layers with GPU tensors"""
    print("Testing GNN layers...")
    
    # Test on GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create layer
    layer = FractionalGraphConv(in_channels=10, out_channels=20)
    
    # Create test data on GPU
    x = torch.randn(5, 10, device=device)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], device=device, dtype=torch.long)
    
    # Try forward pass
    try:
        out = layer(x, edge_index)
        print(f"[OK] GNN forward pass successful: {out.shape}")
        print(f"  Output device: {out.device}")
    except Exception as e:
        print(f"[FAIL] GNN forward pass failed: {e}")
        import traceback
        traceback.print_exc()

def test_gpu_optimization():
    """Test GPU optimization"""
    print("\nTesting GPU optimization...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create test data
    x = torch.randn(16, 1024, device=device)
    alpha = 0.5
    
    # Create engine
    engine = GPUOptimizedSpectralEngine(
        engine_type="fft",
        use_amp=True,
        chunk_size=512
    )
    
    try:
        result = engine.forward(x, alpha)
        print(f"[OK] GPU optimization forward pass successful: {result.shape}")
        print(f"  Output device: {result.device}")
        print(f"  Output dtype: {result.dtype}")
    except Exception as e:
        print(f"[FAIL] GPU optimization forward pass failed: {e}")
        import traceback
        traceback.print_exc()

def test_probabilistic_layers():
    """Test probabilistic layers"""
    print("\nTesting probabilistic layers...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create layer
    try:
        layer = create_normal_alpha_layer(mean=0.5, std=0.1, learnable=True)
        layer = layer.to(device)
        
        # Create test data on GPU
        x = torch.randn(10, 10, device=device)
        
        # Try forward pass
        out = layer(x)
        print(f"[OK] Probabilistic layer forward pass successful: {out.shape}")
        print(f"  Output device: {out.device}")
        print(f"  Output dtype: {out.dtype}")
    except Exception as e:
        print(f"[FAIL] Probabilistic layer forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("Testing GNN layers, GPU optimization, and probabilistic layers")
    print("=" * 60)
    
    test_gnn_layers()
    test_gpu_optimization()
    test_probabilistic_layers()
    
    print("\n" + "=" * 60)
    print("Testing complete")
    print("=" * 60)
