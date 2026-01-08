"""
Comprehensive test script for GNN layers, GPU optimization, and probabilistic layers
Tests various edge cases and device scenarios
"""
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from hpfracc.ml.gnn_layers import FractionalGraphConv, FractionalGraphAttention, FractionalGraphPooling
from hpfracc.ml.gpu_optimization import GPUOptimizedSpectralEngine
from hpfracc.ml.probabilistic_fractional_orders import create_normal_alpha_layer

def test_gnn_layers_comprehensive():
    """Comprehensive GNN layer tests"""
    print("Testing GNN layers comprehensively...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test 1: Basic forward pass
    print("  Test 1: Basic forward pass")
    layer = FractionalGraphConv(in_channels=10, out_channels=20)
    x = torch.randn(5, 10, device=device)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], device=device, dtype=torch.long)
    try:
        out = layer(x, edge_index)
        assert out.device == x.device, f"Device mismatch: out={out.device}, x={x.device}"
        print("    [OK] Basic forward pass")
    except Exception as e:
        print(f"    [FAIL] Basic forward pass: {e}")
        return False
    
    # Test 2: With edge weights
    print("  Test 2: With edge weights")
    edge_weight = torch.randn(5, device=device)
    try:
        out = layer(x, edge_index, edge_weight=edge_weight)
        assert out.device == x.device, f"Device mismatch: out={out.device}, x={x.device}"
        print("    [OK] Forward pass with edge weights")
    except Exception as e:
        print(f"    [FAIL] Forward pass with edge weights: {e}")
        return False
    
    # Test 3: Attention layer
    print("  Test 3: Attention layer")
    attn_layer = FractionalGraphAttention(in_channels=10, out_channels=20, heads=4)
    try:
        out = attn_layer(x, edge_index)
        assert out.device == x.device, f"Device mismatch: out={out.device}, x={x.device}"
        print("    [OK] Attention layer forward pass")
    except Exception as e:
        print(f"    [FAIL] Attention layer: {e}")
        return False
    
    # Test 4: Pooling layer
    print("  Test 4: Pooling layer")
    pool_layer = FractionalGraphPooling(in_channels=10, out_channels=5)
    batch = torch.zeros(5, dtype=torch.long, device=device)
    try:
        pooled_features, pooled_edge_index, pooled_batch = pool_layer(x, edge_index, batch=batch)
        assert pooled_features.device == x.device, f"Device mismatch: pooled={pooled_features.device}, x={x.device}"
        print("    [OK] Pooling layer forward pass")
    except Exception as e:
        print(f"    [FAIL] Pooling layer: {e}")
        return False
    
    # Test 5: Layer.to(device) method
    print("  Test 5: Layer.to(device) method")
    layer_cpu = FractionalGraphConv(in_channels=10, out_channels=20)
    if device == 'cuda':
        try:
            layer_cpu = layer_cpu.to(device)
            x_cuda = torch.randn(5, 10, device=device)
            edge_index_cuda = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], device=device, dtype=torch.long)
            out = layer_cpu(x_cuda, edge_index_cuda)
            assert out.device == device, f"Device mismatch after .to(): out={out.device}, expected={device}"
            print("    [OK] Layer.to(device) method")
        except Exception as e:
            print(f"    [FAIL] Layer.to(device): {e}")
            return False
    else:
        print("    [SKIP] GPU not available for .to(device) test")
    
    return True

def test_gpu_optimization_comprehensive():
    """Comprehensive GPU optimization tests"""
    print("\nTesting GPU optimization comprehensively...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test 1: Basic forward pass
    print("  Test 1: Basic forward pass")
    engine = GPUOptimizedSpectralEngine(engine_type="fft", use_amp=True, chunk_size=512)
    x = torch.randn(16, 1024, device=device)
    alpha = 0.5
    try:
        result = engine.forward(x, alpha)
        assert result.device == x.device, f"Device mismatch: result={result.device}, x={x.device}"
        print("    [OK] Basic forward pass")
    except Exception as e:
        print(f"    [FAIL] Basic forward pass: {e}")
        return False
    
    # Test 2: Different engine types
    print("  Test 2: Different engine types")
    for engine_type in ["fft", "mellin", "laplacian"]:
        try:
            engine = GPUOptimizedSpectralEngine(engine_type=engine_type, use_amp=False)
            result = engine.forward(x, alpha)
            assert result.device == x.device, f"Device mismatch for {engine_type}"
            print(f"    [OK] {engine_type} engine")
        except Exception as e:
            print(f"    [FAIL] {engine_type} engine: {e}")
            return False
    
    return True

def test_probabilistic_layers_comprehensive():
    """Comprehensive probabilistic layer tests"""
    print("\nTesting probabilistic layers comprehensively...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test 1: Basic forward pass
    print("  Test 1: Basic forward pass")
    layer = create_normal_alpha_layer(mean=0.5, std=0.1, learnable=True)
    layer = layer.to(device)
    x = torch.randn(10, 10, device=device)
    try:
        out = layer(x)
        assert out.device == x.device, f"Device mismatch: out={out.device}, x={x.device}"
        print("    [OK] Basic forward pass")
    except Exception as e:
        print(f"    [FAIL] Basic forward pass: {e}")
        return False
    
    # Test 2: Learnable parameters
    print("  Test 2: Learnable parameters")
    try:
        # Check if parameters are on correct device
        if hasattr(layer, 'probabilistic_order') and hasattr(layer.probabilistic_order, 'loc'):
            param_device = str(layer.probabilistic_order.loc.device)
            expected_device = device if isinstance(device, str) else str(device)
            # Handle 'cpu' vs 'cuda:0' format differences
            if param_device.startswith('cpu') and expected_device.startswith('cpu'):
                pass  # OK
            elif param_device.startswith('cuda') and expected_device.startswith('cuda'):
                pass  # OK
            else:
                assert False, f"Parameter device mismatch: param={param_device}, expected={expected_device}"
        print("    [OK] Learnable parameters on correct device")
    except Exception as e:
        print(f"    [FAIL] Learnable parameters: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Comprehensive Testing: GNN layers, GPU optimization, probabilistic layers")
    print("=" * 60)
    
    success = True
    success &= test_gnn_layers_comprehensive()
    success &= test_gpu_optimization_comprehensive()
    success &= test_probabilistic_layers_comprehensive()
    
    print("\n" + "=" * 60)
    if success:
        print("All comprehensive tests PASSED")
    else:
        print("Some comprehensive tests FAILED")
    print("=" * 60)
