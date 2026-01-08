"""Test script to reproduce Neural fSDE dimension mismatch issues"""
import torch
import sys
import traceback
from hpfracc.ml.neural_fsde import NeuralFSDEConfig, NeuralFractionalSDE, create_neural_fsde

def test_basic_forward():
    """Test basic forward pass"""
    print("=" * 60)
    print("Test 1: Basic forward pass")
    print("=" * 60)
    try:
        config = NeuralFSDEConfig(
            input_dim=2,
            output_dim=2,
            hidden_dim=16,
            fractional_order=0.5,
            diffusion_dim=1
        )
        model = NeuralFractionalSDE(config)
        
        t = torch.tensor([0.0, 0.1, 0.2])
        x0 = torch.tensor([[1.0, 0.5]])
        
        print(f"Model input_dim: {model.input_dim}")
        print(f"Model output_dim: {model.output_dim}")
        print(f"x0 shape: {x0.shape}")
        print(f"t shape: {t.shape}")
        
        with torch.no_grad():
            trajectory = model.forward(x0, t)
        
        print(f"✓ Success! Trajectory shape: {trajectory.shape}")
        return True
    except Exception as e:
        print(f"✗ Failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def test_drift_function():
    """Test drift function directly"""
    print("\n" + "=" * 60)
    print("Test 2: Drift function")
    print("=" * 60)
    try:
        config = NeuralFSDEConfig(
            input_dim=2,
            output_dim=2,
            hidden_dim=16,
            fractional_order=0.5,
            diffusion_dim=1
        )
        model = NeuralFractionalSDE(config)
        
        t = torch.tensor(0.5)
        x = torch.tensor([[1.0, 0.5]])
        
        print(f"Model input_dim: {model.input_dim}")
        print(f"x shape: {x.shape}")
        print(f"t: {t}")
        
        drift = model.drift_function(t, x)
        print(f"✓ Success! Drift shape: {drift.shape}")
        return True
    except Exception as e:
        print(f"✗ Failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def test_diffusion_function():
    """Test diffusion function directly"""
    print("\n" + "=" * 60)
    print("Test 3: Diffusion function")
    print("=" * 60)
    try:
        config = NeuralFSDEConfig(
            input_dim=2,
            output_dim=2,
            hidden_dim=16,
            fractional_order=0.5,
            diffusion_dim=1
        )
        model = NeuralFractionalSDE(config)
        
        t = torch.tensor(0.5)
        x = torch.tensor([[1.0, 0.5]])
        
        print(f"Model input_dim: {model.input_dim}")
        print(f"x shape: {x.shape}")
        print(f"t: {t}")
        
        diffusion = model.diffusion_function(t, x)
        print(f"✓ Success! Diffusion shape: {diffusion.shape}")
        return True
    except Exception as e:
        print(f"✗ Failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def test_create_factory():
    """Test factory function"""
    print("\n" + "=" * 60)
    print("Test 4: Factory function")
    print("=" * 60)
    try:
        model = create_neural_fsde(
            input_dim=2,
            output_dim=2,
            hidden_dim=16,
            fractional_order=0.5
        )
        
        t = torch.tensor([0.0, 0.1, 0.2])
        x0 = torch.tensor([[1.0, 0.5]])
        
        print(f"Model input_dim: {model.input_dim}")
        print(f"Model output_dim: {model.output_dim}")
        
        with torch.no_grad():
            trajectory = model.forward(x0, t)
        
        print(f"✓ Success! Trajectory shape: {trajectory.shape}")
        return True
    except Exception as e:
        print(f"✗ Failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Neural fSDE Debug Test Suite")
    print("=" * 60)
    
    results = []
    results.append(("Basic forward pass", test_basic_forward()))
    results.append(("Drift function", test_drift_function()))
    results.append(("Diffusion function", test_diffusion_function()))
    results.append(("Factory function", test_create_factory()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

