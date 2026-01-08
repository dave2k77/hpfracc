"""Verify the fix works correctly"""
import torch
import torch.nn as nn
from hpfracc.ml.neural_fsde import NeuralFSDEConfig, NeuralFractionalSDE

print("=" * 60)
print("Test 1: Custom network with CORRECT dimensions (should work)")
print("=" * 60)
try:
    drift_net = nn.Sequential(
        nn.Linear(3, 16),  # CORRECT: input_dim + 1 = 2 + 1 = 3
        nn.ReLU(),
        nn.Linear(16, 2)
    )
    
    config = NeuralFSDEConfig(
        input_dim=2,
        output_dim=2,
        drift_net=drift_net
    )
    
    model = NeuralFractionalSDE(config)
    t = torch.tensor(0.5)
    x = torch.tensor([[1.0, 0.5]])
    drift = model.drift_function(t, x)
    print(f"✓ Success! Drift shape: {drift.shape}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Test 2: Custom network with WRONG dimensions (should fail early)")
print("=" * 60)
try:
    drift_net = nn.Sequential(
        nn.Linear(4, 16),  # WRONG: Should be 3
        nn.ReLU(),
        nn.Linear(16, 2)
    )
    
    config = NeuralFSDEConfig(
        input_dim=2,
        output_dim=2,
        drift_net=drift_net
    )
    
    model = NeuralFractionalSDE(config)  # Should fail here
    print("UNEXPECTED SUCCESS - validation should have caught this!")
except ValueError as e:
    print(f"SUCCESS - Correctly caught error during initialization: {e}")
except Exception as e:
    print(f"FAILED - Wrong error type: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Test 3: Default network (should work)")
print("=" * 60)
try:
    config = NeuralFSDEConfig(
        input_dim=2,
        output_dim=2
    )
    
    model = NeuralFractionalSDE(config)
    t = torch.tensor(0.5)
    x = torch.tensor([[1.0, 0.5]])
    drift = model.drift_function(t, x)
    print(f"✓ Success! Drift shape: {drift.shape}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Summary: Fix validation complete")
print("=" * 60)

