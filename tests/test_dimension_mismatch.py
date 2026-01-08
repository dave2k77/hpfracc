"""Test case that should trigger dimension mismatch"""
import torch
import torch.nn as nn
from hpfracc.ml.neural_fsde import NeuralFSDEConfig, NeuralFractionalSDE

# Create a custom network with WRONG input dimension
# This should trigger the dimension mismatch error
drift_net = nn.Sequential(
    nn.Linear(4, 16),  # WRONG: Should be 3 (input_dim + 1 = 2 + 1), but using 4
    nn.ReLU(),
    nn.Linear(16, 2)
)

config = NeuralFSDEConfig(
    input_dim=2,
    output_dim=2,
    hidden_dim=16,
    fractional_order=0.5,
    diffusion_dim=1,
    drift_net=drift_net  # Custom network with wrong dimension
)

print("Creating model with custom network (wrong input dim)...")
model = NeuralFractionalSDE(config)

# Try to use it
t = torch.tensor(0.5)
x = torch.tensor([[1.0, 0.5]])

print("Testing drift function with mismatched dimensions...")
try:
    drift = model.drift_function(t, x)
    print(f"Unexpected success! Drift shape: {drift.shape}")
except Exception as e:
    print(f"Expected error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

