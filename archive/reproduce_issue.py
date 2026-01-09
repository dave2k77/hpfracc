"""Minimal reproduction script for Neural fSDE dimension mismatch"""
import torch
from hpfracc.ml.neural_fsde import NeuralFSDEConfig, NeuralFractionalSDE

# Create model
config = NeuralFSDEConfig(
    input_dim=2,
    output_dim=2,
    hidden_dim=16,
    fractional_order=0.5,
    diffusion_dim=1
)
model = NeuralFractionalSDE(config)

# Test forward pass
t = torch.tensor([0.0, 0.1, 0.2])
x0 = torch.tensor([[1.0, 0.5]])

print("Running forward pass...")
try:
    with torch.no_grad():
        trajectory = model.forward(x0, t)
    print(f"Success! Trajectory shape: {trajectory.shape}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

