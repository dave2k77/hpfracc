"""Test script to reproduce Neural fSDE dimension mismatch issues"""
import torch
from hpfracc.ml.neural_fsde import NeuralFSDEConfig, NeuralFractionalSDE, create_neural_fsde

def test_basic_forward():
    """Test basic forward pass"""
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
    
    with torch.no_grad():
        trajectory = model.forward(x0, t)
    
    assert trajectory.dim() == 3  # (time_steps, batch, dim)
    assert trajectory.shape == (101, 1, 2)

def test_drift_function():
    """Test drift function directly"""
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
    
    drift = model.drift_function(t, x)
    assert drift.shape == (1, 2)

def test_diffusion_function():
    """Test diffusion function directly"""
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
    
    diffusion = model.diffusion_function(t, x)
    assert diffusion.shape == (1, 1)  # (batch_size, diffusion_dim)

def test_create_factory():
    """Test factory function"""
    model = create_neural_fsde(
        input_dim=2,
        output_dim=2,
        hidden_dim=16,
        fractional_order=0.5
    )
    
    t = torch.tensor([0.0, 0.1, 0.2])
    x0 = torch.tensor([[1.0, 0.5]])
    
    with torch.no_grad():
        trajectory = model.forward(x0, t)
    
    assert trajectory.dim() == 3
    assert trajectory.shape == (101, 1, 2)

