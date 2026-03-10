"""Verify the fix works correctly"""
import pytest
import torch
import torch.nn as nn
from hpfracc.ml.neural_fsde import NeuralFSDEConfig, NeuralFractionalSDE

def test_correct_dimensions():
    """Test 1: Custom network with CORRECT dimensions (should work)"""
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
    assert drift.shape == (1, 2)

def test_wrong_dimensions():
    """Test 2: Custom network with WRONG dimensions (should fail early)"""
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
    
    with pytest.raises(ValueError, match="Custom drift network input dimension mismatch"):
        model = NeuralFractionalSDE(config)

def test_default_network():
    """Test 3: Default network (should work)"""
    config = NeuralFSDEConfig(
        input_dim=2,
        output_dim=2
    )
    
    model = NeuralFractionalSDE(config)
    t = torch.tensor(0.5)
    x = torch.tensor([[1.0, 0.5]])
    drift = model.drift_function(t, x)
    assert drift.shape == (1, 2)

