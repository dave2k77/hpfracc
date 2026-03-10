"""Test case that should trigger dimension mismatch"""
import pytest
import torch
import torch.nn as nn
from hpfracc.ml.neural_fsde import NeuralFSDEConfig, NeuralFractionalSDE

def test_dimension_mismatch():
    """Create a custom network with WRONG input dimension.
    This should trigger the dimension mismatch error during initialization.
    """
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
        drift_net=drift_net
    )

    # Initializing the model with the bad config should throw immediately
    with pytest.raises(ValueError, match="Custom drift network input dimension mismatch"):
        model = NeuralFractionalSDE(config)

