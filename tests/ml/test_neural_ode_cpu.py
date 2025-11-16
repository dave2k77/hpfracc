import pytest

pytest.importorskip("torch")

import torch
from hpfracc.ml.neural_ode import NeuralODE


def test_neural_ode_euler_fallback(tiny_ode_data, cpu_only, set_seed):
    set_seed(42)

    batch = tiny_ode_data["batch_size"]
    dim = tiny_ode_data["state_dim"]
    t = torch.tensor(tiny_ode_data["t"], dtype=torch.float32)

    x = torch.randn(batch, dim, dtype=torch.float32)

    model = NeuralODE(input_dim=dim, hidden_dim=8, output_dim=dim, solver="euler")

    y = model(x, t)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (batch, t.numel(), dim)
    assert torch.isfinite(y).all()


def test_neural_ode_forward_shapes(tiny_ode_data, set_seed):
    set_seed(0)

    batch = tiny_ode_data["batch_size"]
    dim = tiny_ode_data["state_dim"]
    t = torch.tensor(tiny_ode_data["t"], dtype=torch.float32)

    x = torch.zeros(batch, dim, dtype=torch.float32)
    model = NeuralODE(input_dim=dim, hidden_dim=16, output_dim=dim, solver="euler")

    y = model(x, t)
    assert y.ndim == 3
    assert y.shape[0] == batch
    assert y.shape[1] == t.numel()
    assert y.shape[2] == dim
