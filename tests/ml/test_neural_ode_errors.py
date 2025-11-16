import pytest

pytest.importorskip("torch")

import torch
from hpfracc.ml.neural_ode import NeuralODE, create_neural_ode


def test_neural_ode_graceful_solver_fallback(tiny_ode_data):
    batch = tiny_ode_data["batch_size"]
    dim = tiny_ode_data["state_dim"]
    t = torch.tensor(tiny_ode_data["t"], dtype=torch.float32)
    x = torch.randn(batch, dim, dtype=torch.float32)

    # Current API should not raise on unknown solver; it just won't match dopri5
    model = NeuralODE(input_dim=dim, hidden_dim=8, output_dim=dim, solver="unknown_solver")
    y = model(x, t)

    assert y.shape == (batch, t.numel(), dim)


def test_create_neural_ode_invalid_model_type():
    with pytest.raises(ValueError):
        create_neural_ode(model_type="does_not_exist", input_dim=2, hidden_dim=4, output_dim=2)
