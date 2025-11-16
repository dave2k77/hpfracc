import os
import math
import pytest

pytestmark = pytest.mark.week3


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not available")
def test_fractional_trainer_single_epoch_decreases_loss():
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from hpfracc.ml.training import create_fractional_trainer, create_fractional_scheduler

    # Simple linear regression: y = 2x + 1 with noise
    torch.manual_seed(0)
    x = torch.linspace(-1.0, 1.0, steps=64).unsqueeze(1)
    y = 2.0 * x + 1.0

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = torch.nn.Sequential(torch.nn.Linear(1, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # Fast-changing scheduler to exercise code paths
    scheduler = create_fractional_scheduler(opt, scheduler_type="step", step_size=1, gamma=0.5)

    trainer = create_fractional_trainer(
        model=model,
        optimizer=opt,
        scheduler=scheduler,
        fractional_order=0.5,
        method="RL",
    )

    # Capture initial loss on a fixed batch
    with torch.no_grad():
        init_loss = torch.nn.functional.mse_loss(model(x), y).item()

    # Train for 1 epoch against itself as validation
    history = trainer.train(loader, loader, num_epochs=1)

    assert "training_losses" in history and len(history["training_losses"]) == 1
    assert math.isfinite(history["training_losses"][0])

    # Loss should not explode and should generally improve vs. initial
    with torch.no_grad():
        final_loss = torch.nn.functional.mse_loss(model(x), y).item()
    assert final_loss <= max(init_loss, 1e-8)


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not available")
def test_fractional_trainer_train_and_validate_steps_smoke():
    import torch
    from hpfracc.ml.training import FractionalTrainer

    torch.manual_seed(1)
    x = torch.randn(8, 3)
    y = torch.randn(8, 2)

    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    trainer = FractionalTrainer(model=model, optimizer=opt)

    loss1 = trainer.train_step(x, y)
    loss2 = trainer.validate_step(x, y)

    assert isinstance(loss1, float) and isinstance(loss2, float)
    assert math.isfinite(loss1) and math.isfinite(loss2)