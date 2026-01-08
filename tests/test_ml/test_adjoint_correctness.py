import torch
import pytest
from hpfracc.ml.adjoint_optimization import adjoint_fractional_derivative, AdjointFractionalLayer, AdjointConfig
from hpfracc.ml.fractional_autograd import fractional_derivative

def test_adjoint_matches_autograd():
    """Verify that the adjoint (checkpointed) implementation matches the standard autograd one exactly."""
    
    # Setup inputs
    x = torch.randn(10, 5, 100, requires_grad=True)
    alpha = 0.6
    method = "RL"
    
    # Compute using standard autograd
    y_std = fractional_derivative(x, alpha, method)
    loss_std = y_std.sum()
    grad_std = torch.autograd.grad(loss_std, x, retain_graph=True)[0]
    
    # Compute using adjoint (checkpointing)
    # We must reset gradients or use a new tensor if we want clean comparison, 
    # but here we use a separate computation path on same tensor (conceptually valid if graph allows)
    # Better to use cloned tensor to be safe
    x_adj = x.detach().clone().requires_grad_(True)
    y_adj = adjoint_fractional_derivative(x_adj, alpha, method)
    loss_adj = y_adj.sum()
    grad_adj = torch.autograd.grad(loss_adj, x_adj, retain_graph=True)[0]
    
    # Check forward pass matching
    diff = (y_std - y_adj).abs().max()
    print(f"Max forward diff: {diff}")
    assert torch.allclose(y_std, y_adj, atol=1e-5), f"Adjoint forward pass mismatch. Max diff: {diff}"
    
    # Check backward pass matching
    grad_diff = (grad_std - grad_adj).abs().max()
    print(f"Max backward diff: {grad_diff}")
    assert torch.allclose(grad_std, grad_adj, atol=1e-5), f"Adjoint backward pass mismatch. Max diff: {grad_diff}"

def test_adjoint_layer():
    """Verify AdjointFractionalLayer works in a model context."""
    config = AdjointConfig(use_adjoint=True)
    layer = AdjointFractionalLayer(alpha=0.5, method="RL", config=config)
    
    x = torch.randn(2, 3, 50, requires_grad=True)
    y = layer(x)
    
    assert y.shape == x.shape
    assert y.requires_grad
    
    loss = y.mean()
    loss.backward()
    assert x.grad is not None

if __name__ == "__main__":
    test_adjoint_matches_autograd()
    test_adjoint_layer()
    print("All Adjoint tests passed!")
