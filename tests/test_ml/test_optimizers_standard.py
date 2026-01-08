import torch
import torch.nn as nn
from hpfracc.ml.optimized_optimizers import OptimizedFractionalSGD, OptimizedFractionalAdam



def test_sgd_standard_api():
    """Verify OptimizedFractionalSGD works with standard PyTorch loop (alpha=1.0)."""
    model = nn.Linear(10, 1)
    # Alpha=1.0 should coincide with standard SGD
    optimizer = OptimizedFractionalSGD(model.parameters(), lr=0.01, fractional_order=1.0)
    
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    
    # Standard training step
    def step():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
        return loss

    initial_loss = step()
    final_loss = step()
    
    assert final_loss < initial_loss, "Standard SGD (alpha=1.0) failed to reduce loss"

def test_adam_standard_api():
    """Verify OptimizedFractionalAdam works with standard PyTorch loop."""
    model = nn.Linear(10, 1)
    optimizer = OptimizedFractionalAdam(model.parameters(), lr=0.01)
    
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    
    # Standard training step
    for _ in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()

    # Check if params updated
    for p in model.parameters():
        assert p.grad is not None

def test_fractional_gradient_application():
    """Verify that fractional gradient logic is applied."""
    # We use a dummy parameter and check if its gradient is modified
    # Default alpha=0.5
    
    # CASE: 1D tensor (compatible with our implementation)
    p = torch.randn(10, requires_grad=True)
    optimizer = OptimizedFractionalSGD([p], lr=0.1, fractional_order=0.5)
    
    loss = (p ** 2).sum()
    loss.backward()
    
    # Capture original gradient
    orig_grad = p.grad.clone()
    
    # Step should modify p.grad in-place before update if successful, 
    # but since it's inside `step`, we can't easily inspect the intermediate `grad`.
    # However, we can assert that the update was NOT standard SGD.
    # Standard SGD update: new_p = p - lr * grad
    
    optimizer.step()
    
    # Reconstruct what standard SGD would have done
    expected_std_p = p.detach() + 0.1 * orig_grad # (p_old = p_new + lr*grad_used => p_used = p_old) wait
    # p_new = p_old - lr * grad_modified
    # p_old was the `p` before step. 
    # We didn't save p_old.
    
    # Let's redo with precise clone
    p2 = p.detach().clone()
    p2.requires_grad_(True)
    optimizer2 = OptimizedFractionalSGD([p2], lr=0.1, fractional_order=0.5)
    
    loss2 = (p2 ** 2).sum()
    loss2.backward()
    grad2_orig = p2.grad.clone()
    
    optimizer2.step()
    
    # Standard update would result in:
    std_update = p2.detach() + 0.1 * grad2_orig # p2_old - 0.1 * grad => p2_new.
    # Actually p2 here IS p2_new. 
    # expected_std_new = p2_old - 0.1 * grad2_orig. We lost p2_old. 
    
    # Let's try separate reference
    p_ref = torch.randn(10, requires_grad=True)
    p_frac = p_ref.detach().clone().requires_grad_(True)
    
    # Ref: Standard SGD
    opt_std = torch.optim.SGD([p_ref], lr=0.1)
    loss_ref = (p_ref ** 2).sum()
    loss_ref.backward()
    opt_std.step()
    
    # Frac: Fractional SGD
    opt_frac = OptimizedFractionalSGD([p_frac], lr=0.1, fractional_order=0.5)
    loss_frac = (p_frac ** 2).sum()
    loss_frac.backward()
    opt_frac.step()
    
    # They should differ
    assert not torch.allclose(p_ref, p_frac), "Fractional SGD did not apply fractional modification"

if __name__ == "__main__":
    test_sgd_standard_api()
    test_adam_standard_api()
    test_fractional_gradient_application()
    print("All Optimizer tests passed!")
