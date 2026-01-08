"""
Optimized Fractional Calculus Optimizers

This module provides highly optimized fractional calculus optimizers that:
1. Inherit from `torch.optim.Optimizer` for standard PyTorch compatibility.
2. Implement "Fractional Gradient Descent" logic where gradients are pre-processed
   with fraction calculus operators before the standard update step.

Author: Davian R. Chin
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Dict, Any, List, Callable, Union, Tuple
import threading
import warnings
import math

# Use the robust adjoint implementation if needed, or standard autograd
from .fractional_autograd import fractional_derivative

class FractionalOptimizer(Optimizer):
    """
    Base class for fractional optimizers. 
    intercepts gradients and applies fractional derivatives before the update step.
    """
    def __init__(self, params, defaults):
        super().__init__(params, defaults)

    def _apply_fractional_gradients(self, group):
        """
        Applies fractional derivative to gradients in place or returns modified list.
        Currently we modify the `grad` attribute in-place temporarily or return tensor.
        """
        alpha = group['fractional_order']
        method = group['method']
        use_fractional = group.get('use_fractional', True)

        if not use_fractional or alpha == 1.0:
            return

        for p in group['params']:
            if p.grad is not None:
                # We apply the fractional derivative to the gradient tensor
                g = p.grad.data
                
                try:
                    # Only apply if dimensions allow (needs at least 1D)
                    if g.dim() >= 1:
                        # Normalize to preserve magnitude stability
                        orig_norm = torch.norm(g)
                        new_grad = fractional_derivative(g, alpha, method)
                        new_norm = torch.norm(new_grad)
                        
                        if new_norm > 1e-12:
                            scale = orig_norm / new_norm
                            p.grad.data = new_grad * scale
                        else:
                            p.grad.data = new_grad
                    
                except Exception as e:
                    # Fallback to scaling if spatial conv fails (e.g. too small)
                    pass


class OptimizedFractionalSGD(FractionalOptimizer):
    """
    Optimized SGD with fractional calculus integration.
    """
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 fractional_order=0.5, method="RL", use_fractional=True):
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        fractional_order=fractional_order, method=method,
                        use_fractional=use_fractional)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._apply_fractional_gradients(group)
            
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]

                if weight_decay != 0:
                    d_p = d_p.add(param, alpha=weight_decay)

                if momentum != 0:
                    buf = momentum_buffer_list[i]
                    if buf is None:
                        buf = d_p.clone().detach()
                        momentum_buffer_list[i] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                param.add_(d_p, alpha=-lr)
                
                if momentum != 0:
                    self.state[param]['momentum_buffer'] = momentum_buffer_list[i]

        return loss


class OptimizedFractionalAdam(FractionalOptimizer):
    """
    Optimized Adam with fractional calculus integration.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False,
                 fractional_order=0.5, method="RL", use_fractional=True):
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        fractional_order=fractional_order, method=method,
                        use_fractional=use_fractional)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._apply_fractional_gradients(group)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients')
                
                amsgrad = group['amsgrad']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if amsgrad:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class OptimizedFractionalRMSprop(FractionalOptimizer):
    """
    Optimized RMSprop with fractional calculus integration.
    """
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False,
                 fractional_order=0.5, method="RL", use_fractional=True):
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay,
                        fractional_order=fractional_order, method=method, use_fractional=use_fractional)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._apply_fractional_gradients(group)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.add_(buf, alpha=-group['lr'])
                else:
                    p.addcdiv_(grad, avg, value=-group['lr'])

        return loss

# Aliases for backward compatibility
OptimizedBaseOptimizer = FractionalOptimizer

# Factory functions
def create_optimized_sgd(lr=0.001, momentum=0.0, **kwargs):
    return lambda params: OptimizedFractionalSGD(params, lr=lr, momentum=momentum, **kwargs)

def create_optimized_adam(lr=0.001, betas=(0.9, 0.999), **kwargs):
    return lambda params: OptimizedFractionalAdam(params, lr=lr, betas=betas, **kwargs)
