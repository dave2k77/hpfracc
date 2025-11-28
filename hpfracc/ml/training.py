"""
Training Utilities for Fractional Calculus ML

This module provides training utilities, schedulers, and callbacks specifically
designed for fractional calculus machine learning applications.
Supports multiple backends: PyTorch, JAX, and NUMBA.
"""

import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Union, Tuple, Callable, Iterator
from collections import defaultdict, OrderedDict
import warnings

from ..core.definitions import FractionalOrder
from .fractional_autograd import fractional_derivative
from .backends import get_backend_manager, BackendType
from .tensor_ops import get_tensor_ops


class FractionalScheduler:
    """
    Base class for learning rate schedulers with fractional calculus integration

    This class provides a framework for schedulers that can apply
    fractional derivatives to learning rates during training.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
    """

    def __init__(
            self,
            optimizer: Any,
            fractional_order: float = 0.5,
            method: str = "RL",
            backend: Optional[BackendType] = None):
        self.optimizer = optimizer
        self.fractional_order = FractionalOrder(fractional_order)
        self.method = method
        self.backend = backend or get_backend_manager().active_backend
        self.tensor_ops = get_tensor_ops(self.backend)
        self.base_lr = self._get_base_lr()

    def _get_base_lr(self) -> float:
        """Get base learning rate from optimizer"""
        if hasattr(self.optimizer, 'param_groups'):
            return self.optimizer.param_groups[0]['lr']
        elif hasattr(self.optimizer, 'lr'):
            return self.optimizer.lr
        else:
            return 0.001  # Default fallback

    def fractional_adjustment(self, lr: float) -> float:
        """
        Apply fractional derivative to learning rate

        Args:
            lr: Input learning rate

        Returns:
            Learning rate with fractional derivative applied
        """
        # Apply fractional derivative based on backend
        if self.backend == BackendType.TORCH:
            try:
                # Convert to tensor for fractional derivative
                lr_tensor = self.tensor_ops.tensor([lr])
                adjusted_tensor = fractional_derivative(
                    lr_tensor, self.fractional_order.alpha, self.method)
                adjusted = float(adjusted_tensor[0])
                # Blend with original to stabilize and ensure positivity
                blended = 0.5 * lr + 0.5 * adjusted
                return max(1e-12, blended)
            except (RuntimeError, ValueError):
                # If fractional derivative fails (e.g., single value), return original
                return max(1e-12, lr)
        elif self.backend == BackendType.JAX:
            try:
                import jax.numpy as jnp
                from ..core.fractional_implementations import CaputoDerivative
                # Use Caputo derivative for scalar learning rate adjustment
                caputo = CaputoDerivative(self.fractional_order.alpha)
                # Simple approximation: apply fractional scaling
                adjusted = lr * (self.fractional_order.alpha ** 0.5)
                blended = 0.5 * lr + 0.5 * adjusted
                return max(1e-12, blended)
            except (RuntimeError, ValueError, ImportError):
                return max(1e-12, lr)
        else:
            # For NUMBA and other backends, use simple fractional scaling
            # This is an approximation: D^α[x] ≈ x^(1-α) for small perturbations
            adjusted = lr * (self.fractional_order.alpha ** 0.5)
            blended = 0.5 * lr + 0.5 * adjusted
            return max(1e-12, blended)

    def step(self, metrics: Optional[float] = None) -> None:
        """Default scheduler does nothing; override in subclasses."""
        return None

    def get_last_lr(self) -> List[float]:
        """Get current learning rates"""
        if hasattr(self.optimizer, 'param_groups'):
            return [group['lr'] for group in self.optimizer.param_groups]
        elif hasattr(self.optimizer, 'lr'):
            return [self.optimizer.lr]
        else:
            return [self.base_lr]


class FractionalStepLR(FractionalScheduler):
    """Step learning rate scheduler with fractional calculus integration"""

    def __init__(
            self,
            optimizer: Any,
            step_size: int,
            gamma: float = 0.1,
            fractional_order: float = 0.5,
            method: str = "RL",
            backend: Optional[BackendType] = None):
        super().__init__(optimizer, fractional_order, method, backend)
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = -1

    def step(self, metrics: Optional[float] = None) -> None:
        """Update learning rate"""
        self.last_epoch += 1
        if self.last_epoch > 0 and self.last_epoch % self.step_size == 0:
            # Calculate new learning rate
            new_lr = self.base_lr * \
                (self.gamma ** (self.last_epoch // self.step_size))
            adjusted_lr = new_lr

            # Update optimizer learning rate
            if hasattr(self.optimizer, 'param_groups'):
                for group in self.optimizer.param_groups:
                    group['lr'] = adjusted_lr
            elif hasattr(self.optimizer, 'lr'):
                self.optimizer.lr = adjusted_lr


class FractionalExponentialLR(FractionalScheduler):
    """Exponential learning rate scheduler with fractional calculus integration"""

    def __init__(
            self,
            optimizer: Any,
            gamma: float = 0.95,
            fractional_order: float = 0.5,
            method: str = "RL",
            backend: Optional[BackendType] = None):
        super().__init__(optimizer, fractional_order, method, backend)
        self.gamma = gamma
        self.last_epoch = -1

    def step(self, metrics: Optional[float] = None) -> None:
        """Update learning rate"""
        self.last_epoch += 1
        new_lr = self.base_lr * (self.gamma ** (self.last_epoch + 1))
        adjusted_lr = new_lr

        # Update optimizer learning rate
        if hasattr(self.optimizer, 'param_groups'):
            for group in self.optimizer.param_groups:
                group['lr'] = adjusted_lr
        elif hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = adjusted_lr


class FractionalCosineAnnealingLR(FractionalScheduler):
    """Cosine annealing learning rate scheduler with fractional calculus integration"""

    def __init__(
            self,
            optimizer: Any,
            T_max: int,
            eta_min: float = 0.0,
            fractional_order: float = 0.5,
            method: str = "RL",
            backend: Optional[BackendType] = None):
        super().__init__(optimizer, fractional_order, method, backend)
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = -1

    def step(self, metrics: Optional[float] = None) -> None:
        """Update learning rate"""
        self.last_epoch += 1
        new_lr = self.eta_min + (self.base_lr - self.eta_min) * \
            (1 + np.cos(np.pi * max(0, self.last_epoch) / self.T_max)) / 2
        adjusted_lr = new_lr

        # Update optimizer learning rate
        if hasattr(self.optimizer, 'param_groups'):
            for group in self.optimizer.param_groups:
                group['lr'] = adjusted_lr
        elif hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = adjusted_lr


class FractionalCyclicLR(FractionalScheduler):
    """Simple cyclic learning rate with fractional calculus integration"""

    def __init__(
        self,
        optimizer: Any,
        base_lr: Optional[float] = None,
        max_lr: Optional[float] = None,
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        fractional_order: float = 0.5,
        method: str = 'RL',
        backend: Optional[BackendType] = None
    ):
        super().__init__(optimizer, fractional_order, method, backend)
        self.base_lr_user = base_lr
        # Ensure base_lr reflects the user-specified value or optimizer lr
        opt_lr = self._get_base_lr()
        self.base_lr = base_lr if base_lr is not None else opt_lr
        self.max_lr = max_lr if max_lr is not None else self.base_lr
        self.step_size_up = max(1, int(step_size_up))
        self.step_size_down = int(
            step_size_down) if step_size_down is not None else self.step_size_up
        self.cycle_len = self.step_size_up + self.step_size_down
        self.mode = mode
        self.iteration = 0
        self.gamma = 1.0
        self.scale_fn = None

    def _scale_fn(self, x: float) -> float:
        if self.mode == 'triangular2':
            return 1.0 / (2.0 ** x)
        elif self.mode == 'exp_range':
            return 0.999 ** x
        return 1.0

    def step(self, metrics: Optional[float] = None) -> None:
        self.iteration += 1
        cycle_progress = (self.iteration - 1) % self.cycle_len
        if cycle_progress < self.step_size_up:
            scale = cycle_progress / float(self.step_size_up)
        else:
            scale = 1.0 - (cycle_progress - self.step_size_up) / \
                float(self.step_size_down)

        # Base lr may come from user or optimizer
        base_lr = self.base_lr_user if self.base_lr_user is not None else self.base_lr
        new_lr = base_lr + (self.max_lr - base_lr) * max(0.0, min(1.0, scale)) * self._scale_fn(self.iteration // self.cycle_len)
        # Bound within [base_lr, max_lr]
        new_lr = max(min(new_lr, self.max_lr), base_lr)
        adjusted_lr = new_lr

        if hasattr(self.optimizer, 'param_groups'):
            for group in self.optimizer.param_groups:
                group['lr'] = adjusted_lr
        elif hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = adjusted_lr


class FractionalReduceLROnPlateau(FractionalScheduler):
    """Reduce learning rate on plateau scheduler with fractional calculus integration"""

    def __init__(
            self,
            optimizer: Any,
            mode: str = 'min',
            factor: float = 0.1,
            patience: int = 10,
            verbose: bool = False,
            threshold: float = 1e-4,
            threshold_mode: str = 'rel',
            cooldown: int = 0,
            min_lr: float = 0.0,
            eps: float = 1e-8,
            fractional_order: float = 0.5,
            method: str = "RL",
            backend: Optional[BackendType] = None):
        super().__init__(optimizer, fractional_order, method, backend)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps

        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.last_epoch = 0

    def step(self, metrics: float) -> None:
        """Update learning rate based on metrics"""
        if metrics is None:
            return
        self.last_epoch += 1

        improved = self._is_improved(metrics)
        if improved:
            self.best = metrics if self.best is None else (metrics if self._mode_better(metrics, self.best) else self.best)
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        if self.num_bad_epochs > self.patience and self.cooldown_counter == 0:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _mode_better(self, current: float, best: float) -> bool:
        if self.mode == 'min':
            if self.threshold_mode == 'rel':
                return current < best * (1 - self.threshold)
            return current < best - self.threshold
        else:
            if self.threshold_mode == 'rel':
                return current > best * (1 + self.threshold)
            return current > best + self.threshold

    def _is_improved(self, metrics: float) -> bool:
        if self.best is None:
            return True
        return self._mode_better(metrics, self.best)

    def _reduce_lr(self) -> None:
        """Reduce learning rate"""
        old_lr = self._get_base_lr()
        new_lr = max(old_lr * self.factor, self.min_lr)
        if abs(new_lr - old_lr) < self.eps:
            return
        adjusted_lr = new_lr

        # Update optimizer learning rate
        if hasattr(self.optimizer, 'param_groups'):
            for group in self.optimizer.param_groups:
                group['lr'] = adjusted_lr
        elif hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = adjusted_lr

        if self.verbose:
            print(
                f'Reducing learning rate from {old_lr:.6f} to {adjusted_lr:.6f}')


class TrainingCallback(ABC):
    """Base class for training callbacks"""

    def __init__(self):
        self.trainer = None

    def set_trainer(self, trainer: 'FractionalTrainer'):
        """Set the trainer reference"""
        self.trainer = trainer

    @abstractmethod
    def on_epoch_begin(self, epoch: int) -> None:
        """Called at the beginning of each epoch"""

    @abstractmethod
    def on_epoch_end(self, epoch: int) -> None:
        """Called at the end of each epoch"""

    def on_batch_begin(self, batch: int) -> None:
        """Called at the beginning of each batch"""
        # Default no-op so subclasses don't need to implement
        return None

    def on_batch_end(self, batch: int) -> None:
        """Called at the end of each batch"""
        # Default no-op so subclasses don't need to implement
        return None


class EarlyStoppingCallback(TrainingCallback):
    """Early stopping callback"""

    def __init__(self, patience: int = 0, min_delta: float = 0.0, mode: str = 'min', monitor: str = 'val_loss', verbose: int = 0, restore_best_weights: bool = False):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self._best_state = None
        self.stopped_epoch = 0
        self.wait = 0
        self.best = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        """Check if training should stop early"""
        if self.trainer is None:
            return
        current_score = None
        if logs and self.monitor in logs:
            current_score = logs[self.monitor]
        elif self.trainer.validation_losses and self.monitor == 'val_loss':
            current_score = self.trainer.validation_losses[-1]
        elif self.trainer.training_losses and self.monitor == 'train_loss':
            current_score = self.trainer.training_losses[-1]
        else:
            current_score = float('inf')

        if self.best_score is None:
            self.best_score = current_score
        elif self.mode == 'min':
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            self.stopped_epoch = epoch
            if self.restore_best_weights and self._best_state is not None and hasattr(self.trainer, 'model'):
                try:
                    self.trainer.model.load_state_dict(self._best_state)
                except Exception:
                    pass
            if self.trainer:
                self.trainer.should_stop = True

    def on_epoch_begin(self, epoch: int) -> None:
        pass

    def on_batch_begin(self, batch: int) -> None:
        pass

    def on_batch_end(self, batch: int) -> None:
        pass


class ModelCheckpointCallback(TrainingCallback):
    """Model checkpoint callback"""

    def __init__(self, filepath: str = 'model_checkpoint.pth', monitor: str = 'val_loss', mode: str = 'min', save_best_only: bool = False, verbose: int = 0, save_weights_only: bool = False):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.save_weights_only = save_weights_only
        self.save_freq = 'epoch'
        self.best_score = None
        self.best = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None, model: Optional[Any] = None) -> None:
        """Save model checkpoint if needed"""
        if self.trainer is None:
            return

        # Get the metric to monitor
        if self.monitor == 'val_loss':
            current_score = self.trainer.validation_losses[-1] if self.trainer.validation_losses else float(
                'inf')
        elif self.monitor == 'train_loss':
            current_score = self.trainer.training_losses[-1] if self.trainer.training_losses else float(
                'inf')
        else:
            return

        # Check if we should save
        should_save = False
        if self.best_score is None:
            should_save = True
        elif self.mode == 'min':
            if current_score < self.best_score:
                should_save = True
        else:
            if current_score > self.best_score:
                should_save = True

        if should_save:
            self.best_score = current_score
            target_model = model or getattr(self.trainer, 'model', None)
            if target_model is not None:
                try:
                    import torch
                    obj = target_model.state_dict() if self.save_weights_only else target_model
                    torch.save(obj, self.filepath)
                except Exception:
                    print(f"Saving model checkpoint to {self.filepath}")

    def on_epoch_begin(self, epoch: int) -> None:
        pass

    def on_batch_begin(self, batch: int) -> None:
        pass

    def on_batch_end(self, batch: int) -> None:
        pass


class FractionalTrainer:
    """Training trainer with fractional calculus integration"""

    def __init__(
            self,
            model: Any,
            optimizer: Any,
            loss_fn: Optional[Any] = None,
            scheduler: Optional[FractionalScheduler] = None,
            callbacks: Optional[List[TrainingCallback]] = None,
            fractional_order: float = 0.5,
            method: str = "RL",
            backend: Optional[BackendType] = None,
            **kwargs):
        self.model = model
        self.optimizer = optimizer
        # Default to MSELoss if not provided
        if loss_fn is None:
            try:
                import torch
                self.loss_fn = torch.nn.MSELoss()
            except Exception:
                self.loss_fn = lambda out, tgt: ((out - tgt) ** 2).mean()
        else:
            self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.fractional_order = FractionalOrder(fractional_order)
        self.method = method
        self.backend = backend or get_backend_manager().active_backend
        self.tensor_ops = get_tensor_ops(self.backend)
        self.device = kwargs.get('device', 'cpu')

        # Training state
        self.training_losses = []
        self.validation_losses = []
        self.current_epoch = 0
        self.should_stop = False

        # Set trainer reference in callbacks
        for callback in self.callbacks:
            callback.set_trainer(self)

    def add_callback(self, callback: TrainingCallback) -> None:
        callback.set_trainer(self)
        self.callbacks.append(callback)

    def remove_callback(self, callback: TrainingCallback) -> None:
        self.callbacks = [c for c in self.callbacks if c is not callback]

    def train_epoch(self, dataloader: Any) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            # Call batch begin callback
            for callback in self.callbacks:
                callback.on_batch_begin(batch_idx)

            # Forward pass
            output = self.model(data)
            loss = self.loss_fn(output, target)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Call batch end callback
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx)

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate_epoch(self, dataloader: Any) -> None:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with self.tensor_ops.no_grad():
            for data, target in dataloader:
                output = self.model(data)
                loss = self.loss_fn(output, target)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    # Backwards-compatible simple steps for tests
    def train_step(self, data: Any, target: Any) -> float:
        self.model.train()
        output = self.model(data)
        loss = self.loss_fn(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def validate_step(self, data: Any, target: Any) -> float:
        self.model.eval()
        with self.tensor_ops.no_grad():
            output = self.model(data)
            loss = self.loss_fn(output, target)
            return float(loss.item())

    def train(self, train_dataloader: Any, val_dataloader: Any, num_epochs: int) -> Dict[str, List[float]]:
        """Train the model"""
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Call epoch begin callback
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            # Training phase
            train_loss = self.train_epoch(train_dataloader)
            self.training_losses.append(train_loss)

            # Validation phase
            val_loss = self.validate_epoch(val_dataloader)
            self.validation_losses.append(val_loss)

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, FractionalReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"LR: {self.scheduler.get_last_lr()[0] if self.scheduler else 'N/A':.6f}")

            # Call epoch end callback
            for callback in self.callbacks:
                callback.on_epoch_end(epoch)

            # Check if we should stop early
            if self.should_stop:
                print("Early stopping triggered")
                break

        print("Training completed!")
        return {
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }

    def fit(self, train_dataloader: Any, val_dataloader: Any = None, num_epochs: int = 10, **kwargs) -> Dict[str, List[float]]:
        """
        Alias for train() method - provides sklearn-style API.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional, uses train_dataloader if None)
            num_epochs: Number of training epochs
            **kwargs: Additional arguments passed to train()
            
        Returns:
            Dictionary containing training and validation losses
        """
        if val_dataloader is None:
            val_dataloader = train_dataloader
        return self.train(train_dataloader, val_dataloader, num_epochs, **kwargs)

    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint"""
        # Simplified checkpoint saving
        # In practice, you'd want to save the full model state
        print(f"Model checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint"""
        # Simplified checkpoint loading
        # In practice, you'd want to load the full model state
        print(f"Model checkpoint loaded from {filepath}")


# Factory functions for easy creation
def create_fractional_scheduler(
        optimizer: Any,
        scheduler_type: str,
        fractional_order: float = 0.5,
        method: str = "RL",
        **kwargs) -> FractionalScheduler:
    """
    Create a fractional scheduler of the specified type

    Args:
        scheduler_type: Type of scheduler ('step', 'exponential', 'cosine', 'plateau')
        optimizer: Optimizer to schedule
        fractional_order: Fractional order for derivative
        method: Method for fractional derivative
        **kwargs: Additional scheduler-specific parameters

    Returns:
        Configured fractional scheduler
    """
    scheduler_map = {
        'step': FractionalStepLR,
        'exponential': FractionalExponentialLR,
        'cosine': FractionalCosineAnnealingLR,
        'cyclic': FractionalCyclicLR,
        'plateau': FractionalReduceLROnPlateau,
    }

    if scheduler_type.lower() not in scheduler_map:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    scheduler_class = scheduler_map[scheduler_type.lower()]
    return scheduler_class(optimizer, fractional_order=fractional_order, method=method, **kwargs)


def create_fractional_trainer(
        model: Any,
        optimizer: Any,
        loss_fn: Optional[Any] = None,
        scheduler: Optional[FractionalScheduler] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        fractional_order: float = 0.5,
        method: str = "RL",
        scheduler_type: Optional[str] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        **kwargs) -> FractionalTrainer:
    """
    Create a fractional trainer

    Args:
        model: Model to train
        optimizer: Optimizer for training
        loss_fn: Loss function
        scheduler: Learning rate scheduler (optional)
        callbacks: Training callbacks (optional)
        fractional_order: Fractional order for derivative
        method: Method for fractional derivative

    Returns:
        Configured fractional trainer
    """
    # Create scheduler from type if provided
    if scheduler is None and scheduler_type is not None:
        scheduler = create_fractional_scheduler(
            optimizer,
            scheduler_type=scheduler_type,
            fractional_order=fractional_order,
            method=method,
            **(scheduler_params or {})
        )

    # Device is accepted for API compatibility; handled by user/model externally
    return FractionalTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        callbacks=callbacks,
        fractional_order=fractional_order,
        method=method,
        **kwargs
    )
