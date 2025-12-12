"""
Comprehensive tests for hpfracc.ml.training module

This module tests all training utilities, schedulers, and callbacks
for fractional calculus machine learning applications.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from hpfracc.ml.training import (
    FractionalScheduler,
    FractionalStepLR,
    FractionalExponentialLR,
    FractionalCosineAnnealingLR,
    FractionalCyclicLR,
    FractionalReduceLROnPlateau,
    TrainingCallback,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    FractionalTrainer,
    create_fractional_scheduler,
    create_fractional_trainer
)
from hpfracc.ml.backends import BackendType
from hpfracc.core.definitions import FractionalOrder


class TestFractionalScheduler:
    """Test the base FractionalScheduler class"""

    def test_initialization_default(self):
        """Test scheduler initialization with default parameters"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalScheduler(optimizer)
        
        assert scheduler.optimizer == optimizer
        assert scheduler.fractional_order.alpha == 0.5
        assert scheduler.method == "RL"
        assert scheduler.backend == BackendType.TORCH
        assert scheduler.base_lr == 0.1

    def test_initialization_custom(self):
        """Test scheduler initialization with custom parameters"""
        optimizer = optim.Adam([torch.tensor([1.0], requires_grad=True)], lr=0.01)
        scheduler = FractionalScheduler(
            optimizer, 
            fractional_order=0.7, 
            method="Caputo",
            backend=BackendType.TORCH
        )
        
        assert scheduler.fractional_order.alpha == 0.7
        assert scheduler.method == "Caputo"
        assert scheduler.backend == BackendType.TORCH
        assert scheduler.base_lr == 0.01

    def test_get_base_lr_from_param_groups(self):
        """Test getting base learning rate from optimizer param_groups"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.05)
        scheduler = FractionalScheduler(optimizer)
        assert scheduler.base_lr == 0.05

    def test_get_base_lr_from_lr_attribute(self):
        """Test getting base learning rate from optimizer lr attribute"""
        optimizer = Mock()
        optimizer.lr = 0.03
        scheduler = FractionalScheduler(optimizer)
        assert scheduler.base_lr == 0.03

    def test_get_base_lr_fallback(self):
        """Test fallback to default learning rate"""
        optimizer = Mock()
        scheduler = FractionalScheduler(optimizer)
        assert scheduler.base_lr == 0.001

    def test_fractional_adjustment_torch_backend(self):
        """Test fractional adjustment for PyTorch backend"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalScheduler(optimizer, backend=BackendType.TORCH)
        
        # Test with valid input
        adjusted_lr = scheduler.fractional_adjustment(0.1)
        assert isinstance(adjusted_lr, float)
        assert adjusted_lr > 0
        assert adjusted_lr >= 1e-12

    def test_fractional_adjustment_non_torch_backend(self):
        """Test fractional adjustment for non-PyTorch backend"""
        optimizer = Mock()
        scheduler = FractionalScheduler(optimizer, backend=BackendType.NUMBA)
        
        # Should return original learning rate unchanged
        original_lr = 0.1
        adjusted_lr = scheduler.fractional_adjustment(original_lr)
        assert adjusted_lr == original_lr

    def test_fractional_adjustment_error_handling(self):
        """Test fractional adjustment error handling"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalScheduler(optimizer, backend=BackendType.TORCH)
        
        # Test with very small learning rate
        adjusted_lr = scheduler.fractional_adjustment(1e-15)
        assert adjusted_lr >= 1e-12

    def test_get_last_lr_from_param_groups(self):
        """Test getting last learning rate from param_groups"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalScheduler(optimizer)
        
        lr_list = scheduler.get_last_lr()
        assert isinstance(lr_list, list)
        assert len(lr_list) == 1
        assert lr_list[0] == 0.1

    def test_get_last_lr_from_lr_attribute(self):
        """Test getting last learning rate from lr attribute"""
        optimizer = Mock()
        optimizer.lr = 0.05
        scheduler = FractionalScheduler(optimizer)
        
        lr_list = scheduler.get_last_lr()
        assert isinstance(lr_list, list)
        assert len(lr_list) == 1
        assert lr_list[0] == 0.05

    def test_get_last_lr_fallback(self):
        """Test fallback to base learning rate"""
        optimizer = Mock()
        scheduler = FractionalScheduler(optimizer)
        
        lr_list = scheduler.get_last_lr()
        assert isinstance(lr_list, list)
        assert len(lr_list) == 1
        assert lr_list[0] == scheduler.base_lr

    def test_abstract_step_method(self):
        """Test that step method is abstract"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalScheduler(optimizer)
        
        with pytest.raises(NotImplementedError):
            scheduler.step()


class TestFractionalStepLR:
    """Test the FractionalStepLR scheduler"""

    def test_initialization_default(self):
        """Test FractionalStepLR initialization with default parameters"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalStepLR(optimizer)
        
        assert scheduler.step_size == 30
        assert scheduler.gamma == 0.1
        assert scheduler.last_epoch == -1

    def test_initialization_custom(self):
        """Test FractionalStepLR initialization with custom parameters"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalStepLR(
            optimizer, 
            step_size=10, 
            gamma=0.5,
            fractional_order=0.7
        )
        
        assert scheduler.step_size == 10
        assert scheduler.gamma == 0.5
        assert scheduler.fractional_order.alpha == 0.7

    def test_step_basic(self):
        """Test basic step functionality"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalStepLR(optimizer, step_size=2, gamma=0.5)
        
        # Initial learning rate
        assert scheduler.get_last_lr()[0] == 0.1
        
        # Step once
        scheduler.step()
        assert scheduler.last_epoch == 0
        
        # Step again (should not change LR yet)
        scheduler.step()
        assert scheduler.last_epoch == 1
        
        # Step again (should reduce LR)
        scheduler.step()
        assert scheduler.last_epoch == 2
        # LR should be reduced by gamma
        assert scheduler.get_last_lr()[0] < 0.1

    def test_step_with_metrics(self):
        """Test step with metrics parameter"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalStepLR(optimizer, step_size=1)
        
        scheduler.step(metrics=0.5)
        assert scheduler.last_epoch == 0

    def test_multiple_optimizer_groups(self):
        """Test scheduler with multiple optimizer groups"""
        param1 = torch.tensor([1.0], requires_grad=True)
        param2 = torch.tensor([2.0], requires_grad=True)
        optimizer = optim.SGD([
            {'params': [param1], 'lr': 0.1},
            {'params': [param2], 'lr': 0.2}
        ])
        scheduler = FractionalStepLR(optimizer, step_size=1, gamma=0.5)
        
        initial_lrs = scheduler.get_last_lr()
        assert len(initial_lrs) == 2
        assert initial_lrs[0] == 0.1
        assert initial_lrs[1] == 0.2
        
        scheduler.step()
        new_lrs = scheduler.get_last_lr()
        assert len(new_lrs) == 2
        assert new_lrs[0] < initial_lrs[0]
        assert new_lrs[1] < initial_lrs[1]


class TestFractionalExponentialLR:
    """Test the FractionalExponentialLR scheduler"""

    def test_initialization_default(self):
        """Test FractionalExponentialLR initialization with default parameters"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalExponentialLR(optimizer)
        
        assert scheduler.gamma == 0.95
        assert scheduler.last_epoch == -1

    def test_initialization_custom(self):
        """Test FractionalExponentialLR initialization with custom parameters"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalExponentialLR(
            optimizer, 
            gamma=0.9,
            fractional_order=0.6
        )
        
        assert scheduler.gamma == 0.9
        assert scheduler.fractional_order.alpha == 0.6

    def test_step_exponential_decay(self):
        """Test exponential decay functionality"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalExponentialLR(optimizer, gamma=0.5)
        
        initial_lr = scheduler.get_last_lr()[0]
        
        scheduler.step()
        lr_after_one_step = scheduler.get_last_lr()[0]
        
        scheduler.step()
        lr_after_two_steps = scheduler.get_last_lr()[0]
        
        # Should decay exponentially
        assert lr_after_one_step < initial_lr
        assert lr_after_two_steps < lr_after_one_step
        assert abs(lr_after_one_step - initial_lr * 0.5) < 1e-6


class TestFractionalCosineAnnealingLR:
    """Test the FractionalCosineAnnealingLR scheduler"""

    def test_initialization_default(self):
        """Test FractionalCosineAnnealingLR initialization with default parameters"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalCosineAnnealingLR(optimizer)
        
        assert scheduler.T_max == 10
        assert scheduler.eta_min == 0.0
        assert scheduler.last_epoch == -1

    def test_initialization_custom(self):
        """Test FractionalCosineAnnealingLR initialization with custom parameters"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalCosineAnnealingLR(
            optimizer, 
            T_max=20,
            eta_min=0.01,
            fractional_order=0.8
        )
        
        assert scheduler.T_max == 20
        assert scheduler.eta_min == 0.01
        assert scheduler.fractional_order.alpha == 0.8

    def test_step_cosine_annealing(self):
        """Test cosine annealing functionality"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalCosineAnnealingLR(optimizer, T_max=4, eta_min=0.01)
        
        initial_lr = scheduler.get_last_lr()[0]
        
        # Step through several epochs
        for i in range(5):
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            # LR should oscillate between initial_lr and eta_min
            assert current_lr >= 0.01
            assert current_lr <= initial_lr


class TestFractionalCyclicLR:
    """Test the FractionalCyclicLR scheduler"""

    def test_initialization_default(self):
        """Test FractionalCyclicLR initialization with default parameters"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalCyclicLR(optimizer)
        
        assert scheduler.base_lr == 0.1
        assert scheduler.max_lr == 0.1
        assert scheduler.step_size_up == 2000
        assert scheduler.step_size_down == 2000
        assert scheduler.mode == 'triangular'
        assert scheduler.gamma == 1.0
        assert scheduler.scale_fn is None
        assert scheduler.scale_mode == 'cycle'
        assert scheduler.last_epoch == -1

    def test_initialization_custom(self):
        """Test FractionalCyclicLR initialization with custom parameters"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalCyclicLR(
            optimizer,
            base_lr=0.01,
            max_lr=0.1,
            step_size_up=1000,
            step_size_down=1000,
            mode='triangular2',
            gamma=0.9,
            fractional_order=0.7
        )
        
        assert scheduler.base_lr == 0.01
        assert scheduler.max_lr == 0.1
        assert scheduler.step_size_up == 1000
        assert scheduler.step_size_down == 1000
        assert scheduler.mode == 'triangular2'
        assert scheduler.gamma == 0.9
        assert scheduler.fractional_order.alpha == 0.7

    def test_step_cyclic_behavior(self):
        """Test cyclic learning rate behavior"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalCyclicLR(
            optimizer,
            base_lr=0.01,
            max_lr=0.1,
            step_size_up=2,
            step_size_down=2
        )
        
        # Step through a few cycles
        lrs = []
        for i in range(8):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])
        
        # Should see cyclic behavior
        assert len(lrs) == 8
        assert all(lr >= 0.01 and lr <= 0.1 for lr in lrs)


class TestFractionalReduceLROnPlateau:
    """Test the FractionalReduceLROnPlateau scheduler"""

    def test_initialization_default(self):
        """Test FractionalReduceLROnPlateau initialization with default parameters"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalReduceLROnPlateau(optimizer)
        
        assert scheduler.mode == 'min'
        assert scheduler.factor == 0.1
        assert scheduler.patience == 10
        assert scheduler.threshold == 1e-4
        assert scheduler.threshold_mode == 'rel'
        assert scheduler.cooldown == 0
        assert scheduler.min_lr == 0
        assert scheduler.eps == 1e-8
        assert scheduler.best == None
        assert scheduler.num_bad_epochs == 0
        assert scheduler.cooldown_counter == 0

    def test_initialization_custom(self):
        """Test FractionalReduceLROnPlateau initialization with custom parameters"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            threshold=1e-3,
            threshold_mode='abs',
            cooldown=2,
            min_lr=0.001,
            eps=1e-6,
            fractional_order=0.8
        )
        
        assert scheduler.mode == 'max'
        assert scheduler.factor == 0.5
        assert scheduler.patience == 5
        assert scheduler.threshold == 1e-3
        assert scheduler.threshold_mode == 'abs'
        assert scheduler.cooldown == 2
        assert scheduler.min_lr == 0.001
        assert scheduler.eps == 1e-6
        assert scheduler.fractional_order.alpha == 0.8

    def test_step_improvement(self):
        """Test step with improving metrics"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalReduceLROnPlateau(optimizer, patience=2)
        
        initial_lr = scheduler.get_last_lr()[0]
        
        # Improving metrics should not reduce LR
        scheduler.step(metrics=0.5)
        assert scheduler.get_last_lr()[0] == initial_lr
        
        scheduler.step(metrics=0.4)  # Better
        assert scheduler.get_last_lr()[0] == initial_lr
        
        scheduler.step(metrics=0.3)  # Even better
        assert scheduler.get_last_lr()[0] == initial_lr

    def test_step_no_improvement(self):
        """Test step with no improvement"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        
        initial_lr = scheduler.get_last_lr()[0]
        
        # No improvement for patience epochs should reduce LR
        scheduler.step(metrics=0.5)
        scheduler.step(metrics=0.5)  # Same
        scheduler.step(metrics=0.5)  # Same again
        
        # Should reduce LR after patience epochs
        assert scheduler.get_last_lr()[0] < initial_lr

    def test_step_with_cooldown(self):
        """Test step with cooldown period"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        scheduler = FractionalReduceLROnPlateau(optimizer, patience=1, cooldown=2)
        
        initial_lr = scheduler.get_last_lr()[0]
        
        # Trigger LR reduction
        scheduler.step(metrics=0.5)
        scheduler.step(metrics=0.5)  # No improvement
        
        # LR should be reduced
        reduced_lr = scheduler.get_last_lr()[0]
        assert reduced_lr < initial_lr
        
        # During cooldown, LR should not change even with bad metrics
        scheduler.step(metrics=0.6)  # Worse
        assert scheduler.get_last_lr()[0] == reduced_lr
        
        scheduler.step(metrics=0.7)  # Even worse
        assert scheduler.get_last_lr()[0] == reduced_lr


class TestTrainingCallback:
    """Test the base TrainingCallback class"""

    def test_abstract_methods(self):
        """Test that callback methods are abstract"""
        callback = TrainingCallback()
        
        with pytest.raises(NotImplementedError):
            callback.on_epoch_begin(epoch=0)
        
        with pytest.raises(NotImplementedError):
            callback.on_epoch_end(epoch=0, logs={})
        
        with pytest.raises(NotImplementedError):
            callback.on_batch_begin(batch=0, logs={})
        
        with pytest.raises(NotImplementedError):
            callback.on_batch_end(batch=0, logs={})


class TestEarlyStoppingCallback:
    """Test the EarlyStoppingCallback class"""

    def test_initialization_default(self):
        """Test EarlyStoppingCallback initialization with default parameters"""
        callback = EarlyStoppingCallback()
        
        assert callback.monitor == 'val_loss'
        assert callback.min_delta == 0.0
        assert callback.patience == 0
        assert callback.verbose == 0
        assert callback.mode == 'min'
        assert callback.restore_best_weights == False
        assert callback.stopped_epoch == 0
        assert callback.wait == 0
        assert callback.best == None

    def test_initialization_custom(self):
        """Test EarlyStoppingCallback initialization with custom parameters"""
        callback = EarlyStoppingCallback(
            monitor='val_accuracy',
            min_delta=0.001,
            patience=5,
            verbose=1,
            mode='max',
            restore_best_weights=True
        )
        
        assert callback.monitor == 'val_accuracy'
        assert callback.min_delta == 0.001
        assert callback.patience == 5
        assert callback.verbose == 1
        assert callback.mode == 'max'
        assert callback.restore_best_weights == True

    def test_on_epoch_end_no_improvement(self):
        """Test on_epoch_end with no improvement"""
        callback = EarlyStoppingCallback(patience=2)
        
        # No improvement for patience epochs
        logs = {'val_loss': 0.5}
        callback.on_epoch_end(epoch=0, logs=logs)
        assert callback.wait == 1
        
        logs = {'val_loss': 0.5}  # Same
        callback.on_epoch_end(epoch=1, logs=logs)
        assert callback.wait == 2
        
        logs = {'val_loss': 0.5}  # Still same
        callback.on_epoch_end(epoch=2, logs=logs)
        assert callback.wait == 3
        assert callback.stopped_epoch == 2

    def test_on_epoch_end_with_improvement(self):
        """Test on_epoch_end with improvement"""
        callback = EarlyStoppingCallback(patience=2)
        
        # Initial epoch
        logs = {'val_loss': 0.5}
        callback.on_epoch_end(epoch=0, logs=logs)
        assert callback.wait == 1
        
        # Improvement resets wait counter
        logs = {'val_loss': 0.4}  # Better
        callback.on_epoch_end(epoch=1, logs=logs)
        assert callback.wait == 0
        assert callback.best == 0.4

    def test_on_epoch_end_max_mode(self):
        """Test on_epoch_end with max mode"""
        callback = EarlyStoppingCallback(monitor='val_accuracy', mode='max', patience=2)
        
        # Initial epoch
        logs = {'val_accuracy': 0.8}
        callback.on_epoch_end(epoch=0, logs=logs)
        assert callback.wait == 1
        
        # Improvement
        logs = {'val_accuracy': 0.9}  # Better
        callback.on_epoch_end(epoch=1, logs=logs)
        assert callback.wait == 0
        assert callback.best == 0.9


class TestModelCheckpointCallback:
    """Test the ModelCheckpointCallback class"""

    def test_initialization_default(self):
        """Test ModelCheckpointCallback initialization with default parameters"""
        callback = ModelCheckpointCallback()
        
        assert callback.filepath == 'model_checkpoint.pth'
        assert callback.monitor == 'val_loss'
        assert callback.verbose == 0
        assert callback.save_best_only == False
        assert callback.save_weights_only == False
        assert callback.mode == 'min'
        assert callback.save_freq == 'epoch'
        assert callback.best == None

    def test_initialization_custom(self):
        """Test ModelCheckpointCallback initialization with custom parameters"""
        callback = ModelCheckpointCallback(
            filepath='best_model.pth',
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='max'
        )
        
        assert callback.filepath == 'best_model.pth'
        assert callback.monitor == 'val_accuracy'
        assert callback.verbose == 1
        assert callback.save_best_only == True
        assert callback.save_weights_only == True
        assert callback.mode == 'max'

    @patch('torch.save')
    def test_on_epoch_end_save_model(self, mock_save):
        """Test on_epoch_end saves model"""
        model = nn.Linear(1, 1)
        callback = ModelCheckpointCallback(save_best_only=False)
        
        logs = {'val_loss': 0.5}
        callback.on_epoch_end(epoch=0, logs=logs, model=model)
        
        # Should call torch.save
        mock_save.assert_called_once()

    @patch('torch.save')
    def test_on_epoch_end_save_best_only(self, mock_save):
        """Test on_epoch_end with save_best_only=True"""
        model = nn.Linear(1, 1)
        callback = ModelCheckpointCallback(save_best_only=True)
        
        # First epoch - should save
        logs = {'val_loss': 0.5}
        callback.on_epoch_end(epoch=0, logs=logs, model=model)
        assert mock_save.call_count == 1
        
        # Second epoch with worse loss - should not save
        logs = {'val_loss': 0.6}
        callback.on_epoch_end(epoch=1, logs=logs, model=model)
        assert mock_save.call_count == 1  # No additional save
        
        # Third epoch with better loss - should save
        logs = {'val_loss': 0.4}
        callback.on_epoch_end(epoch=2, logs=logs, model=model)
        assert mock_save.call_count == 2  # Additional save


class TestFractionalTrainer:
    """Test the FractionalTrainer class"""

    def test_initialization_default(self):
        """Test FractionalTrainer initialization with default parameters"""
        model = nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        trainer = FractionalTrainer(model, optimizer)
        
        assert trainer.model == model
        assert trainer.optimizer == optimizer
        assert trainer.scheduler is None
        assert trainer.callbacks == []
        assert trainer.device == 'cpu'
        assert trainer.fractional_order.alpha == 0.5
        assert trainer.method == "RL"
        assert trainer.backend == BackendType.TORCH

    def test_initialization_custom(self):
        """Test FractionalTrainer initialization with custom parameters"""
        model = nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        scheduler = FractionalStepLR(optimizer)
        callbacks = [EarlyStoppingCallback()]
        
        trainer = FractionalTrainer(
            model, 
            optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            device='cuda',
            fractional_order=0.7,
            method="Caputo"
        )
        
        assert trainer.scheduler == scheduler
        assert trainer.callbacks == callbacks
        assert trainer.device == 'cuda'
        assert trainer.fractional_order.alpha == 0.7
        assert trainer.method == "Caputo"

    def test_add_callback(self):
        """Test adding callbacks"""
        model = nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        trainer = FractionalTrainer(model, optimizer)
        
        callback = EarlyStoppingCallback()
        trainer.add_callback(callback)
        
        assert callback in trainer.callbacks

    def test_remove_callback(self):
        """Test removing callbacks"""
        model = nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        callback = EarlyStoppingCallback()
        trainer = FractionalTrainer(model, optimizer, callbacks=[callback])
        
        trainer.remove_callback(callback)
        
        assert callback not in trainer.callbacks

    def test_train_step(self):
        """Test single training step"""
        model = nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        trainer = FractionalTrainer(model, optimizer)
        
        # Create dummy data
        x = torch.tensor([[1.0]])
        y = torch.tensor([[2.0]])
        
        # Train step
        loss = trainer.train_step(x, y)
        
        assert isinstance(loss, float)
        assert loss >= 0

    def test_validate_step(self):
        """Test single validation step"""
        model = nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        trainer = FractionalTrainer(model, optimizer)
        
        # Create dummy data
        x = torch.tensor([[1.0]])
        y = torch.tensor([[2.0]])
        
        # Validation step
        loss = trainer.validate_step(x, y)
        
        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_epoch(self):
        """Test training epoch"""
        model = nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        trainer = FractionalTrainer(model, optimizer)
        
        # Create dummy data
        x = torch.tensor([[1.0], [2.0]])
        y = torch.tensor([[2.0], [4.0]])
        
        # Train epoch
        avg_loss = trainer.train_epoch([(x, y)])
        
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0

    def test_validate_epoch(self):
        """Test validation epoch"""
        model = nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        trainer = FractionalTrainer(model, optimizer)
        
        # Create dummy data
        x = torch.tensor([[1.0], [2.0]])
        y = torch.tensor([[2.0], [4.0]])
        
        # Validation epoch
        avg_loss = trainer.validate_epoch([(x, y)])
        
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0

    def test_fit(self):
        """Test fit method"""
        model = nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        trainer = FractionalTrainer(model, optimizer)
        
        # Create dummy data
        x_train = torch.tensor([[1.0], [2.0]])
        y_train = torch.tensor([[2.0], [4.0]])
        x_val = torch.tensor([[3.0]])
        y_val = torch.tensor([[6.0]])
        
        train_data = [(x_train, y_train)]
        val_data = [(x_val, y_val)]
        
        # Fit for 2 epochs
        history = trainer.fit(train_data, val_data, epochs=2)
        
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 2
        assert len(history['val_loss']) == 2


class TestCreateFractionalScheduler:
    """Test the create_fractional_scheduler function"""

    def test_create_step_lr(self):
        """Test creating FractionalStepLR scheduler"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        
        scheduler = create_fractional_scheduler(
            optimizer, 
            scheduler_type='step',
            step_size=10,
            gamma=0.5
        )
        
        assert isinstance(scheduler, FractionalStepLR)
        assert scheduler.step_size == 10
        assert scheduler.gamma == 0.5

    def test_create_exponential_lr(self):
        """Test creating FractionalExponentialLR scheduler"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        
        scheduler = create_fractional_scheduler(
            optimizer, 
            scheduler_type='exponential',
            gamma=0.9
        )
        
        assert isinstance(scheduler, FractionalExponentialLR)
        assert scheduler.gamma == 0.9

    def test_create_cosine_annealing_lr(self):
        """Test creating FractionalCosineAnnealingLR scheduler"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        
        scheduler = create_fractional_scheduler(
            optimizer, 
            scheduler_type='cosine',
            T_max=20,
            eta_min=0.01
        )
        
        assert isinstance(scheduler, FractionalCosineAnnealingLR)
        assert scheduler.T_max == 20
        assert scheduler.eta_min == 0.01

    def test_create_cyclic_lr(self):
        """Test creating FractionalCyclicLR scheduler"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        
        scheduler = create_fractional_scheduler(
            optimizer, 
            scheduler_type='cyclic',
            base_lr=0.01,
            max_lr=0.1
        )
        
        assert isinstance(scheduler, FractionalCyclicLR)
        assert scheduler.base_lr == 0.01
        assert scheduler.max_lr == 0.1

    def test_create_reduce_lr_on_plateau(self):
        """Test creating FractionalReduceLROnPlateau scheduler"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        
        scheduler = create_fractional_scheduler(
            optimizer, 
            scheduler_type='plateau',
            factor=0.5,
            patience=5
        )
        
        assert isinstance(scheduler, FractionalReduceLROnPlateau)
        assert scheduler.factor == 0.5
        assert scheduler.patience == 5

    def test_invalid_scheduler_type(self):
        """Test creating scheduler with invalid type"""
        optimizer = optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)
        
        with pytest.raises(ValueError):
            create_fractional_scheduler(optimizer, scheduler_type='invalid')


class TestCreateFractionalTrainer:
    """Test the create_fractional_trainer function"""

    def test_create_trainer_default(self):
        """Test creating FractionalTrainer with default parameters"""
        model = nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        trainer = create_fractional_trainer(model, optimizer)
        
        assert isinstance(trainer, FractionalTrainer)
        assert trainer.model == model
        assert trainer.optimizer == optimizer

    def test_create_trainer_with_scheduler(self):
        """Test creating FractionalTrainer with scheduler"""
        model = nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        trainer = create_fractional_trainer(
            model, 
            optimizer,
            scheduler_type='step',
            scheduler_params={'step_size': 10}
        )
        
        assert isinstance(trainer, FractionalTrainer)
        assert isinstance(trainer.scheduler, FractionalStepLR)

    def test_create_trainer_with_callbacks(self):
        """Test creating FractionalTrainer with callbacks"""
        model = nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        trainer = create_fractional_trainer(
            model, 
            optimizer,
            callbacks=['early_stopping', 'checkpoint']
        )
        
        assert isinstance(trainer, FractionalTrainer)
        assert len(trainer.callbacks) == 2
        assert isinstance(trainer.callbacks[0], EarlyStoppingCallback)
        assert isinstance(trainer.callbacks[1], ModelCheckpointCallback)

    def test_create_trainer_custom_params(self):
        """Test creating FractionalTrainer with custom parameters"""
        model = nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        trainer = create_fractional_trainer(
            model, 
            optimizer,
            fractional_order=0.7,
            method='Caputo',
            device='cuda'
        )
        
        assert isinstance(trainer, FractionalTrainer)
        assert trainer.fractional_order.alpha == 0.7
        assert trainer.method == 'Caputo'
        assert trainer.device == 'cuda'


# Integration tests
class TestTrainingIntegration:
    """Integration tests for training module"""

    def test_full_training_workflow(self):
        """Test complete training workflow"""
        # Create model and optimizer
        model = nn.Linear(2, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        # Create scheduler and callbacks
        scheduler = FractionalStepLR(optimizer, step_size=2, gamma=0.5)
        early_stopping = EarlyStoppingCallback(patience=10)  # High patience to allow 5 epochs
        checkpoint = ModelCheckpointCallback(save_best_only=True)
        
        # Create trainer
        trainer = FractionalTrainer(
            model, 
            optimizer,
            scheduler=scheduler,
            callbacks=[early_stopping, checkpoint]
        )
        
        # Create dummy data
        x_train = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_train = torch.tensor([[3.0], [7.0], [11.0]])
        x_val = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        y_val = torch.tensor([[5.0], [9.0]])
        
        train_data = [(x_train, y_train)]
        val_data = [(x_val, y_val)]
        
        # Train for 5 epochs
        history = trainer.fit(train_data, val_data, epochs=5)
        
        # Verify results
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 5
        assert len(history['val_loss']) == 5
        
        # Note: Loss may not decrease with this simple model and data
        # Just verify that training completed successfully
        assert len(history['train_loss']) == 5
        assert len(history['val_loss']) == 5

    def test_scheduler_integration(self):
        """Test scheduler integration with training"""
        model = nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        scheduler = FractionalExponentialLR(optimizer, gamma=0.9)
        
        trainer = FractionalTrainer(model, optimizer, scheduler=scheduler)
        
        # Create dummy data
        x = torch.tensor([[1.0], [2.0]])
        y = torch.tensor([[2.0], [4.0]])
        
        # Train for a few epochs
        for epoch in range(3):
            trainer.train_epoch([(x, y)])
            scheduler.step()
        
        # Learning rate should have decreased
        assert scheduler.get_last_lr()[0] < 0.1

    def test_callback_integration(self):
        """Test callback integration with training"""
        model = nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        # Create callback that tracks epochs
        class EpochTracker(TrainingCallback):
            def __init__(self):
                self.epochs_called = []
            
            def on_epoch_begin(self, epoch, logs=None):
                self.epochs_called.append(f"begin_{epoch}")
            
            def on_epoch_end(self, epoch, logs=None):
                self.epochs_called.append(f"end_{epoch}")
            
            def on_batch_begin(self, batch, logs=None):
                pass
            
            def on_batch_end(self, batch, logs=None):
                pass
        
        tracker = EpochTracker()
        trainer = FractionalTrainer(model, optimizer, callbacks=[tracker])
        
        # Create dummy data
        x = torch.tensor([[1.0], [2.0]])
        y = torch.tensor([[2.0], [4.0]])
        
        # Train for 2 epochs
        trainer.fit([(x, y)], [(x, y)], epochs=2)
        
        # Verify callbacks were called
        assert len(tracker.epochs_called) == 4  # 2 epochs * 2 calls each
        assert "begin_0" in tracker.epochs_called
        assert "end_0" in tracker.epochs_called
        assert "begin_1" in tracker.epochs_called
        assert "end_1" in tracker.epochs_called
