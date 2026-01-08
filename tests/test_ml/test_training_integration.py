import torch
import torch.nn as nn
import unittest
from unittest.mock import MagicMock
import sys

# Mock Optuna if not available
try:
    import optuna
except ImportError:
    optuna = None

from hpfracc.ml.training import FractionalTrainer, OptunaPruningCallback
from hpfracc.ml.optimized_optimizers import OptimizedFractionalSGD
from hpfracc.ml.variance_aware_training import create_variance_aware_trainer

class TestTrainingIntegration(unittest.TestCase):
    
    def setUp(self):
        self.model = nn.Linear(10, 1)
        self.optimizer = OptimizedFractionalSGD(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        
    def test_optuna_callback(self):
        """Test OptunaPruningCallback integration."""
        if optuna is None:
            print("Optuna not installed, skipping test.")
            return

        # Mock trial
        trial = MagicMock()
        trial.should_prune.return_value = False
        
        callback = OptunaPruningCallback(trial, monitor='val_loss')
        trainer = FractionalTrainer(
            self.model, self.optimizer, self.loss_fn, callbacks=[callback]
        )
        
        # Simulate validation loss
        trainer.validation_losses = [0.5]
        
        # Call on_epoch_end
        callback.on_epoch_end(0, logs={'val_loss': 0.5})
        
        # Check if report was called
        trial.report.assert_called_with(0.5, 0)
        
        # Test pruning
        trial.should_prune.return_value = True
        with self.assertRaises(optuna.exceptions.TrialPruned):
            callback.on_epoch_end(1, logs={'val_loss': 0.6})
            
    def test_variance_aware_trainer_integration(self):
        """Test VarianceAwareTrainer with new Optimizers."""
        trainer = create_variance_aware_trainer(
            self.model, self.optimizer, self.loss_fn
        )
        
        # Dummy data
        data = torch.randn(5, 10)
        target = torch.randn(5, 1)
        dataloader = [(data, target)]
        
        # Train one epoch
        results = trainer.train(dataloader, num_epochs=1)
        
        self.assertIn('losses', results)
        self.assertIn('variance_history', results)
        self.assertTrue(len(results['losses']) > 0)

if __name__ == "__main__":
    unittest.main()
