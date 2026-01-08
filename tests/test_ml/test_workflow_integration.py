import torch
import unittest
from hpfracc.ml.workflow import ModelValidator, QualityThreshold, QualityMetric

class TestWorkflowIntegration(unittest.TestCase):
    def test_validator_uses_trainer(self):
        """Test that ModelValidator works with the new FractionalTrainer integration."""
        model = torch.nn.Linear(10, 1)
        validator = ModelValidator()
        
        # Create dummy data
        x = torch.randn(20, 10)
        y = torch.randn(20, 1)
        
        # Run validation
        results = validator.validate_model(model, x, y)
        
        print("\nValidation Results:", results)
        
        self.assertIn('loss', results['metrics'])
        self.assertIn('accuracy', results['metrics']) # R2 score for regression
        self.assertIsInstance(results['metrics']['loss'], float)
        self.assertIn('validation_passed', results)

if __name__ == '__main__':
    unittest.main()
