"""
Comprehensive tests for hpfracc.ml.workflow module

This module tests all workflow management classes for transitioning models
from development to production, including validation and quality gates.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
from datetime import datetime

from hpfracc.ml.workflow import (
    QualityMetric,
    QualityThreshold,
    QualityGate,
    ModelValidator,
    DevelopmentWorkflow,
    ProductionWorkflow
)
from hpfracc.ml.registry import ModelRegistry, DeploymentStatus


class TestQualityMetric:
    """Test the QualityMetric enum"""

    def test_quality_metric_values(self):
        """Test quality metric enum values"""
        assert QualityMetric.ACCURACY.value == "accuracy"
        assert QualityMetric.PRECISION.value == "precision"
        assert QualityMetric.RECALL.value == "recall"
        assert QualityMetric.F1_SCORE.value == "f1_score"
        assert QualityMetric.LOSS.value == "loss"
        assert QualityMetric.INFERENCE_TIME.value == "inference_time"
        assert QualityMetric.MEMORY_USAGE.value == "memory_usage"
        assert QualityMetric.MODEL_SIZE.value == "model_size"
        assert QualityMetric.CUSTOM.value == "custom"

    def test_quality_metric_iteration(self):
        """Test iterating over quality metrics"""
        metrics = list(QualityMetric)
        assert len(metrics) == 9
        assert QualityMetric.ACCURACY in metrics
        assert QualityMetric.CUSTOM in metrics


class TestQualityThreshold:
    """Test the QualityThreshold class"""

    def test_initialization_default(self):
        """Test threshold initialization with default parameters"""
        threshold = QualityThreshold(QualityMetric.ACCURACY)
        
        assert threshold.metric == QualityMetric.ACCURACY
        assert threshold.min_value is None
        assert threshold.max_value is None
        assert threshold.target_value is None
        assert threshold.tolerance == 0.05

    def test_initialization_custom(self):
        """Test threshold initialization with custom parameters"""
        threshold = QualityThreshold(
            metric=QualityMetric.LOSS,
            min_value=0.1,
            max_value=0.5,
            target_value=0.3,
            tolerance=0.02
        )
        
        assert threshold.metric == QualityMetric.LOSS
        assert threshold.min_value == 0.1
        assert threshold.max_value == 0.5
        assert threshold.target_value == 0.3
        assert threshold.tolerance == 0.02

    def test_check_threshold_min_value(self):
        """Test threshold check with min_value constraint"""
        threshold = QualityThreshold(QualityMetric.ACCURACY, min_value=0.8)
        
        assert threshold.check_threshold(0.9) == True
        assert threshold.check_threshold(0.8) == True
        assert threshold.check_threshold(0.7) == False

    def test_check_threshold_max_value(self):
        """Test threshold check with max_value constraint"""
        threshold = QualityThreshold(QualityMetric.LOSS, max_value=0.3)
        
        assert threshold.check_threshold(0.2) == True
        assert threshold.check_threshold(0.3) == True
        assert threshold.check_threshold(0.4) == False

    def test_check_threshold_target_value(self):
        """Test threshold check with target_value constraint"""
        threshold = QualityThreshold(QualityMetric.ACCURACY, target_value=0.85, tolerance=0.05)
        
        assert threshold.check_threshold(0.85) == True
        assert threshold.check_threshold(0.82) == True  # Within tolerance
        assert threshold.check_threshold(0.88) == True  # Within tolerance
        assert threshold.check_threshold(0.79) == False  # Outside tolerance
        assert threshold.check_threshold(0.91) == False  # Outside tolerance

    def test_check_threshold_multiple_constraints(self):
        """Test threshold check with multiple constraints"""
        threshold = QualityThreshold(
            QualityMetric.ACCURACY,
            min_value=0.8,
            max_value=0.95,
            target_value=0.9,
            tolerance=0.02
        )
        
        assert threshold.check_threshold(0.9) == True   # Exact target
        assert threshold.check_threshold(0.88) == True  # Within tolerance
        assert threshold.check_threshold(0.92) == True  # Within tolerance
        assert threshold.check_threshold(0.85) == False # Outside tolerance
        assert threshold.check_threshold(0.95) == False # Outside tolerance
        assert threshold.check_threshold(0.75) == False # Below min
        assert threshold.check_threshold(0.98) == False # Above max

    def test_check_threshold_no_constraints(self):
        """Test threshold check with no constraints"""
        threshold = QualityThreshold(QualityMetric.ACCURACY)
        
        # Should always pass with no constraints
        assert threshold.check_threshold(0.0) == True
        assert threshold.check_threshold(1.0) == True
        assert threshold.check_threshold(-1.0) == True


class TestQualityGate:
    """Test the QualityGate class"""

    def test_initialization_default(self):
        """Test quality gate initialization with default parameters"""
        thresholds = [
            QualityThreshold(QualityMetric.ACCURACY, min_value=0.8),
            QualityThreshold(QualityMetric.LOSS, max_value=0.3)
        ]
        
        gate = QualityGate(
            name="Test Gate",
            description="Test quality gate",
            thresholds=thresholds
        )
        
        assert gate.name == "Test Gate"
        assert gate.description == "Test quality gate"
        assert gate.thresholds == thresholds
        assert gate.required == True
        assert gate.weight == 1.0

    def test_initialization_custom(self):
        """Test quality gate initialization with custom parameters"""
        thresholds = [QualityThreshold(QualityMetric.ACCURACY, min_value=0.9)]
        
        gate = QualityGate(
            name="Custom Gate",
            description="Custom quality gate",
            thresholds=thresholds,
            required=False,
            weight=0.5
        )
        
        assert gate.name == "Custom Gate"
        assert gate.description == "Custom quality gate"
        assert gate.thresholds == thresholds
        assert gate.required == False
        assert gate.weight == 0.5

    def test_evaluate_all_passed(self):
        """Test gate evaluation when all thresholds pass"""
        thresholds = [
            QualityThreshold(QualityMetric.ACCURACY, min_value=0.8),
            QualityThreshold(QualityMetric.LOSS, max_value=0.3)
        ]
        
        gate = QualityGate("Test Gate", "Test", thresholds)
        
        metrics = {
            'accuracy': 0.9,
            'loss': 0.2
        }
        
        result = gate.evaluate(metrics)
        
        assert result['gate_name'] == "Test Gate"
        assert result['passed'] == True
        assert result['required'] == True
        assert result['weight'] == 1.0
        
        # Check individual threshold results
        assert result['results']['accuracy']['passed'] == True
        assert result['results']['accuracy']['value'] == 0.9
        assert result['results']['loss']['passed'] == True
        assert result['results']['loss']['value'] == 0.2

    def test_evaluate_some_failed(self):
        """Test gate evaluation when some thresholds fail"""
        thresholds = [
            QualityThreshold(QualityMetric.ACCURACY, min_value=0.8),
            QualityThreshold(QualityMetric.LOSS, max_value=0.3)
        ]
        
        gate = QualityGate("Test Gate", "Test", thresholds)
        
        metrics = {
            'accuracy': 0.9,  # Passes
            'loss': 0.4       # Fails
        }
        
        result = gate.evaluate(metrics)
        
        assert result['passed'] == False
        assert result['results']['accuracy']['passed'] == True
        assert result['results']['loss']['passed'] == False

    def test_evaluate_missing_metric(self):
        """Test gate evaluation with missing metrics"""
        thresholds = [
            QualityThreshold(QualityMetric.ACCURACY, min_value=0.8),
            QualityThreshold(QualityMetric.LOSS, max_value=0.3)
        ]
        
        gate = QualityGate("Test Gate", "Test", thresholds)
        
        metrics = {
            'accuracy': 0.9
            # Missing 'loss' metric
        }
        
        result = gate.evaluate(metrics)
        
        assert result['passed'] == False
        assert result['results']['accuracy']['passed'] == True
        assert result['results']['loss']['passed'] == False
        assert result['results']['loss']['value'] is None
        assert 'error' in result['results']['loss']

    def test_evaluate_empty_metrics(self):
        """Test gate evaluation with empty metrics"""
        thresholds = [
            QualityThreshold(QualityMetric.ACCURACY, min_value=0.8)
        ]
        
        gate = QualityGate("Test Gate", "Test", thresholds)
        
        result = gate.evaluate({})
        
        assert result['passed'] == False
        assert result['results']['accuracy']['passed'] == False
        assert result['results']['accuracy']['value'] is None


class TestModelValidator:
    """Test the ModelValidator class"""

    def test_initialization_default(self):
        """Test validator initialization with default configuration"""
        validator = ModelValidator()
        
        assert validator.config == {}
        assert len(validator.quality_gates) == 3  # Default gates
        assert validator.logger is not None

    def test_initialization_custom(self):
        """Test validator initialization with custom configuration"""
        config = {'test_param': 'test_value'}
        validator = ModelValidator(config)
        
        assert validator.config == config

    def test_setup_default_quality_gates(self):
        """Test setup of default quality gates"""
        validator = ModelValidator()
        gates = validator.quality_gates
        
        assert len(gates) == 3
        
        # Check Basic Performance gate
        basic_gate = gates[0]
        assert basic_gate.name == "Basic Performance"
        assert basic_gate.required == True
        assert len(basic_gate.thresholds) == 2
        
        # Check Efficiency gate
        efficiency_gate = gates[1]
        assert efficiency_gate.name == "Efficiency"
        assert efficiency_gate.required == False
        assert efficiency_gate.weight == 0.7
        
        # Check Model Size gate
        size_gate = gates[2]
        assert size_gate.name == "Model Size"
        assert size_gate.required == False
        assert size_gate.weight == 0.5

    def test_add_quality_gate(self):
        """Test adding custom quality gate"""
        validator = ModelValidator()
        initial_count = len(validator.quality_gates)
        
        custom_gate = QualityGate(
            name="Custom Gate",
            description="Custom gate",
            thresholds=[QualityThreshold(QualityMetric.ACCURACY, min_value=0.95)]
        )
        
        validator.add_quality_gate(custom_gate)
        
        assert len(validator.quality_gates) == initial_count + 1
        assert custom_gate in validator.quality_gates

    @patch('torch.jit.script')
    def test_validate_model_success(self, mock_script):
        """Test successful model validation"""
        validator = ModelValidator()
        
        # Create mock model
        model = nn.Linear(1, 1)
        mock_script.return_value = model
        
        # Mock metrics
        metrics = {
            'accuracy': 0.9,
            'loss': 0.2,
            'inference_time': 50.0,
            'memory_usage': 256.0,
            'model_size': 50.0
        }
        
        result = validator.validate_model(model, metrics)
        
        assert result['valid'] == True
        assert result['model'] == model
        assert 'validation_time' in result
        assert len(result['gate_results']) == 3  # Default gates

    @patch('torch.jit.script')
    def test_validate_model_failure(self, mock_script):
        """Test model validation failure"""
        validator = ModelValidator()
        
        # Create mock model
        model = nn.Linear(1, 1)
        mock_script.return_value = model
        
        # Mock metrics that fail thresholds
        metrics = {
            'accuracy': 0.5,  # Below min threshold of 0.8
            'loss': 0.5,       # Above max threshold of 0.3
            'inference_time': 150.0,  # Above max threshold of 100.0
            'memory_usage': 1024.0,   # Above max threshold of 512.0
            'model_size': 200.0       # Above max threshold of 100.0
        }
        
        result = validator.validate_model(model, metrics)
        
        assert result['valid'] == False
        assert result['model'] == model
        
        # Check that gates failed
        gate_results = result['gate_results']
        assert gate_results[0]['passed'] == False  # Basic Performance
        assert gate_results[1]['passed'] == False  # Efficiency
        assert gate_results[2]['passed'] == False  # Model Size

    @patch('torch.jit.script')
    def test_validate_model_partial_failure(self, mock_script):
        """Test model validation with partial failure"""
        validator = ModelValidator()
        
        # Create mock model
        model = nn.Linear(1, 1)
        mock_script.return_value = model
        
        # Mock metrics that pass required gates but fail optional ones
        metrics = {
            'accuracy': 0.9,  # Passes
            'loss': 0.2,      # Passes
            'inference_time': 150.0,  # Fails efficiency
            'memory_usage': 1024.0,   # Fails efficiency
            'model_size': 200.0       # Fails model size
        }
        
        result = validator.validate_model(model, metrics)
        
        # Should still be valid since required gates pass
        assert result['valid'] == True
        
        gate_results = result['gate_results']
        assert gate_results[0]['passed'] == True   # Basic Performance (required)
        assert gate_results[1]['passed'] == False  # Efficiency (optional)
        assert gate_results[2]['passed'] == False  # Model Size (optional)

    def test_compute_validation_score(self):
        """Test validation score computation"""
        validator = ModelValidator()
        
        gate_results = [
            {'passed': True, 'required': True, 'weight': 1.0},
            {'passed': False, 'required': False, 'weight': 0.7},
            {'passed': True, 'required': False, 'weight': 0.5}
        ]
        
        score = validator._compute_validation_score(gate_results)
        
        # Score should be weighted average of passed gates
        # (1.0 * 1.0 + 0.0 * 0.7 + 1.0 * 0.5) / (1.0 + 0.7 + 0.5) = 1.5 / 2.2 ≈ 0.68
        expected_score = (1.0 * 1.0 + 0.0 * 0.7 + 1.0 * 0.5) / (1.0 + 0.7 + 0.5)
        assert abs(score - expected_score) < 1e-6

    def test_generate_validation_report(self):
        """Test validation report generation"""
        validator = ModelValidator()
        
        gate_results = [
            {
                'gate_name': 'Test Gate',
                'passed': True,
                'required': True,
                'weight': 1.0,
                'results': {'accuracy': {'passed': True, 'value': 0.9}}
            }
        ]
        
        report = validator._generate_validation_report(gate_results, 0.8)
        
        assert 'summary' in report
        assert 'gate_details' in report
        assert 'recommendations' in report
        assert report['summary']['overall_score'] == 0.8
        assert report['summary']['passed_gates'] == 1
        assert report['summary']['total_gates'] == 1


class TestDevelopmentWorkflow:
    """Test the DevelopmentWorkflow class"""

    def test_initialization_default(self):
        """Test development workflow initialization with default parameters"""
        workflow = DevelopmentWorkflow()
        
        assert workflow.validator is not None
        assert workflow.model_registry is not None
        assert workflow.experiment_tracker is not None
        assert workflow.logger is not None

    def test_initialization_custom(self):
        """Test development workflow initialization with custom parameters"""
        validator = ModelValidator()
        registry = ModelRegistry()
        
        workflow = DevelopmentWorkflow(
            validator=validator,
            model_registry=registry
        )
        
        assert workflow.validator == validator
        assert workflow.model_registry == registry

    def test_create_experiment(self):
        """Test creating a new experiment"""
        workflow = DevelopmentWorkflow()
        
        experiment = workflow.create_experiment(
            name="test_experiment",
            description="Test experiment",
            tags=["test", "ml"]
        )
        
        assert experiment['name'] == "test_experiment"
        assert experiment['description'] == "Test experiment"
        assert experiment['tags'] == ["test", "ml"]
        assert 'experiment_id' in experiment
        assert 'created_at' in experiment
        assert 'status' in experiment

    def test_log_experiment_metrics(self):
        """Test logging experiment metrics"""
        workflow = DevelopmentWorkflow()
        
        # Create experiment first
        experiment = workflow.create_experiment("test_exp", "Test")
        experiment_id = experiment['experiment_id']
        
        # Log metrics
        metrics = {
            'accuracy': 0.9,
            'loss': 0.1,
            'epoch': 10
        }
        
        workflow.log_experiment_metrics(experiment_id, metrics)
        
        # Verify metrics were logged
        assert experiment_id in workflow.experiment_tracker
        assert 'metrics' in workflow.experiment_tracker[experiment_id]

    def test_register_model(self):
        """Test registering a model"""
        workflow = DevelopmentWorkflow()
        
        # Create experiment
        experiment = workflow.create_experiment("test_exp", "Test")
        experiment_id = experiment['experiment_id']
        
        # Create mock model
        model = nn.Linear(1, 1)
        
        # Register model
        model_info = workflow.register_model(
            experiment_id=experiment_id,
            model=model,
            metrics={'accuracy': 0.9},
            metadata={'version': '1.0'}
        )
        
        assert 'model_id' in model_info
        assert 'experiment_id' in model_info
        assert model_info['experiment_id'] == experiment_id
        assert 'registered_at' in model_info

    def test_validate_and_promote_model(self):
        """Test validating and promoting a model"""
        workflow = DevelopmentWorkflow()
        
        # Create experiment and register model
        experiment = workflow.create_experiment("test_exp", "Test")
        experiment_id = experiment['experiment_id']
        
        model = nn.Linear(1, 1)
        model_info = workflow.register_model(
            experiment_id=experiment_id,
            model=model,
            metrics={'accuracy': 0.9}
        )
        model_id = model_info['model_id']
        
        # Mock validation to pass
        with patch.object(workflow.validator, 'validate_model') as mock_validate:
            mock_validate.return_value = {
                'valid': True,
                'model': model,
                'validation_score': 0.9,
                'gate_results': []
            }
            
            # Validate and promote
            result = workflow.validate_and_promote_model(model_id)
            
            assert result['promoted'] == True
            assert result['model_id'] == model_id
            assert 'promoted_at' in result

    def test_validate_and_promote_model_failure(self):
        """Test validation failure during promotion"""
        workflow = DevelopmentWorkflow()
        
        # Create experiment and register model
        experiment = workflow.create_experiment("test_exp", "Test")
        experiment_id = experiment['experiment_id']
        
        model = nn.Linear(1, 1)
        model_info = workflow.register_model(
            experiment_id=experiment_id,
            model=model,
            metrics={'accuracy': 0.5}  # Low accuracy
        )
        model_id = model_info['model_id']
        
        # Mock validation to fail
        with patch.object(workflow.validator, 'validate_model') as mock_validate:
            mock_validate.return_value = {
                'valid': False,
                'model': model,
                'validation_score': 0.3,
                'gate_results': []
            }
            
            # Validate and promote
            result = workflow.validate_and_promote_model(model_id)
            
            assert result['promoted'] == False
            assert result['model_id'] == model_id
            assert 'validation_failed' in result

    def test_get_experiment_summary(self):
        """Test getting experiment summary"""
        workflow = DevelopmentWorkflow()
        
        # Create experiment
        experiment = workflow.create_experiment("test_exp", "Test")
        experiment_id = experiment['experiment_id']
        
        # Log some metrics
        workflow.log_experiment_metrics(experiment_id, {'accuracy': 0.9})
        workflow.log_experiment_metrics(experiment_id, {'accuracy': 0.95})
        
        # Get summary
        summary = workflow.get_experiment_summary(experiment_id)
        
        assert 'experiment_id' in summary
        assert 'name' in summary
        assert 'metrics_count' in summary
        assert 'models_count' in summary
        assert summary['metrics_count'] == 2


class TestProductionWorkflow:
    """Test the ProductionWorkflow class"""

    def test_initialization_default(self):
        """Test production workflow initialization with default parameters"""
        workflow = ProductionWorkflow()
        
        assert workflow.model_registry is not None
        assert workflow.deployment_manager is not None
        assert workflow.monitoring_system is not None
        assert workflow.logger is not None

    def test_initialization_custom(self):
        """Test production workflow initialization with custom parameters"""
        registry = ModelRegistry()
        
        workflow = ProductionWorkflow(model_registry=registry)
        
        assert workflow.model_registry == registry

    def test_deploy_model(self):
        """Test deploying a model to production"""
        workflow = ProductionWorkflow()
        
        # Mock model registry
        with patch.object(workflow.model_registry, 'get_model') as mock_get:
            mock_model = nn.Linear(1, 1)
            mock_get.return_value = {
                'model': mock_model,
                'metadata': {'version': '1.0'},
                'status': DeploymentStatus.VALIDATION
            }
            
            # Mock deployment manager
            with patch.object(workflow.deployment_manager, 'deploy') as mock_deploy:
                mock_deploy.return_value = {
                    'deployment_id': 'deploy_123',
                    'status': 'deployed',
                    'endpoint': 'http://api.example.com/model'
                }
                
                # Deploy model
                result = workflow.deploy_model('model_123')
                
                assert result['deployed'] == True
                assert result['model_id'] == 'model_123'
                assert result['deployment_id'] == 'deploy_123'
                assert 'deployed_at' in result

    def test_deploy_model_not_found(self):
        """Test deploying a model that doesn't exist"""
        workflow = ProductionWorkflow()
        
        # Mock model registry to return None
        with patch.object(workflow.model_registry, 'get_model') as mock_get:
            mock_get.return_value = None
            
            # Deploy model
            result = workflow.deploy_model('nonexistent_model')
            
            assert result['deployed'] == False
            assert 'error' in result
            assert 'not found' in result['error'].lower()

    def test_deploy_model_not_validated(self):
        """Test deploying a model that isn't validated"""
        workflow = ProductionWorkflow()
        
        # Mock model registry
        with patch.object(workflow.model_registry, 'get_model') as mock_get:
            mock_model = nn.Linear(1, 1)
            mock_get.return_value = {
                'model': mock_model,
                'metadata': {'version': '1.0'},
                'status': DeploymentStatus.DEVELOPMENT  # Not validated
            }
            
            # Deploy model
            result = workflow.deploy_model('model_123')
            
            assert result['deployed'] == False
            assert 'error' in result
            assert 'not validated' in result['error'].lower()

    def test_monitor_deployment(self):
        """Test monitoring a deployment"""
        workflow = ProductionWorkflow()
        
        # Mock monitoring system
        with patch.object(workflow.monitoring_system, 'get_metrics') as mock_get_metrics:
            mock_get_metrics.return_value = {
                'requests_per_minute': 100,
                'average_latency': 50.0,
                'error_rate': 0.01,
                'cpu_usage': 0.7,
                'memory_usage': 0.6
            }
            
            # Monitor deployment
            metrics = workflow.monitor_deployment('deploy_123')
            
            assert 'requests_per_minute' in metrics
            assert 'average_latency' in metrics
            assert 'error_rate' in metrics
            assert 'cpu_usage' in metrics
            assert 'memory_usage' in metrics

    def test_rollback_deployment(self):
        """Test rolling back a deployment"""
        workflow = ProductionWorkflow()
        
        # Mock deployment manager
        with patch.object(workflow.deployment_manager, 'rollback') as mock_rollback:
            mock_rollback.return_value = {
                'rollback_id': 'rollback_123',
                'status': 'rolled_back',
                'previous_deployment': 'deploy_122'
            }
            
            # Rollback deployment
            result = workflow.rollback_deployment('deploy_123')
            
            assert result['rolled_back'] == True
            assert result['deployment_id'] == 'deploy_123'
            assert result['rollback_id'] == 'rollback_123'
            assert 'rolled_back_at' in result

    def test_get_deployment_status(self):
        """Test getting deployment status"""
        workflow = ProductionWorkflow()
        
        # Mock deployment manager
        with patch.object(workflow.deployment_manager, 'get_status') as mock_get_status:
            mock_get_status.return_value = {
                'deployment_id': 'deploy_123',
                'status': 'active',
                'created_at': datetime.now(),
                'endpoint': 'http://api.example.com/model',
                'version': '1.0'
            }
            
            # Get status
            status = workflow.get_deployment_status('deploy_123')
            
            assert status['deployment_id'] == 'deploy_123'
            assert status['status'] == 'active'
            assert 'endpoint' in status
            assert 'version' in status

    def test_list_deployments(self):
        """Test listing all deployments"""
        workflow = ProductionWorkflow()
        
        # Mock deployment manager
        with patch.object(workflow.deployment_manager, 'list_deployments') as mock_list:
            mock_list.return_value = [
                {'deployment_id': 'deploy_123', 'status': 'active'},
                {'deployment_id': 'deploy_124', 'status': 'inactive'}
            ]
            
            # List deployments
            deployments = workflow.list_deployments()
            
            assert len(deployments) == 2
            assert deployments[0]['deployment_id'] == 'deploy_123'
            assert deployments[1]['deployment_id'] == 'deploy_124'


# Integration tests
class TestWorkflowIntegration:
    """Integration tests for workflow modules"""

    def test_full_development_to_production_workflow(self):
        """Test complete workflow from development to production"""
        # Create development workflow
        dev_workflow = DevelopmentWorkflow()
        
        # Create experiment
        experiment = dev_workflow.create_experiment(
            name="integration_test",
            description="Integration test experiment"
        )
        experiment_id = experiment['experiment_id']
        
        # Log metrics
        dev_workflow.log_experiment_metrics(experiment_id, {'accuracy': 0.9})
        
        # Register model
        model = nn.Linear(1, 1)
        model_info = dev_workflow.register_model(
            experiment_id=experiment_id,
            model=model,
            metrics={'accuracy': 0.9}
        )
        model_id = model_info['model_id']
        
        # Mock validation to pass
        with patch.object(dev_workflow.validator, 'validate_model') as mock_validate:
            mock_validate.return_value = {
                'valid': True,
                'model': model,
                'validation_score': 0.9,
                'gate_results': []
            }
            
            # Promote model
            promotion_result = dev_workflow.validate_and_promote_model(model_id)
            assert promotion_result['promoted'] == True
        
        # Create production workflow
        prod_workflow = ProductionWorkflow()
        
        # Mock production deployment
        with patch.object(prod_workflow.model_registry, 'get_model') as mock_get:
            mock_get.return_value = {
                'model': model,
                'metadata': {'version': '1.0'},
                'status': DeploymentStatus.VALIDATION
            }
            
            with patch.object(prod_workflow.deployment_manager, 'deploy') as mock_deploy:
                mock_deploy.return_value = {
                    'deployment_id': 'deploy_123',
                    'status': 'deployed',
                    'endpoint': 'http://api.example.com/model'
                }
                
                # Deploy model
                deployment_result = prod_workflow.deploy_model(model_id)
                assert deployment_result['deployed'] == True

    def test_quality_gate_evaluation_integration(self):
        """Test quality gate evaluation integration"""
        validator = ModelValidator()
        
        # Create custom quality gate
        custom_gate = QualityGate(
            name="Custom Performance",
            description="Custom performance requirements",
            thresholds=[
                QualityThreshold(QualityMetric.ACCURACY, min_value=0.95),
                QualityThreshold(QualityMetric.INFERENCE_TIME, max_value=50.0)
            ],
            required=True,
            weight=1.0
        )
        
        validator.add_quality_gate(custom_gate)
        
        # Test with passing metrics
        passing_metrics = {
            'accuracy': 0.96,
            'inference_time': 45.0
        }
        
        result = custom_gate.evaluate(passing_metrics)
        assert result['passed'] == True
        
        # Test with failing metrics
        failing_metrics = {
            'accuracy': 0.90,  # Below threshold
            'inference_time': 60.0  # Above threshold
        }
        
        result = custom_gate.evaluate(failing_metrics)
        assert result['passed'] == False

    def test_model_validation_integration(self):
        """Test model validation integration"""
        validator = ModelValidator()
        
        # Create a simple model
        model = nn.Linear(10, 1)
        
        # Mock torch.jit.script to avoid actual scripting
        with patch('torch.jit.script') as mock_script:
            mock_script.return_value = model
            
            # Test validation with comprehensive metrics
            metrics = {
                'accuracy': 0.85,
                'loss': 0.25,
                'precision': 0.82,
                'recall': 0.88,
                'f1_score': 0.85,
                'inference_time': 80.0,
                'memory_usage': 400.0,
                'model_size': 80.0
            }
            
            result = validator.validate_model(model, metrics)
            
            assert 'valid' in result
            assert 'model' in result
            assert 'validation_score' in result
            assert 'gate_results' in result
            assert 'validation_time' in result
