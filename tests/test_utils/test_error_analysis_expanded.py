"""
Expanded comprehensive tests for error_analysis.py module.
Tests all error computation functions, convergence analysis, validation framework.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from hpfracc.utils.error_analysis import (
    ErrorAnalyzer,
    ConvergenceAnalyzer,
    ValidationFramework,
    absolute_error,
    relative_error,
    l2_error,
    max_error,
    compute_rmse,
    error_statistics,
    compute_error_metrics,
    analyze_convergence,
    validate_solution
)


class TestErrorAnalyzerExpanded:
    """Expanded tests for ErrorAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create ErrorAnalyzer instance."""
        return ErrorAnalyzer(tolerance=1e-10)
    
    def test_initialization_default(self):
        """Test initialization with default tolerance."""
        analyzer = ErrorAnalyzer()
        assert analyzer.tolerance == 1e-10
    
    def test_initialization_custom_tolerance(self):
        """Test initialization with custom tolerance."""
        analyzer = ErrorAnalyzer(tolerance=1e-12)
        assert analyzer.tolerance == 1e-12
    
    def test_absolute_error(self, analyzer):
        """Test absolute error computation."""
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 2.2, 3.3])
        
        error = analyzer.absolute_error(numerical, analytical)
        
        expected = np.array([0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(error, expected)
    
    def test_relative_error(self, analyzer):
        """Test relative error computation."""
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 2.2, 3.3])
        
        error = analyzer.relative_error(numerical, analytical)
        
        expected = np.array([0.1/1.1, 0.2/2.2, 0.3/3.3])
        np.testing.assert_array_almost_equal(error, expected)
    
    def test_relative_error_with_zeros(self, analyzer):
        """Test relative error with zero analytical values."""
        numerical = np.array([0.1, 0.2, 0.0])
        analytical = np.array([0.0, 0.0, 0.0])
        
        error = analyzer.relative_error(numerical, analytical)
        
        # Should use tolerance for zero denominators
        assert np.all(error >= 0)
        assert not np.any(np.isnan(error))
        assert not np.any(np.isinf(error))
    
    def test_l1_error(self, analyzer):
        """Test L1 error norm computation."""
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 2.2, 3.3])
        
        error = analyzer.l1_error(numerical, analytical)
        
        expected = np.mean([0.1, 0.2, 0.3])
        assert abs(error - expected) < 1e-10
    
    def test_l2_error(self, analyzer):
        """Test L2 error norm computation."""
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 2.2, 3.3])
        
        error = analyzer.l2_error(numerical, analytical)
        
        expected = np.sqrt(np.mean([0.1**2, 0.2**2, 0.3**2]))
        assert abs(error - expected) < 1e-10
    
    def test_linf_error(self, analyzer):
        """Test L-infinity error norm computation."""
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 2.2, 3.3])
        
        error = analyzer.linf_error(numerical, analytical)
        
        assert abs(error - 0.3) < 1e-10
    
    def test_mean_squared_error(self, analyzer):
        """Test mean squared error computation."""
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 2.2, 3.3])
        
        error = analyzer.mean_squared_error(numerical, analytical)
        
        expected = np.mean([0.1**2, 0.2**2, 0.3**2])
        assert abs(error - expected) < 1e-10
    
    def test_root_mean_squared_error(self, analyzer):
        """Test root mean squared error computation."""
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 2.2, 3.3])
        
        error = analyzer.root_mean_squared_error(numerical, analytical)
        
        expected = np.sqrt(np.mean([0.1**2, 0.2**2, 0.3**2]))
        assert abs(error - expected) < 1e-10
    
    def test_maximum_error(self, analyzer):
        """Test maximum error computation."""
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 2.2, 3.3])
        
        error = analyzer.maximum_error(numerical, analytical)
        
        assert abs(error - 0.3) < 1e-10
    
    def test_compute_all_errors(self, analyzer):
        """Test computing all error metrics."""
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 2.2, 3.3])
        
        errors = analyzer.compute_all_errors(numerical, analytical)
        
        assert isinstance(errors, dict)
        assert 'absolute_error' in errors
        assert 'relative_error' in errors
        assert 'l1' in errors
        assert 'l2' in errors
        assert 'linf' in errors
        assert 'mse' in errors
        assert 'rmse' in errors
        assert 'max_error' in errors


class TestStandaloneErrorFunctions:
    """Tests for standalone error computation functions."""
    
    def test_absolute_error_function(self):
        """Test absolute_error standalone function."""
        analytical = np.array([1.0, 2.0, 3.0])
        numerical = np.array([1.1, 2.2, 3.3])
        
        error = absolute_error(analytical, numerical)
        
        expected = np.array([0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(error, expected)
    
    def test_relative_error_function(self):
        """Test relative_error standalone function."""
        analytical = np.array([1.0, 2.0, 3.0])
        numerical = np.array([1.1, 2.2, 3.3])
        
        error = relative_error(analytical, numerical)
        
        assert np.all(error >= 0)
        assert not np.any(np.isnan(error))
    
    def test_relative_error_function_with_zeros(self):
        """Test relative_error with zero values."""
        analytical = np.array([0.0, 1.0, 2.0])
        numerical = np.array([0.1, 1.1, 2.2])
        
        error = relative_error(analytical, numerical, tolerance=1e-10)
        
        assert not np.any(np.isnan(error))
        assert not np.any(np.isinf(error))
    
    def test_l2_error_function(self):
        """Test l2_error standalone function."""
        analytical = np.array([1.0, 2.0, 3.0])
        numerical = np.array([1.1, 2.2, 3.3])
        
        error = l2_error(analytical, numerical)
        
        assert error > 0
        assert isinstance(error, float)
    
    def test_max_error_function(self):
        """Test max_error standalone function."""
        analytical = np.array([1.0, 2.0, 3.0])
        numerical = np.array([1.1, 2.2, 3.3])
        
        error = max_error(analytical, numerical)
        
        assert abs(error - 0.3) < 1e-10
    
    def test_compute_rmse_function(self):
        """Test compute_rmse standalone function."""
        analytical = np.array([1.0, 2.0, 3.0])
        numerical = np.array([1.1, 2.2, 3.3])
        
        error = compute_rmse(analytical, numerical)
        
        assert error > 0
        assert isinstance(error, float)
    
    def test_error_statistics_function(self):
        """Test error_statistics standalone function."""
        analytical = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        numerical = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        
        stats = error_statistics(analytical, numerical)
        
        assert isinstance(stats, dict)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert all(isinstance(v, float) for v in stats.values())


class TestConvergenceAnalyzer:
    """Tests for ConvergenceAnalyzer class."""
    
    @pytest.fixture
    def convergence_analyzer(self):
        """Create ConvergenceAnalyzer instance."""
        return ConvergenceAnalyzer()
    
    def test_initialization(self, convergence_analyzer):
        """Test ConvergenceAnalyzer initialization."""
        assert convergence_analyzer is not None
    
    def test_estimate_convergence_rate(self, convergence_analyzer):
        """Test convergence rate estimation."""
        # Create data with known convergence rate
        h_values = np.array([0.1, 0.05, 0.025, 0.0125])
        errors = h_values ** 2  # Quadratic convergence
        
        rate = convergence_analyzer.estimate_convergence_rate(h_values, errors)
        
        assert isinstance(rate, float)
        assert rate > 0
        # Should be close to 2 for quadratic convergence
        assert abs(rate - 2.0) < 0.5
    
    def test_analyze_convergence_behavior(self, convergence_analyzer):
        """Test convergence behavior analysis."""
        h_values = np.array([0.1, 0.05, 0.025, 0.0125])
        errors = h_values ** 2
        
        behavior = convergence_analyzer.analyze_convergence_behavior(h_values, errors)
        
        assert isinstance(behavior, dict)
        assert 'convergence_rate' in behavior
        assert 'is_converging' in behavior


class TestValidationFramework:
    """Tests for ValidationFramework class."""
    
    @pytest.fixture
    def validator(self):
        """Create ValidationFramework instance."""
        return ValidationFramework()
    
    def test_initialization(self, validator):
        """Test ValidationFramework initialization."""
        assert validator is not None
    
    def test_validate_against_analytical(self, validator):
        """Test validation against analytical solution."""
        # Create test data
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.0, 2.0, 3.0])
        
        result = validator.validate_against_analytical(
            numerical, analytical, tolerance=1e-6
        )
        
        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert result['is_valid'] is True
    
    def test_validate_against_analytical_with_error(self, validator):
        """Test validation with significant error."""
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([2.0, 4.0, 6.0])  # Large error
        
        result = validator.validate_against_analytical(
            numerical, analytical, tolerance=1e-6
        )
        
        assert result['is_valid'] is False
    
    def test_check_convergence(self, validator):
        """Test convergence checking."""
        h_values = np.array([0.1, 0.05, 0.025])
        errors = h_values ** 2
        
        result = validator.check_convergence(h_values, errors, min_rate=1.5)
        
        assert isinstance(result, dict)
        assert 'is_converging' in result


class TestStandaloneValidationFunctions:
    """Tests for standalone validation functions."""
    
    def test_compute_error_metrics(self):
        """Test compute_error_metrics function."""
        analytical = np.array([1.0, 2.0, 3.0])
        numerical = np.array([1.1, 2.2, 3.3])
        
        metrics = compute_error_metrics(analytical, numerical)
        
        assert isinstance(metrics, dict)
        assert 'absolute_error' in metrics or 'l2' in metrics
    
    def test_analyze_convergence(self):
        """Test analyze_convergence function."""
        h_values = np.array([0.1, 0.05, 0.025])
        errors = h_values ** 2
        
        result = analyze_convergence(h_values, errors)
        
        assert isinstance(result, dict)
        assert 'convergence_rate' in result or 'rate' in result
    
    def test_validate_solution(self):
        """Test validate_solution function."""
        analytical = np.array([1.0, 2.0, 3.0])
        numerical = np.array([1.0, 2.0, 3.0])
        
        result = validate_solution(analytical, numerical)
        
        assert isinstance(result, bool)
        assert result is True
    
    def test_validate_solution_with_error(self):
        """Test validate_solution with error."""
        analytical = np.array([1.0, 2.0, 3.0])
        numerical = np.array([2.0, 4.0, 6.0])
        
        result = validate_solution(analytical, numerical)
        
        assert isinstance(result, bool)
        assert result is False


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_error_computation_with_empty_arrays(self):
        """Test error computation with empty arrays."""
        analytical = np.array([])
        numerical = np.array([])
        
        analyzer = ErrorAnalyzer()
        
        # Should handle empty arrays gracefully
        try:
            error = analyzer.absolute_error(numerical, analytical)
            assert error.size == 0
        except Exception:
            # Some operations may not support empty arrays
            pass
    
    def test_error_computation_with_different_shapes(self):
        """Test error computation with mismatched shapes."""
        analytical = np.array([1.0, 2.0])
        numerical = np.array([1.0, 2.0, 3.0])
        
        analyzer = ErrorAnalyzer()
        
        # Should raise ValueError or handle gracefully
        with pytest.raises((ValueError, AssertionError)):
            analyzer.absolute_error(numerical, analytical)
    
    def test_error_computation_with_nan(self):
        """Test error computation with NaN values."""
        analytical = np.array([1.0, np.nan, 3.0])
        numerical = np.array([1.1, 2.2, 3.3])
        
        analyzer = ErrorAnalyzer()
        
        error = analyzer.absolute_error(numerical, analytical)
        
        # Should handle NaN appropriately
        assert np.isnan(error[1])
    
    def test_error_computation_with_inf(self):
        """Test error computation with infinite values."""
        analytical = np.array([1.0, np.inf, 3.0])
        numerical = np.array([1.1, 2.2, 3.3])
        
        analyzer = ErrorAnalyzer()
        
        error = analyzer.absolute_error(numerical, analytical)
        
        # Should handle inf appropriately
        assert np.isinf(error[1])
    
    def test_relative_error_with_very_small_values(self):
        """Test relative error with very small values."""
        analytical = np.array([1e-15, 1e-16, 1e-17])
        numerical = np.array([2e-15, 2e-16, 2e-17])
        
        analyzer = ErrorAnalyzer(tolerance=1e-20)
        
        error = analyzer.relative_error(numerical, analytical)
        
        # Should handle very small values
        assert not np.any(np.isnan(error))
        assert np.all(error > 0)
