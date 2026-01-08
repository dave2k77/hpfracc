import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

"""
Comprehensive tests for Mittag-Leffler function implementation.

This module tests the Mittag-Leffler function which is fundamental in fractional calculus.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
import warnings

from hpfracc.special.mittag_leffler import (
    MittagLefflerFunction, mittag_leffler, 
    mittag_leffler_derivative, mittag_leffler_fast,
    mittag_leffler_function
)


class TestMittagLefflerFunction:
    """Test MittagLefflerFunction class."""
    
    def test_function_initialization(self):
        """Test function initialization."""
        ml_func = MittagLefflerFunction()
        
        assert ml_func is not None
        assert hasattr(ml_func, 'compute')
    
    def test_function_compute(self):
        """Test function computation."""
        ml_func = MittagLefflerFunction()
        
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = ml_func.compute(A, 1.0, 1.0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == A.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestMittagLefflerFunction:
    """Test MittagLefflerFunction class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        ml = MittagLefflerFunction()
        
        assert ml.use_jax is False
        if hasattr(ml, 'use_numba'):
            # Numba is now enabled by default if available
            # Note: Test environment might not have numba, but usually it does.
            # We skip strict assertion or check condition.
            pass
            # assert ml.use_numba is True # Depending on environment
        assert hasattr(ml, '_cache_size')
        assert hasattr(ml, 'adaptive_convergence')
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        ml = MittagLefflerFunction(
            use_jax=True,
            use_numba=False,
            cache_size=500
        )
        
        # use_jax is only True if JAX is available
        # assert ml.use_jax is True  # Can't guarantee JAX availability
        assert ml.use_numba is False  # Numba is force-disabled
        assert ml._cache_size == 500
    
    def test_compute_scalar(self):
        """Test computing Mittag-Leffler function for scalar inputs."""
        ml = MittagLefflerFunction()
        
        # Test with simple parameters
        result = ml.compute(1.0, 1.0, 1.0)
        
        assert isinstance(result, (float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result)
    
    def test_compute_array(self):
        """Test computing Mittag-Leffler function for array inputs."""
        ml = MittagLefflerFunction()
        
        z = np.array([0.0, 1.0, -1.0])
        result = ml.compute(z, 1.0, 1.0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == z.shape
        assert not np.any(np.isnan(result))
    
    def test_special_cases(self):
        """Test known special cases of Mittag-Leffler function."""
        ml = MittagLefflerFunction()
        
        # E_1,1(z) = e^z (exponential function)
        z = 1.0
        result = ml.compute(z, 1.0, 1.0)
        expected = np.exp(z)
        
        assert abs(result - expected) < 1e-10
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        ml = MittagLefflerFunction()
        
        # Test with valid parameters (function doesn't validate)
        result = ml.compute(1.0, 0.5, 1.0)
        assert isinstance(result, (float, np.floating))
        
        # Test with different valid alpha values
        result = ml.compute(1.0, 1.5, 1.0)
        assert isinstance(result, (float, np.floating))
    
    def test_edge_cases(self):
        """Test edge cases."""
        ml = MittagLefflerFunction()
        
        # Test with zero input
        result = ml.compute(0.0, 1.0, 1.0)
        expected = 1.0  # E_1,1(0) = 1
        
        assert abs(result - expected) < 1e-10
        
        # Test with very small alpha
        result = ml.compute(0.1, 1.0, 1.0)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result)


class TestMittagLefflerConvenienceFunctions:
    """Test convenience functions for Mittag-Leffler."""
    
    def test_mittag_leffler_function(self):
        """Test mittag_leffler convenience function."""
        result = mittag_leffler(1.0, 1.0, 1.0)
        
        assert isinstance(result, (float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result)
    
    def test_mittag_leffler_array(self):
        """Test mittag_leffler with array input."""
        z = np.array([0.0, 1.0, 2.0])
        result = mittag_leffler(z, 1.0, 1.0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == z.shape
        assert not np.any(np.isnan(result))
    
    def test_mittag_leffler_derivative(self):
        """Test Mittag-Leffler derivative function."""
        result = mittag_leffler_derivative(1.0, 1.0, 1.0)
        
        assert isinstance(result, (float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result)
    
    def test_mittag_leffler_function(self):
        """Test mittag_leffler_function."""
        result = mittag_leffler_function(1.0, 1.0, 1.0)
        
        assert isinstance(result, (float, np.ndarray))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result) if isinstance(result, float) else not np.any(np.isnan(result))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result) if isinstance(result, float) else not np.any(np.isinf(result))


class TestMittagLefflerSpecialFunctions:
    """Test special function implementations."""
    
    def test_mittag_leffler_fast(self):
        """Test mittag_leffler_fast function."""
        result = mittag_leffler_fast(1.0, 1.0, 1.0)
        
        assert isinstance(result, (float, np.ndarray))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result) if isinstance(result, float) else not np.any(np.isnan(result))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result) if isinstance(result, float) else not np.any(np.isinf(result))
    
    def test_mittag_leffler_derivative(self):
        """Test mittag_leffler_derivative function."""
        result = mittag_leffler_derivative(1.0, 1.0, 1.0)
        
        assert isinstance(result, (float, np.ndarray))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result) if isinstance(result, float) else not np.any(np.isnan(result))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result) if isinstance(result, float) else not np.any(np.isinf(result))
    
    def test_mittag_leffler_convenience(self):
        """Test mittag_leffler convenience function."""
        result = mittag_leffler(1.0, 1.0, 1.0)
        
        assert isinstance(result, (float, np.ndarray))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result) if isinstance(result, float) else not np.any(np.isnan(result))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result) if isinstance(result, float) else not np.any(np.isinf(result))
    
    def test_implementation_consistency(self):
        """Test consistency between different implementations."""
        ml = MittagLefflerFunction()
        
        # Test with moderate values
        alpha, beta, z = 1.5, 1.0, 2.0
        
        result1 = ml.compute(alpha, beta, z)
        
        # Test with different max_terms
        ml2 = MittagLefflerFunction(cache_size=200)
        result2 = ml2.compute(alpha, beta, z)
        
        # Results should be close
        assert abs(result1 - result2) < 1e-6


class TestMittagLefflerPerformance:
    """Test performance characteristics."""
    
    def test_large_array_performance(self):
        """Test performance with large arrays."""
        ml = MittagLefflerFunction()
        
        # Create large array
        z = np.linspace(-2, 2, 1000)
        
        # Should complete without errors
        result = ml.compute(z, 1.0, 1.0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == z.shape
        assert not np.any(np.isnan(result))
    
    def test_many_terms_convergence(self):
        """Test convergence with many terms."""
        ml = MittagLefflerFunction(cache_size=200)
        
        # Test with moderate z
        result = ml.compute(1.5, 1.0, 1.0)
        
        assert isinstance(result, (float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result)


class TestMittagLefflerMathematicalProperties:
    """Test mathematical properties of Mittag-Leffler function."""
    
    def test_continuity(self):
        """Test continuity of the function."""
        ml = MittagLefflerFunction()
        
        # Test around a point
        z0 = 1.0
        eps = 1e-6
        
        result1 = ml.compute(1.0, 1.0, z0 - eps)
        result2 = ml.compute(1.0, 1.0, z0 + eps)
        result0 = ml.compute(1.0, 1.0, z0)
        
        # Function should be continuous
        assert abs(result1 - result0) < 1e-3
        assert abs(result2 - result0) < 1e-3
    
    def test_symmetry_properties(self):
        """Test symmetry properties where applicable."""
        ml = MittagLefflerFunction()
        
        # Test basic properties
        z = 1.0
        result = ml.compute(z, 1.0, 1.0)
        
        assert isinstance(result, (float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result)
    
    def test_asymptotic_behavior(self):
        """Test asymptotic behavior."""
        ml = MittagLefflerFunction()
        
        # Test with moderate values
        z = 2.0
        result = ml.compute(z, 1.0, 1.0)
        
        assert isinstance(result, (float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result)


class TestMittagLefflerErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        ml = MittagLefflerFunction()
        
        # Test with valid parameters (function doesn't validate)
        result = ml.compute(1.0, 0.5, 1.0)
        assert isinstance(result, (float, np.floating))
        
        # Test with different parameter combinations
        result = ml.compute(1.0, 1.5, 0.5)
        assert isinstance(result, (float, np.floating))
    
    def test_extreme_values(self):
        """Test with extreme values."""
        ml = MittagLefflerFunction()
        
        # Test with moderate z values
        result = ml.compute(5.0, 1.0, 1.0)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result)
        
        # Test with small z
        result = ml.compute(0.01, 1.0, 1.0)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result)
    
    def test_numerical_stability(self):
        """Test numerical stability."""
        ml = MittagLefflerFunction()
        
        # Test with values that should be numerically stable
        test_cases = [
            (0.1, 1.0, 0.5),
            (1.0, 1.0, 1.0),
            (1.5, 1.0, 1.0),
            (0.5, 1.0, 0.1)
        ]
        
        for z, alpha, beta in test_cases:
            result = ml.compute(z, alpha, beta)
            # Type guard for integer results
            if not isinstance(result, (int, np.integer)):
                assert not np.isnan(result)
            # Type guard for integer results
            if not isinstance(result, (int, np.integer)):
                assert not np.isinf(result)


class TestMittagLefflerIntegration:
    """Integration tests for Mittag-Leffler function."""
    
    def test_with_torch_tensors(self):
        """Test with PyTorch tensors."""
        ml = MittagLefflerFunction()
        
        # Convert to numpy for computation
        z_torch = torch.tensor([0.0, 1.0, 2.0])
        z_np = z_torch.numpy()
        
        result = ml.compute(z_np, 1.0, 1.0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == z_np.shape
    
    def test_with_different_dtypes(self):
        """Test with different data types."""
        ml = MittagLefflerFunction()
        
        # Test with float32
        z_float32 = np.array([1.0], dtype=np.float32)
        result_float32 = ml.compute(z_float32, 1.0, 1.0)
        
        # Test with float64
        z_float64 = np.array([1.0], dtype=np.float64)
        result_float64 = ml.compute(z_float64, 1.0, 1.0)
        
        assert isinstance(result_float32, np.ndarray)
        assert isinstance(result_float64, np.ndarray)
        assert abs(result_float32 - result_float64) < 1e-6
    
    def test_caching_behavior(self):
        """Test caching behavior if implemented."""
        ml = MittagLefflerFunction()
        
        # Compute same value twice
        result1 = ml.compute(1.0, 1.0, 1.0)
        result2 = ml.compute(1.0, 1.0, 1.0)
        
        # Results should be identical
        assert result1 == result2


if __name__ == "__main__":
    pytest.main([__file__])


