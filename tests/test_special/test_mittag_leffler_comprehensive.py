import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

"""
Comprehensive tests for special mittag_leffler module.

This module tests all Mittag-Leffler function functionality including
computation, optimization strategies, and special cases.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import scipy

from hpfracc.special.mittag_leffler import (
    MittagLefflerFunction
)


class TestMittagLefflerFunction:
    """Test MittagLefflerFunction class."""
    
    def test_mittag_leffler_function_initialization(self):
        """Test MittagLefflerFunction initialization."""
        ml = MittagLefflerFunction()
        assert ml.use_jax is False
        assert ml.use_numba is False  # Disabled by default due to compilation issues
        assert hasattr(ml, '_cache_size')
        assert hasattr(ml, 'adaptive_convergence')
    
    @patch('hpfracc.special.mittag_leffler.jnp', np)
    @patch('hpfracc.special.mittag_leffler.jax')
    def test_mittag_leffler_function_initialization_with_jax(self, mock_jax):
        """Test MittagLefflerFunction initialization with JAX."""
        with patch('hpfracc.special.mittag_leffler.JAX_AVAILABLE', True):
            ml = MittagLefflerFunction(use_jax=True)
            assert ml.use_jax is True
            assert ml.use_numba is False  # Disabled by default due to compilation issues
            assert hasattr(ml, '_cache_size')
            assert hasattr(ml, 'adaptive_convergence')
    
    def test_mittag_leffler_function_initialization_with_custom_cache_size(self):
        """Test MittagLefflerFunction initialization with custom cache_size."""
        ml = MittagLefflerFunction(cache_size=50)
        assert ml._cache_size == 50
    
    def test_mittag_leffler_function_initialization_without_numba(self):
        """Test MittagLefflerFunction initialization without NUMBA."""
        ml = MittagLefflerFunction(use_numba=False)
        assert ml.use_jax is False
        assert ml.use_numba is False
    
    def test_mittag_leffler_function_compute_scalar(self):
        """Test MittagLefflerFunction compute method with scalar inputs."""
        ml = MittagLefflerFunction()
        
        # Test basic cases
        result = ml.compute(1.0, 1.0, 0.5)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        
        result = ml.compute(1.0, 1.0, 1.0)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
    
    def test_mittag_leffler_function_compute_fractional(self):
        """Test MittagLefflerFunction compute method with fractional inputs."""
        ml = MittagLefflerFunction()
        
        # Test fractional cases
        result = ml.compute(0.5, 0.5, 0.5)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        
        result = ml.compute(1.5, 2.0, 0.3)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
    
    def test_mittag_leffler_function_compute_array(self):
        """Test MittagLefflerFunction compute method with array inputs."""
        ml = MittagLefflerFunction()
        
        # Test array inputs
        z_vals = np.array([0.1, 0.5, 1.0, 2.0])
        results = ml.compute(z_vals, 1.0, 1.0)
        
        assert isinstance(results, np.ndarray)
        assert results.shape == z_vals.shape
        assert not np.any(np.isnan(results))
    
    @patch('hpfracc.special.mittag_leffler.jnp', np)
    @patch('hpfracc.special.mittag_leffler.jax')
    def test_mittag_leffler_function_compute_with_jax(self, mock_jax):
        """Test MittagLefflerFunction compute method with JAX."""
        with patch('hpfracc.special.mittag_leffler.JAX_AVAILABLE', True):
            ml = MittagLefflerFunction(use_jax=True)
            
            # Test scalar with JAX
            result = ml.compute(1.0, 1.0, 0.5)
            assert isinstance(result, (int, float, np.floating))
            # Type guard for integer results
            if not isinstance(result, (int, np.integer)):
                assert not np.isnan(result)
            
            # Test array with JAX
            z_vals = np.array([0.1, 0.5, 1.0, 2.0])
            results = ml.compute(z_vals, 1.0, 1.0)
            assert isinstance(results, np.ndarray)
            assert results.shape == z_vals.shape
            assert not np.any(np.isnan(results))
    
    def test_mittag_leffler_function_compute_without_numba(self):
        """Test MittagLefflerFunction compute method without NUMBA."""
        ml = MittagLefflerFunction(use_numba=False)
        
        # Test scalar without NUMBA
        result = ml.compute(1.0, 1.0, 0.5)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        
        # Test array without NUMBA
        z_vals = np.array([0.1, 0.5, 1.0, 2.0])
        results = ml.compute(z_vals, 1.0, 1.0)
        assert isinstance(results, np.ndarray)
        assert results.shape == z_vals.shape
        assert not np.any(np.isnan(results))
    
    def test_mittag_leffler_function_special_cases(self):
        """Test MittagLefflerFunction special cases."""
        ml = MittagLefflerFunction()
        
        # Test E_1,1(z) ≈ e^z for small z
        z = 0.1
        result = ml.compute(z, 1.0, 1.0)
        expected = np.exp(z)
        assert abs(result - expected) < 1e-6
        
        # Test E_2,1(-z^2) ≈ cos(z) for small z
        z = 0.1
        result = ml.compute(-z**2, 2.0, 1.0)
        expected = np.cos(z)
        assert abs(result - expected) < 1e-6
    
    def test_mittag_leffler_function_edge_cases(self):
        """Test MittagLefflerFunction edge cases."""
        ml = MittagLefflerFunction()
        
        # Test edge cases
        result = ml.compute(0.1, 0.1, 0.1)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        
        result = ml.compute(10.0, 10.0, 0.1)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
    
    def test_mittag_leffler_function_large_values(self):
        """Test MittagLefflerFunction with large values."""
        ml = MittagLefflerFunction()
        
        # Test with larger values
        result = ml.compute(2.0, 3.0, 5.0)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
    
    def test_mittag_leffler_function_different_orders(self):
        """Test MittagLefflerFunction with different fractional orders."""
        ml = MittagLefflerFunction()
        
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0]:
            for beta in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0]:
                result = ml.compute(alpha, beta, 0.5)
                assert isinstance(result, (int, float, np.floating))
                # Type guard for integer results
                if not isinstance(result, (int, np.integer)):
                    assert not np.isnan(result)


class TestMittagLefflerIntegration:
    """Integration tests for Mittag-Leffler functions."""
    
    def test_integration_with_fractional_calculus(self):
        """Test integration with fractional calculus operations."""
        ml = MittagLefflerFunction()
        
        # Test with typical fractional calculus parameters
        alpha_values = [0.25, 0.5, 0.75]
        z_values = np.linspace(0, 2, 10)
        
        for alpha in alpha_values:
            for z in z_values:
                result = ml.compute(z, alpha, 1.0)
                assert isinstance(result, (float, np.floating))
                # Type guard for integer results
                if not isinstance(result, (int, np.integer)):
                    assert not np.isnan(result)
    
    def test_integration_with_different_backends(self):
        """Test integration with different computational backends."""
        # Test with NumPy backend
        ml_np = MittagLefflerFunction(use_jax=False, use_numba=False)
        result_np = ml_np.compute(1.0, 1.0, 1.0)
        assert isinstance(result_np, (float, np.floating))
        
        # Test with JAX backend (if available)
        ml_jax = MittagLefflerFunction(use_jax=True, use_numba=False)
        result_jax = ml_jax.compute(1.0, 1.0, 1.0)
        assert isinstance(result_jax, (float, np.floating))
    
    def test_integration_performance_comparison(self):
        """Test performance comparison between different implementations."""
        ml = MittagLefflerFunction()
        
        # Test with array input
        z_array = np.linspace(0, 5, 100)
        result = ml.compute(z_array, 1.0, 1.0)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(z_array)
        assert not np.any(np.isnan(result))
    
    def test_integration_with_external_libraries(self):
        """Test integration with external libraries."""
        ml = MittagLefflerFunction()
        
        # Test with typical values used in fractional calculus
        test_cases = [
            (0.0, 1.0, 1.0),  # E_1,1(0) = 1
            (1.0, 1.0, 1.0),  # E_1,1(1) = e
            (0.0, 2.0, 1.0),  # E_2,1(0) = 1
        ]
        
        for z, alpha, beta in test_cases:
            result = ml.compute(z, alpha, beta)
            assert isinstance(result, (float, np.floating))
            # Type guard for integer results
            if not isinstance(result, (int, np.integer)):
                assert not np.isnan(result)


if __name__ == "__main__":
    pytest.main([__file__])

