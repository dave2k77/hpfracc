import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

#!/usr/bin/env python3
"""
Direct tests for binomial coefficients to avoid PyTorch import issues.

This test suite focuses on improving coverage for hpfracc/special/binomial_coeffs.py
by importing the module directly without going through the main package.
"""

import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add the project root to the path to import modules directly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import the module directly to avoid PyTorch import issues
try:
    from hpfracc.special.binomial_coeffs import BinomialCoefficients, binomial_coefficient, generalized_binomial
except ImportError as e:
    pytest.skip(f"Could not import binomial_coeffs: {e}", allow_module_level=True)


class TestBinomialCoefficientsDirect:
    """Direct tests for BinomialCoefficients class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.bc = BinomialCoefficients()
    
    def test_binomial_coefficients_initialization(self):
        """Test BinomialCoefficients initialization."""
        # Test default initialization
        bc = BinomialCoefficients()
        assert bc.use_jax == False
        assert bc.use_numba == True
        assert bc.cache_size == 1000
        
        # Test with custom parameters
        bc = BinomialCoefficients(use_jax=True, use_numba=False, cache_size=500)
        assert bc.use_jax == True
        assert bc.use_numba == False
        assert bc.cache_size == 500
    
    def test_binomial_coefficients_basic_computation(self):
        """Test basic binomial coefficient computation."""
        # Test with integer values
        result = self.bc.compute(5, 2)
        expected = 10  # 5! / (2! * 3!) = 120 / (2 * 6) = 10
        assert abs(result - expected) < 1e-10
        
        # Test with n = k
        result = self.bc.compute(5, 5)
        expected = 1
        assert abs(result - expected) < 1e-10
        
        # Test with k = 0
        result = self.bc.compute(5, 0)
        expected = 1
        assert abs(result - expected) < 1e-10
    
    def test_binomial_coefficients_fractional(self):
        """Test fractional binomial coefficient computation."""
        # Test with fractional n
        result = self.bc.compute(2.5, 1)
        expected = 2.5  # C(2.5, 1) = 2.5
        assert abs(result - expected) < 1e-6
        
        # Test with fractional n and k
        result = self.bc.compute(3.5, 1.5)
        # This should be computed using gamma functions
        assert result > 0
        assert np.isfinite(result)
    
    def test_binomial_coefficients_edge_cases(self):
        """Test edge cases for binomial coefficient computation."""
        # Test with very small values
        result = self.bc.compute(0.001, 0.0005)
        assert np.isfinite(result)
        
        # Test with large values
        result = self.bc.compute(100, 50)
        assert np.isfinite(result)
        assert result > 0
    
    def test_binomial_coefficients_negative_values(self):
        """Test binomial coefficient computation with negative values."""
        # Test with negative n
        result = self.bc.compute(-2, 1)
        # C(-2, 1) = (-2)! / (1! * (-3)!) = -2
        assert abs(result - (-2)) < 1e-6
        
        # Test with negative k
        result = self.bc.compute(5, -1)
        # This should be 0 by definition
        assert abs(result) < 1e-10
    
    def test_binomial_coefficients_array_input(self):
        """Test binomial coefficient computation with array inputs."""
        # Test with numpy arrays
        n_vals = np.array([1, 2, 3, 4, 5])
        k_vals = np.array([0, 1, 1, 2, 2])
        
        results = self.bc.compute(n_vals, k_vals)
        assert len(results) == len(n_vals)
        
        # Check some known values
        assert abs(results[0] - 1) < 1e-10  # C(1, 0) = 1
        assert abs(results[1] - 2) < 1e-10  # C(2, 1) = 2
        assert abs(results[2] - 3) < 1e-10  # C(3, 1) = 3
    
    def test_binomial_coefficients_jax_computation(self):
        """Test JAX computation when available."""
        # Test with JAX enabled
        bc_jax = BinomialCoefficients(use_jax=True)
        
        if hasattr(bc_jax, '_binomial_jax'):
            # Test basic computation
            result = bc_jax.compute(5, 2)
            expected = 10
            assert abs(result - expected) < 1e-6
            
            # Test with array inputs
            n_vals = np.array([1, 2, 3])
            k_vals = np.array([0, 1, 1])
            results = bc_jax.compute(n_vals, k_vals)
            assert len(results) == 3
    
    def test_binomial_coefficients_caching(self):
        """Test caching functionality."""
        # Test that repeated computations use cache
        result1 = self.bc.compute(10, 5)
        result2 = self.bc.compute(10, 5)
        
        assert abs(result1 - result2) < 1e-10
        
        # Test cache size limit
        bc_small_cache = BinomialCoefficients(cache_size=2)
        
        # Fill cache
        bc_small_cache.compute(1, 1)
        bc_small_cache.compute(2, 1)
        
        # This should evict the first entry
        bc_small_cache.compute(3, 1)
        
        # Should still work
        result = bc_small_cache.compute(3, 1)
        assert np.isfinite(result)
    
    def test_binomial_coefficients_error_handling(self):
        """Test error handling."""
        # Test with invalid inputs
        with pytest.raises((ValueError, TypeError)):
            self.bc.compute("invalid", 1)
        
        with pytest.raises((ValueError, TypeError)):
            self.bc.compute(1, "invalid")
    
    def test_binomial_coefficients_large_values(self):
        """Test binomial coefficient computation with large values."""
        # Test with moderately large values
        result = self.bc.compute(50, 25)
        assert np.isfinite(result)
        assert result > 0
        
        # Test with very large values
        result = self.bc.compute(1000, 500)
        assert np.isfinite(result)
        assert result > 0
    
    def test_binomial_coefficients_fractional_large(self):
        """Test fractional binomial coefficient computation with large values."""
        # Test with large fractional values
        result = self.bc.compute(100.5, 50.25)
        assert np.isfinite(result)
        assert result > 0
        
        # Test with very large fractional values
        result = self.bc.compute(1000.7, 500.3)
        assert np.isfinite(result)
        assert result > 0
    
    def test_binomial_coefficients_zero_values(self):
        """Test binomial coefficient computation with zero values."""
        # Test with zero n
        result = self.bc.compute(0, 0)
        expected = 1  # C(0, 0) = 1
        assert abs(result - expected) < 1e-10
        
        # Test with zero k
        result = self.bc.compute(5, 0)
        expected = 1  # C(5, 0) = 1
        assert abs(result - expected) < 1e-10
    
    def test_binomial_coefficients_one_values(self):
        """Test binomial coefficient computation with one values."""
        # Test with n = 1
        result = self.bc.compute(1, 0)
        expected = 1  # C(1, 0) = 1
        assert abs(result - expected) < 1e-10
        
        result = self.bc.compute(1, 1)
        expected = 1  # C(1, 1) = 1
        assert abs(result - expected) < 1e-10
        
        # Test with k = 1
        result = self.bc.compute(5, 1)
        expected = 5  # C(5, 1) = 5
        assert abs(result - expected) < 1e-10
    
    def test_binomial_coefficients_fractional_small(self):
        """Test fractional binomial coefficient computation with small values."""
        # Test with small fractional values
        result = self.bc.compute(0.5, 0.25)
        assert np.isfinite(result)
        assert result > 0
        
        # Test with very small fractional values
        result = self.bc.compute(0.001, 0.0005)
        assert np.isfinite(result)
        assert result > 0
    
    def test_binomial_coefficients_negative_fractional(self):
        """Test fractional binomial coefficient computation with negative values."""
        # Test with negative fractional n
        result = self.bc.compute(-0.5, 0.25)
        assert np.isfinite(result)
        
        # Test with negative fractional k
        result = self.bc.compute(0.5, -0.25)
        assert np.isfinite(result)


class TestBinomialCoefficientFunctions:
    """Test standalone binomial coefficient functions."""
    
    def test_binomial_coefficient_function(self):
        """Test binomial_coefficient function."""
        # Test basic computation
        result = binomial_coefficient(5, 2)
        expected = 10
        assert abs(result - expected) < 1e-10
        
        # Test with fractional values
        result = binomial_coefficient(2.5, 1)
        expected = 2.5
        assert abs(result - expected) < 1e-6
    
    def test_generalized_binomial_function(self):
        """Test generalized_binomial function."""
        # Test basic computation
        result = generalized_binomial(5, 2)
        expected = 10
        assert abs(result - expected) < 1e-10
        
        # Test with fractional values
        result = generalized_binomial(2.5, 1)
        expected = 2.5
        assert abs(result - expected) < 1e-6
        
        # Test with array inputs
        n_vals = np.array([1, 2, 3])
        k_vals = np.array([0, 1, 1])
        results = generalized_binomial(n_vals, k_vals)
        assert len(results) == 3
        
        # Check some known values
        assert abs(results[0] - 1) < 1e-10  # C(1, 0) = 1
        assert abs(results[1] - 2) < 1e-10  # C(2, 1) = 2
        assert abs(results[2] - 3) < 1e-10  # C(3, 1) = 3
    
    def test_function_edge_cases(self):
        """Test edge cases for standalone functions."""
        # Test with zero values
        result = binomial_coefficient(0, 0)
        expected = 1
        assert abs(result - expected) < 1e-10
        
        # Test with negative values
        result = binomial_coefficient(-2, 1)
        expected = -2
        assert abs(result - expected) < 1e-6
        
        # Test with large values
        result = generalized_binomial(100, 50)
        assert np.isfinite(result)
        assert result > 0


class TestBinomialCoefficientsAdvanced:
    """Advanced tests for binomial coefficients."""
    
    def test_binomial_coefficients_numba_optimization(self):
        """Test NUMBA optimization."""
        bc = BinomialCoefficients(use_numba=True)
        
        # Test that NUMBA optimization works
        result = bc.compute(50, 25)
        assert np.isfinite(result)
        assert result > 0
        
        # Test with array inputs
        n_vals = np.array([10, 20, 30])
        k_vals = np.array([5, 10, 15])
        results = bc.compute(n_vals, k_vals)
        assert len(results) == 3
        assert all(np.isfinite(results))
    
    def test_binomial_coefficients_performance(self):
        """Test performance with large computations."""
        bc = BinomialCoefficients()
        
        # Test with large values
        result = bc.compute(1000, 500)
        assert np.isfinite(result)
        assert result > 0
        
        # Test with very large values
        result = bc.compute(10000, 5000)
        assert np.isfinite(result)
        assert result > 0
    
    def test_binomial_coefficients_mixed_types(self):
        """Test binomial coefficient computation with mixed types."""
        bc = BinomialCoefficients()
        
        # Test with mixed integer and float
        result = bc.compute(5, 2.0)
        expected = 10
        assert abs(result - expected) < 1e-10
        
        # Test with mixed float and integer
        result = bc.compute(5.0, 2)
        expected = 10
        assert abs(result - expected) < 1e-10
    
    def test_binomial_coefficients_special_values(self):
        """Test binomial coefficient computation with special values."""
        bc = BinomialCoefficients()
        
        # Test with very small positive values
        result = bc.compute(1e-10, 1e-11)
        assert np.isfinite(result)
        
        # Test with very large values
        result = bc.compute(1e10, 1e9)
        assert np.isfinite(result)
        assert result > 0


if __name__ == "__main__":
    pytest.main([__file__])

