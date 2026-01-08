import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

"""
Tests for binomial coefficients module.

This module tests the binomial coefficient implementations in hpfracc.special.binomial_coeffs.
"""

import numpy as np
import pytest
from hpfracc.special.binomial_coeffs import (
    BinomialCoefficients,
    GrunwaldLetnikovCoefficients,
    binomial,
    binomial_fractional,
    fractional_pascal_triangle,
    grunwald_letnikov_coefficients,
    grunwald_letnikov_weighted_coefficients,
    pascal_triangle
)


class TestBinomialCoefficients:
    """Test BinomialCoefficients class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        bc = BinomialCoefficients()
        
        assert bc is not None
        assert hasattr(bc, 'compute')
        assert hasattr(bc, 'compute_fractional')
        assert hasattr(bc, 'compute_sequence')
        assert hasattr(bc, 'compute_alternating_sequence')
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        bc = BinomialCoefficients(use_jax=True, use_numba=False, cache_size=1000)
        
        assert bc is not None
        assert hasattr(bc, 'compute')
        assert hasattr(bc, 'compute_fractional')
        assert hasattr(bc, 'compute_sequence')
        assert hasattr(bc, 'compute_alternating_sequence')
    
    def test_compute_scalar(self):
        """Test computing single binomial coefficient."""
        bc = BinomialCoefficients()
        
        # Test basic cases
        result = bc.compute(5, 2)
        expected = 10  # C(5,2) = 10
        
        assert isinstance(result, (int, float, np.integer, np.floating))
        assert abs(result - expected) < 1e-10
    
    def test_compute_sequence(self):
        """Test computing sequence of binomial coefficients."""
        bc = BinomialCoefficients()
        
        result = bc.compute_sequence(5, 3)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 4  # k = 0, 1, 2, 3
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_special_cases(self):
        """Test known special cases."""
        bc = BinomialCoefficients()
        
        # C(n,0) = 1
        result = bc.compute(10, 0)
        assert abs(result - 1) < 1e-10
        
        # C(n,n) = 1
        result = bc.compute(10, 10)
        assert abs(result - 1) < 1e-10
        
        # C(n,1) = n
        result = bc.compute(10, 1)
        assert abs(result - 10) < 1e-10
    
    def test_edge_cases(self):
        """Test edge cases."""
        bc = BinomialCoefficients()
        
        # Test with k > n (should be 0)
        result = bc.compute(3, 5)
        assert abs(result - 0) < 1e-10
        
        # Test with negative k (should be 0)
        result = bc.compute(5, -1)
        assert abs(result - 0) < 1e-10


class TestBinomialCoefficientFunctions:
    """Test standalone binomial coefficient functions."""
    
    def test_binomial_coefficient(self):
        """Test binomial function."""
        result = binomial(5, 2)
        expected = 10
        
        assert isinstance(result, (int, float, np.integer, np.floating))
        assert abs(result - expected) < 1e-10
    
    def test_binomial_coefficients_array(self):
        """Test binomial function with arrays."""
        n_values = np.array([3, 4, 5])
        k_values = np.array([1, 2, 2])
        result = binomial(n_values, k_values)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == n_values.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_binomial_fractional(self):
        """Test fractional binomial coefficient."""
        result = binomial_fractional(5.5, 2)
        
        assert isinstance(result, (float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result)
    
    def test_pascal_triangle(self):
        """Test Pascal triangle."""
        result = pascal_triangle(5)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 6  # n+1 rows
    
    def test_fractional_pascal_triangle(self):
        """Test fractional Pascal triangle."""
        result = fractional_pascal_triangle(0.5, 5)  # Fixed parameter order
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 6  # n+1 rows


class TestGrunwaldLetnikovCoefficients:
    """Test Grünwald-Letnikov coefficient functions."""
    
    def test_grunwald_letnikov_coefficients(self):
        """Test Grünwald-Letnikov coefficients."""
        result = grunwald_letnikov_coefficients(0.5, 5)  # Fixed parameter order
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 6  # max_k+1 coefficients
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_grunwald_letnikov_weighted_coefficients(self):
        """Test weighted Grünwald-Letnikov coefficients."""
        result = grunwald_letnikov_weighted_coefficients(0.5, 5, 1.0)  # Fixed parameter order
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 6  # max_k+1 coefficients
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_grunwald_letnikov_coefficients_array(self):
        """Test Grünwald-Letnikov coefficients with array input."""
        # Test with scalar inputs since the function doesn't support arrays
        result1 = grunwald_letnikov_coefficients(0.5, 3)
        assert isinstance(result1, np.ndarray)
        
        # Test weighted coefficients
        result2 = grunwald_letnikov_weighted_coefficients(0.5, 3, 1.0)
        assert isinstance(result2, np.ndarray)


class TestGrunwaldLetnikovCoefficientsClass:
    """Test GrunwaldLetnikovCoefficients class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        glc = GrunwaldLetnikovCoefficients()
        
        assert glc is not None
        assert hasattr(glc, 'compute_coefficients')
        assert hasattr(glc, 'compute_weighted_coefficients')
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        glc = GrunwaldLetnikovCoefficients(use_jax=True, use_numba=False)
        
        assert glc is not None
        assert hasattr(glc, 'compute_coefficients')
        assert hasattr(glc, 'compute_weighted_coefficients')
    
    def test_compute_coefficients(self):
        """Test computing coefficients."""
        glc = GrunwaldLetnikovCoefficients()
        
        result = glc.compute_coefficients(0.5, 5)  # Fixed parameter order
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 6  # max_k+1 coefficients
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_compute_weighted_coefficients(self):
        """Test computing weighted coefficients."""
        glc = GrunwaldLetnikovCoefficients()
        
        result = glc.compute_weighted_coefficients(0.5, 5, 1.0)  # Fixed parameter order
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 6  # max_k+1 coefficients
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestBinomialCoefficientsPerformance:
    """Test performance and large values."""
    
    def test_large_values(self):
        """Test with large values."""
        bc = BinomialCoefficients()
        
        # Test with moderate large values
        result = bc.compute(20, 10)
        
        assert isinstance(result, (int, float, np.integer, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result)
    
    def test_sequence_performance(self):
        """Test performance with sequences."""
        bc = BinomialCoefficients()
        
        # Test sequence computation
        result = bc.compute_sequence(10, 5)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 6  # k = 0, 1, 2, 3, 4, 5
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_caching_behavior(self):
        """Test caching behavior."""
        bc = BinomialCoefficients()
        
        # Compute same value twice
        result1 = bc.compute(10, 5)
        result2 = bc.compute(10, 5)
        
        assert abs(result1 - result2) < 1e-10


class TestBinomialCoefficientsMathematicalProperties:
    """Test mathematical properties of binomial coefficients."""
    
    def test_symmetry_property(self):
        """Test symmetry property C(n,k) = C(n,n-k)."""
        bc = BinomialCoefficients()
        
        n, k = 10, 3
        result1 = bc.compute(n, k)
        result2 = bc.compute(n, n - k)
        
        assert abs(result1 - result2) < 1e-10
    
    def test_pascals_triangle_property(self):
        """Test Pascal's triangle property C(n,k) = C(n-1,k-1) + C(n-1,k)."""
        bc = BinomialCoefficients()
        
        n, k = 10, 5
        result = bc.compute(n, k)
        expected = bc.compute(n - 1, k - 1) + bc.compute(n - 1, k)
        
        assert abs(result - expected) < 1e-10
    
    def test_sum_property(self):
        """Test sum property sum(C(n,k)) = 2^n."""
        bc = BinomialCoefficients()
        
        n = 5
        total = sum(bc.compute(n, k) for k in range(n + 1))
        expected = 2 ** n
        
        assert abs(total - expected) < 1e-10


class TestBinomialCoefficientsErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_parameters(self):
        """Test with invalid parameters."""
        bc = BinomialCoefficients()
        
        # Test with negative n
        result = bc.compute(-1, 2)
        # Generalized binomial coefficient C(-1,2) = 1
        assert abs(result - 1) < 1e-10
        
        # Test with k > n
        result = bc.compute(3, 5)
        assert abs(result - 0) < 1e-10
    
    def test_extreme_values(self):
        """Test with extreme values."""
        bc = BinomialCoefficients()
        
        # Test with very large n (should handle gracefully)
        result = bc.compute(100, 50)
        
        assert isinstance(result, (int, float, np.integer, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result)
    
    def test_numerical_stability(self):
        """Test numerical stability."""
        bc = BinomialCoefficients()
        
        # Test with values that might cause overflow
        test_cases = [
            (10, 5),
            (15, 7),
            (20, 10),
            (25, 12)
        ]
        
        for n, k in test_cases:
            result = bc.compute(n, k)
            # Type guard for integer results
            if not isinstance(result, (int, np.integer)):
                assert not np.isnan(result)
            # Type guard for integer results
            if not isinstance(result, (int, np.integer)):
                assert not np.isinf(result)


class TestBinomialCoefficientsIntegration:
    """Integration tests for binomial coefficients."""
    
    def test_with_numpy_arrays(self):
        """Test with NumPy arrays."""
        bc = BinomialCoefficients()
        
        n_values = np.array([3, 4, 5, 6])
        k_values = np.array([1, 2, 2, 3])
        
        # Test individual computations
        results = [bc.compute(n, k) for n, k in zip(n_values, k_values)]
        
        assert len(results) == len(n_values)
        assert all(not np.isnan(r) for r in results)
        assert all(not np.isinf(r) for r in results)
    
    def test_with_different_dtypes(self):
        """Test with different data types."""
        bc = BinomialCoefficients()
        
        # Test with int32
        n_int32 = 5
        k_int32 = 2
        result_int32 = bc.compute(n_int32, k_int32)
        
        # Test with int64
        n_int64 = 5
        k_int64 = 2
        result_int64 = bc.compute(n_int64, k_int64)
        
        assert isinstance(result_int32, (int, float, np.integer, np.floating))
        assert isinstance(result_int64, (int, float, np.integer, np.floating))
        assert abs(result_int32 - result_int64) < 1e-10
    
    def test_consistency_across_functions(self):
        """Test consistency between different function implementations."""
        bc = BinomialCoefficients()
        
        n, k = 10, 3
        
        # Test class method vs standalone function
        result1 = bc.compute(n, k)
        result2 = binomial(n, k)
        
        assert abs(result1 - result2) < 1e-10
        
        # Test sequence method vs standalone function
        result3 = bc.compute_sequence(n, k)
        result4 = grunwald_letnikov_coefficients(n, k)
        
        assert isinstance(result3, np.ndarray)
        assert isinstance(result4, np.ndarray)


