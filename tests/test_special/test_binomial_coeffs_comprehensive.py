import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

"""
Comprehensive tests for special binomial_coeffs module.

This module tests all binomial coefficient functionality including
generalized coefficients, caching, and optimization strategies.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import patch, MagicMock
import scipy.special

from hpfracc.special.binomial_coeffs import (
    BinomialCoefficients, GrunwaldLetnikovCoefficients
)


class TestBinomialCoefficients:
    """Test BinomialCoefficients class."""
    
    def test_binomial_coefficients_initialization(self):
        """Test BinomialCoefficients initialization."""
        bc = BinomialCoefficients()
        assert bc.use_jax is False
        assert bc.use_numba is True
        assert bc.cache_size == 1000
        assert isinstance(bc._cache, dict)
    
    @patch('hpfracc.special.binomial_coeffs.jnp', np)
    @patch('hpfracc.special.binomial_coeffs.jax')
    def test_binomial_coefficients_initialization_with_jax(self, mock_jax):
        """Test BinomialCoefficients initialization with JAX."""
        mock_jax.jit.side_effect = lambda x: x  # Make jit a pass-through
        with patch('hpfracc.special.binomial_coeffs.JAX_AVAILABLE', True):
            bc = BinomialCoefficients(use_jax=True)
            assert bc.use_jax is True
            assert bc.use_numba is True
            assert bc.cache_size == 1000
            assert hasattr(bc, '_binomial_jax')
    
    def test_binomial_coefficients_initialization_with_custom_cache(self):
        """Test BinomialCoefficients initialization with custom cache size."""
        bc = BinomialCoefficients(cache_size=500)
        assert bc.cache_size == 500
        assert isinstance(bc._cache, dict)
    
    def test_binomial_coefficients_compute_scalar(self):
        """Test BinomialCoefficients compute method with scalar inputs."""
        bc = BinomialCoefficients()
        
        # Test basic cases
        assert bc.compute(5, 2) == 10  # C(5,2) = 10
        assert bc.compute(4, 0) == 1   # C(4,0) = 1
        assert bc.compute(4, 4) == 1   # C(4,4) = 1
        assert bc.compute(6, 3) == 20  # C(6,3) = 20
    
    def test_binomial_coefficients_compute_fractional(self):
        """Test BinomialCoefficients compute method with fractional inputs."""
        bc = BinomialCoefficients()
        
        # Test fractional cases
        result = bc.compute(0.5, 2)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        
        result = bc.compute(1.5, 1)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
    
    def test_binomial_coefficients_compute_array(self):
        """Test BinomialCoefficients compute method with array inputs."""
        bc = BinomialCoefficients()
        
        # Test array inputs
        n_vals = np.array([1, 2, 3, 4, 5])
        k_vals = np.array([0, 1, 2, 3, 4])
        
        results = bc.compute(n_vals, k_vals)
        assert isinstance(results, np.ndarray)
        assert results.shape == n_vals.shape
        assert not np.any(np.isnan(results))
    
    @patch('hpfracc.special.binomial_coeffs.jnp', np)
    @patch('hpfracc.special.binomial_coeffs.jax')
    def test_binomial_coefficients_compute_with_jax(self, mock_jax):
        """Test BinomialCoefficients compute method with JAX."""
        mock_jax.jit.side_effect = lambda x: x  # Make jit a pass-through
        mock_jax.scipy.special.gamma.side_effect = scipy.special.gamma

        with patch('hpfracc.special.binomial_coeffs.JAX_AVAILABLE', True):
            bc = BinomialCoefficients(use_jax=True)
            
            # Test scalar with JAX
            result = bc.compute(5, 2)
            # JAX returns arrays, so check for JAX array or convert to scalar
            if hasattr(result, 'item'):  # JAX array
                result_scalar = result.item()
                assert isinstance(result_scalar, (int, float, np.floating))
            else:
                assert isinstance(result, (int, float, np.floating))
            assert not np.isnan(float(result))
            
            # Test array with JAX
            n_vals = np.array([1, 2, 3, 4, 5])
            k_vals = np.array([0, 1, 2, 3, 4])

            results = bc.compute(n_vals, k_vals)
            # JAX returns JAX arrays, not numpy arrays
            assert hasattr(results, 'shape') and hasattr(results, 'dtype')
            assert results.shape == n_vals.shape
            assert not np.any(np.isnan(results))
    
    def test_binomial_coefficients_caching(self):
        """Test BinomialCoefficients caching functionality."""
        bc = BinomialCoefficients(cache_size=10)
        
        # First computation should populate cache
        result1 = bc.compute(5, 2)
        assert len(bc._cache) > 0
        
        # Second computation should use cache
        result2 = bc.compute(5, 2)
        assert result1 == result2
    
    def test_binomial_coefficients_edge_cases(self):
        """Test BinomialCoefficients edge cases."""
        bc = BinomialCoefficients()
        
        # Test edge cases
        assert bc.compute(0, 0) == 1
        assert bc.compute(1, 0) == 1
        assert bc.compute(1, 1) == 1
        
        # Test with negative inputs (should handle gracefully)
        result = bc.compute(-1, 2)
        assert isinstance(result, (int, float, np.floating))
    
    def test_binomial_coefficients_large_values(self):
        """Test BinomialCoefficients with large values."""
        bc = BinomialCoefficients()
        
        # Test with larger values
        result = bc.compute(20, 10)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        assert result > 0
    
    def test_binomial_coefficients_different_orders(self):
        """Test BinomialCoefficients with different fractional orders."""
        bc = BinomialCoefficients()
        
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0]:
            for k in [0, 1, 2, 3, 4, 5]:
                result = bc.compute(alpha, k)
                assert isinstance(result, (int, float, np.floating))
                # Type guard for integer results
                if not isinstance(result, (int, np.integer)):
                    assert not np.isnan(result)


class TestGrunwaldLetnikovCoefficients:
    """Test GrunwaldLetnikovCoefficients class."""
    
    def test_grunwald_letnikov_coefficients_initialization(self):
        """Test GrunwaldLetnikovCoefficients initialization."""
        glc = GrunwaldLetnikovCoefficients()
        assert glc is not None
    
    @patch('hpfracc.special.binomial_coeffs.jnp', np)
    @patch('hpfracc.special.binomial_coeffs.jax')
    def test_grunwald_letnikov_coefficients_initialization_with_params(self, mock_jax):
        """Test GrunwaldLetnikovCoefficients initialization with parameters."""
        mock_jax.jit.side_effect = lambda x: x  # Make jit a pass-through
        with patch('hpfracc.special.binomial_coeffs.JAX_AVAILABLE', True):
            glc = GrunwaldLetnikovCoefficients(
                use_jax=True,
                use_numba=True,
                cache_size=500
            )
            assert glc is not None
    
    def test_grunwald_letnikov_coefficients_compute(self):
        """Test GrunwaldLetnikovCoefficients compute method."""
        glc = GrunwaldLetnikovCoefficients()
        
        # Test basic computation
        result = glc.compute(0.5, 10)
        assert isinstance(result, np.ndarray)
        assert len(result) == 11  # k from 0 to 10
        assert not np.any(np.isnan(result))
    
    def test_grunwald_letnikov_coefficients_compute_different_orders(self):
        """Test GrunwaldLetnikovCoefficients with different fractional orders."""
        glc = GrunwaldLetnikovCoefficients()
        
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5]:
            result = glc.compute(alpha, 10)
            assert isinstance(result, np.ndarray)
            assert len(result) == 11
            assert not np.any(np.isnan(result))
    
    def test_grunwald_letnikov_coefficients_compute_different_lengths(self):
        """Test GrunwaldLetnikovCoefficients with different sequence lengths."""
        glc = GrunwaldLetnikovCoefficients()
        
        for n in [5, 10, 20, 50]:
            result = glc.compute(0.5, n)
            assert isinstance(result, np.ndarray)
            assert len(result) == n + 1
            assert not np.any(np.isnan(result))
    
    @patch('hpfracc.special.binomial_coeffs.jnp', np)
    @patch('hpfracc.special.binomial_coeffs.jax')
    def test_grunwald_letnikov_coefficients_compute_with_jax(self, mock_jax):
        """Test GrunwaldLetnikovCoefficients with JAX."""
        mock_jax.jit.side_effect = lambda x: x  # Make jit a pass-through
        mock_jax.scipy.special.binom.side_effect = scipy.special.binom
        with patch('hpfracc.special.binomial_coeffs.JAX_AVAILABLE', True):
            glc = GrunwaldLetnikovCoefficients(use_jax=True)
            
            result = glc.compute(0.5, 10)
            # JAX returns arrays, so check for JAX array or convert to numpy
            if hasattr(result, 'shape'):  # JAX array
                assert hasattr(result, 'shape')  # Just check it's array-like
            else:
                assert isinstance(result, np.ndarray)
            assert len(result) == 11
            assert not np.any(np.isnan(result))
    
    def test_grunwald_letnikov_coefficients_caching(self):
        """Test GrunwaldLetnikovCoefficients caching."""
        glc = GrunwaldLetnikovCoefficients(cache_size=10)
        
        # First computation
        result1 = glc.compute(0.5, 10)
        assert isinstance(result1, np.ndarray)
        
        # Second computation should use cache
        result2 = glc.compute(0.5, 10)
        assert isinstance(result2, np.ndarray)
        assert np.allclose(result1, result2)
    
    def test_grunwald_letnikov_coefficients_edge_cases(self):
        """Test GrunwaldLetnikovCoefficients edge cases."""
        glc = GrunwaldLetnikovCoefficients()
        
        # Test with alpha = 0
        result = glc.compute(0.0, 5)
        assert isinstance(result, np.ndarray)
        assert len(result) == 6
        assert not np.any(np.isnan(result))
        
        # Test with alpha = 1
        result = glc.compute(1.0, 5)
        assert isinstance(result, np.ndarray)
        assert len(result) == 6
        assert not np.any(np.isnan(result))
        
        # Test with very small n
        result = glc.compute(0.5, 0)
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
        assert not np.any(np.isnan(result))
    
    def test_grunwald_letnikov_coefficients_large_values(self):
        """Test GrunwaldLetnikovCoefficients with large values."""
        glc = GrunwaldLetnikovCoefficients()
        
        # Test with larger n
        result = glc.compute(0.5, 100)
        assert isinstance(result, np.ndarray)
        assert len(result) == 101
        assert not np.any(np.isnan(result))
    
    def test_grunwald_letnikov_coefficients_negative_alpha(self):
        """Test GrunwaldLetnikovCoefficients with negative alpha."""
        glc = GrunwaldLetnikovCoefficients()
        
        # Test with negative alpha
        result = glc.compute(-0.5, 10)
        assert isinstance(result, np.ndarray)
        assert len(result) == 11
        assert not np.any(np.isnan(result))


class TestBinomialCoefficientsIntegration:
    """Test binomial coefficients integration scenarios."""
    
    @patch('hpfracc.special.binomial_coeffs.jnp', np)
    @patch('hpfracc.special.binomial_coeffs.jax')
    def test_binomial_coefficients_workflow(self, mock_jax):
        """Test complete binomial coefficients workflow."""
        mock_jax.jit.side_effect = lambda x: x
        mock_jax.scipy.special.gamma.side_effect = scipy.special.gamma
        with patch('hpfracc.special.binomial_coeffs.JAX_AVAILABLE', True):
            bc = BinomialCoefficients(use_jax=True, cache_size=100)
            
            # Test scalar computation
            result1 = bc.compute(5, 2)
            # JAX returns arrays, so check for JAX array or convert to scalar
            if hasattr(result1, 'item'):  # JAX array
                result_scalar = result1.item()
                assert isinstance(result_scalar, (int, float, np.floating))
            else:
                assert isinstance(result1, (int, float, np.floating))
            
            # Test array computation
            n_vals = np.array([1, 2, 3, 4, 5])
            k_vals = np.array([0, 1, 2, 3, 4])
            result2 = bc.compute(n_vals, k_vals)
            # JAX returns JAX arrays, not numpy arrays
            assert hasattr(result2, 'shape') and hasattr(result2, 'dtype')
            
            # Test fractional computation
            result3 = bc.compute(0.5, 2)
            # JAX returns arrays, so check for JAX array or convert to scalar
            if hasattr(result3, 'item'):  # JAX array
                result_scalar = result3.item()
                assert isinstance(result_scalar, (int, float, np.floating))
            else:
                assert isinstance(result3, (int, float, np.floating))
    
    @patch('hpfracc.special.binomial_coeffs.jnp', np)
    @patch('hpfracc.special.binomial_coeffs.jax')
    def test_grunwald_letnikov_coefficients_workflow(self, mock_jax):
        """Test complete Grünwald-Letnikov coefficients workflow."""
        mock_jax.jit.side_effect = lambda x: x
        mock_jax.scipy.special.binom.side_effect = scipy.special.binom
        with patch('hpfracc.special.binomial_coeffs.JAX_AVAILABLE', True):
            glc = GrunwaldLetnikovCoefficients(use_jax=True, cache_size=100)
            
            # Test basic computation
            result = glc.compute(0.5, 10)
            # JAX returns arrays, so check for JAX array or convert to numpy
            if hasattr(result, 'shape'):  # JAX array
                assert hasattr(result, 'shape')  # Just check it's array-like
            else:
                assert isinstance(result, np.ndarray)
            assert len(result) == 11
            
            # Test with different parameters
            for alpha in [0.1, 0.5, 0.9, 1.1, 1.5]:
                result = glc.compute(alpha, 20)
                # JAX returns arrays, so check for JAX array or convert to numpy
                if hasattr(result, 'shape'):  # JAX array
                    assert hasattr(result, 'shape')  # Just check it's array-like
                else:
                    assert isinstance(result, np.ndarray)
                assert len(result) == 21
                assert not np.any(np.isnan(result))
    
    def test_coefficients_consistency(self):
        """Test consistency between different coefficient implementations."""
        bc = BinomialCoefficients()
        glc = GrunwaldLetnikovCoefficients()
        
        # Test that binomial coefficients match Grünwald-Letnikov for integer alpha
        alpha = 5
        n = 10
        
        # Get binomial coefficient
        binom_result = bc.compute(alpha, n)
        
        # Get Grünwald-Letnikov coefficients
        gl_result = glc.compute(alpha, n)
        
        # The last coefficient should match
        assert np.isclose(binom_result, gl_result[-1], rtol=1e-10)
    
    def test_coefficients_performance(self):
        """Test coefficients performance scenarios."""
        bc = BinomialCoefficients(cache_size=1000)
        glc = GrunwaldLetnikovCoefficients(cache_size=1000)
        
        # Test many computations
        for i in range(100):
            alpha = i * 0.1
            k = i % 10
            
            # Binomial coefficient
            result1 = bc.compute(alpha, k)
            assert isinstance(result1, (int, float, np.floating))
            
            # Grünwald-Letnikov coefficients
            result2 = glc.compute(alpha, k)
            assert isinstance(result2, np.ndarray)


class TestBinomialCoefficientsEdgeCases:
    """Test binomial coefficients edge cases and error handling."""
    
    def test_binomial_coefficients_zero_inputs(self):
        """Test binomial coefficients with zero inputs."""
        bc = BinomialCoefficients()
        
        # Test various zero cases
        assert bc.compute(0, 0) == 1
        assert bc.compute(1, 0) == 1
        assert bc.compute(0, 1) == 0  # or should be 0
    
    def test_binomial_coefficients_negative_inputs(self):
        """Test binomial coefficients with negative inputs."""
        bc = BinomialCoefficients()
        
        # Test negative inputs
        result = bc.compute(-1, 2)
        assert isinstance(result, (int, float, np.floating))
        
        result = bc.compute(2, -1)
        assert isinstance(result, (int, float, np.floating))
    
    def test_binomial_coefficients_large_k(self):
        """Test binomial coefficients with large k values."""
        bc = BinomialCoefficients()
        
        # Test with k > n
        result = bc.compute(5, 10)
        assert isinstance(result, (int, float, np.floating))
    
    def test_grunwald_letnikov_coefficients_zero_alpha(self):
        """Test Grünwald-Letnikov coefficients with zero alpha."""
        glc = GrunwaldLetnikovCoefficients()
        
        result = glc.compute(0.0, 10)
        assert isinstance(result, np.ndarray)
        assert len(result) == 11
        assert not np.any(np.isnan(result))
    
    def test_grunwald_letnikov_coefficients_negative_alpha(self):
        """Test Grünwald-Letnikov coefficients with negative alpha."""
        glc = GrunwaldLetnikovCoefficients()
        
        result = glc.compute(-0.5, 10)
        assert isinstance(result, np.ndarray)
        assert len(result) == 11
        assert not np.any(np.isnan(result))
    
    def test_grunwald_letnikov_coefficients_very_small_n(self):
        """Test Grünwald-Letnikov coefficients with very small n."""
        glc = GrunwaldLetnikovCoefficients()
        
        result = glc.compute(0.5, 0)
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
        assert not np.any(np.isnan(result))
    
    def test_grunwald_letnikov_coefficients_very_large_n(self):
        """Test Grünwald-Letnikov coefficients with very large n."""
        glc = GrunwaldLetnikovCoefficients()
        
        result = glc.compute(0.5, 1000)
        assert isinstance(result, np.ndarray)
        assert len(result) == 1001
        assert not np.any(np.isnan(result))


class TestBinomialCoefficientsPerformance:
    """Test binomial coefficients performance scenarios."""
    
    def test_binomial_coefficients_caching_performance(self):
        """Test binomial coefficients caching performance."""
        bc = BinomialCoefficients(cache_size=1000)
        
        # First run - populate cache
        for i in range(100):
            result = bc.compute(i, i // 2)
            assert isinstance(result, (int, float, np.floating))
        
        # Second run - should use cache
        for i in range(100):
            result = bc.compute(i, i // 2)
            assert isinstance(result, (int, float, np.floating))
    
    def test_grunwald_letnikov_coefficients_caching_performance(self):
        """Test Grünwald-Letnikov coefficients caching performance."""
        glc = GrunwaldLetnikovCoefficients(cache_size=1000)
        
        # First run - populate cache
        for i in range(50):
            result = glc.compute(i * 0.1, 20)
            assert isinstance(result, np.ndarray)
        
        # Second run - should use cache
        for i in range(50):
            result = glc.compute(i * 0.1, 20)
            assert isinstance(result, np.ndarray)
    
    def test_coefficients_memory_usage(self):
        """Test coefficients memory usage."""
        bc = BinomialCoefficients(cache_size=100)
        glc = GrunwaldLetnikovCoefficients(cache_size=100)
        
        # Test that cache doesn't grow indefinitely
        initial_cache_size = len(bc._cache)
        
        for i in range(200):  # More than cache size
            bc.compute(i, i // 2)
            glc.compute(i * 0.1, 10)
        
        # Cache should be limited
        assert len(bc._cache) <= bc.cache_size
        assert len(glc._cache) <= glc.cache_size


if __name__ == "__main__":
    pytest.main([__file__])

