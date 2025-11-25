import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

"""
Comprehensive tests for special gamma_beta module.

This module tests all gamma and beta function functionality including
computation, optimization strategies, and edge cases.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import patch, MagicMock

from hpfracc.special.gamma_beta import (
    GammaFunction, BetaFunction, JAX_AVAILABLE
)


class TestGammaFunction:
    """Test GammaFunction class."""
    
    def test_gamma_function_initialization(self):
        """Test GammaFunction initialization."""
        gamma = GammaFunction()
        assert gamma.use_jax is False
        assert gamma.use_numba is True
    
    def test_gamma_function_initialization_with_jax(self):
        """Test GammaFunction initialization with JAX."""
        print(f"JAX_AVAILABLE in test_gamma_function_initialization_with_jax: {JAX_AVAILABLE}")
        gamma = GammaFunction(use_jax=True)
        if JAX_AVAILABLE:
            assert gamma.use_jax is True
            assert gamma.use_numba is True
            assert hasattr(gamma, '_gamma_jax')
        else:
            assert gamma.use_jax is False
            assert not hasattr(gamma, '_gamma_jax')
    
    def test_gamma_function_initialization_without_numba(self):
        """Test GammaFunction initialization without NUMBA."""
        gamma = GammaFunction(use_numba=False)
        assert gamma.use_jax is False
        assert gamma.use_numba is False
    
    def test_gamma_function_compute_scalar(self):
        """Test GammaFunction compute method with scalar inputs."""
        gamma = GammaFunction()
        
        # Test basic cases
        assert abs(gamma.compute(1.0) - 1.0) < 1e-10  # Γ(1) = 1
        assert abs(gamma.compute(2.0) - 1.0) < 1e-10  # Γ(2) = 1! = 1
        assert abs(gamma.compute(3.0) - 2.0) < 1e-10  # Γ(3) = 2! = 2
        assert abs(gamma.compute(4.0) - 6.0) < 1e-10  # Γ(4) = 3! = 6
    
    def test_gamma_function_compute_fractional(self):
        """Test GammaFunction compute method with fractional inputs."""
        gamma = GammaFunction()
        
        # Test fractional cases
        result = gamma.compute(0.5)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        assert result > 0
        
        result = gamma.compute(1.5)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        assert result > 0
    
    def test_gamma_function_compute_array(self):
        """Test GammaFunction compute method with array inputs."""
        gamma = GammaFunction()
        
        # Test array inputs
        z_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        results = gamma.compute(z_vals)
        
        assert isinstance(results, np.ndarray)
        assert results.shape == z_vals.shape
        assert not np.any(np.isnan(results))
        assert np.all(results > 0)
    
    def test_gamma_function_compute_with_jax(self):
        """Test GammaFunction compute method with JAX."""
        gamma = GammaFunction(use_jax=True)
        
        # Test scalar with JAX
        result = gamma.compute(2.0)
        # JAX returns arrays, so check for JAX array or convert to scalar
        if hasattr(result, 'item'):  # JAX array
            result_scalar = result.item()
            assert isinstance(result_scalar, (int, float, np.floating))
        else:
            assert isinstance(result, (int, float, np.floating))
        assert not np.isnan(float(result))
        
        # Test array with JAX
        z_vals = np.array([1.0, 2.0, 3.0, 4.0])
        results = gamma.compute(z_vals)
        assert isinstance(results, np.ndarray)
        assert results.shape == z_vals.shape
        assert not np.any(np.isnan(results))
    
    def test_gamma_function_compute_without_numba(self):
        """Test GammaFunction compute method without NUMBA."""
        gamma = GammaFunction(use_numba=False)
        
        # Test scalar without NUMBA
        result = gamma.compute(2.0)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        
        # Test array without NUMBA
        z_vals = np.array([1.0, 2.0, 3.0, 4.0])
        results = gamma.compute(z_vals)
        assert isinstance(results, np.ndarray)
        assert results.shape == z_vals.shape
        assert not np.any(np.isnan(results))
    
    def test_gamma_function_edge_cases(self):
        """Test GammaFunction edge cases."""
        gamma = GammaFunction()
        
        # Test edge cases
        result = gamma.compute(0.1)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        
        result = gamma.compute(10.0)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        assert result > 0
    
    def test_gamma_function_large_values(self):
        """Test GammaFunction with large values."""
        gamma = GammaFunction()
        
        # Test with larger values
        result = gamma.compute(20.0)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        assert result > 0
    
    def test_gamma_function_different_orders(self):
        """Test GammaFunction with different fractional orders."""
        gamma = GammaFunction()
        
        for z in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0, 2.5, 3.0]:
            result = gamma.compute(z)
            assert isinstance(result, (int, float, np.floating))
            # Type guard for integer results
            if not isinstance(result, (int, np.integer)):
                assert not np.isnan(result)
            assert result > 0


class TestBetaFunction:
    """Test BetaFunction class."""
    
    def test_beta_function_initialization(self):
        """Test BetaFunction initialization."""
        beta = BetaFunction()
        assert beta.use_jax is False
        assert beta.use_numba is True
    
    def test_beta_function_initialization_with_jax(self):
        """Test BetaFunction initialization with JAX."""
        print(f"JAX_AVAILABLE in test_beta_function_initialization_with_jax: {JAX_AVAILABLE}")
        beta = BetaFunction(use_jax=True)
        if JAX_AVAILABLE:
            assert beta.use_jax is True
            assert beta.use_numba is True
            assert hasattr(beta, '_beta_jax')
        else:
            assert beta.use_jax is False
            assert not hasattr(beta, '_beta_jax')
    
    def test_beta_function_initialization_without_numba(self):
        """Test BetaFunction initialization without NUMBA."""
        beta = BetaFunction(use_numba=False)
        assert beta.use_jax is False
        assert beta.use_numba is False
    
    def test_beta_function_compute_scalar(self):
        """Test BetaFunction compute method with scalar inputs."""
        beta = BetaFunction()
        
        # Test basic cases
        result = beta.compute(1.0, 1.0)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        assert result > 0
        
        result = beta.compute(2.0, 3.0)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        assert result > 0
    
    def test_beta_function_compute_fractional(self):
        """Test BetaFunction compute method with fractional inputs."""
        beta = BetaFunction()
        
        # Test fractional cases
        result = beta.compute(0.5, 0.5)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        assert result > 0
        
        result = beta.compute(1.5, 2.5)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        assert result > 0
    
    def test_beta_function_compute_array(self):
        """Test BetaFunction compute method with array inputs."""
        beta = BetaFunction()
        
        # Test array inputs
        a_vals = np.array([1.0, 2.0, 3.0, 4.0])
        b_vals = np.array([1.0, 1.0, 2.0, 2.0])
        results = beta.compute(a_vals, b_vals)
        
        assert isinstance(results, np.ndarray)
        assert results.shape == a_vals.shape
        assert not np.any(np.isnan(results))
        assert np.all(results > 0)
    
    def test_beta_function_compute_with_jax(self):
        """Test BetaFunction compute method with JAX."""
        beta = BetaFunction(use_jax=True)
        
        # Test scalar with JAX
        result = beta.compute(2.0, 3.0)
        # JAX returns arrays, so check for JAX array or convert to scalar
        if hasattr(result, 'item'):  # JAX array
            result_scalar = result.item()
            assert isinstance(result_scalar, (int, float, np.floating))
        else:
            assert isinstance(result, (int, float, np.floating))
        assert not np.isnan(float(result))
        
        # Test array with JAX
        a_vals = np.array([1.0, 2.0, 3.0])
        b_vals = np.array([1.0, 1.0, 2.0])
        results = beta.compute(a_vals, b_vals)
        assert isinstance(results, np.ndarray)
        assert results.shape == a_vals.shape
        assert not np.any(np.isnan(results))
    
    def test_beta_function_compute_without_numba(self):
        """Test BetaFunction compute method without NUMBA."""
        beta = BetaFunction(use_numba=False)
        
        # Test scalar without NUMBA
        result = beta.compute(2.0, 3.0)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        
        # Test array without NUMBA
        a_vals = np.array([1.0, 2.0, 3.0])
        b_vals = np.array([1.0, 1.0, 2.0])
        results = beta.compute(a_vals, b_vals)
        assert isinstance(results, np.ndarray)
        assert results.shape == a_vals.shape
        assert not np.any(np.isnan(results))
    
    def test_beta_function_edge_cases(self):
        """Test BetaFunction edge cases."""
        beta = BetaFunction()
        
        # Test edge cases
        result = beta.compute(0.1, 0.1)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        
        result = beta.compute(10.0, 10.0)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        assert result > 0
    
    def test_beta_function_large_values(self):
        """Test BetaFunction with large values."""
        beta = BetaFunction()
        
        # Test with larger values
        result = beta.compute(20.0, 15.0)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        assert result > 0
    
    def test_beta_function_different_orders(self):
        """Test BetaFunction with different fractional orders."""
        beta = BetaFunction()
        
        for a in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0]:
            for b in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0]:
                result = beta.compute(a, b)
                assert isinstance(result, (int, float, np.floating))
                # Type guard for integer results
                if not isinstance(result, (int, np.integer)):
                    assert not np.isnan(result)
                assert result > 0


class TestGammaBetaIntegration:
    """Test gamma and beta function integration scenarios."""
    
    def test_gamma_beta_workflow(self):
        """Test complete gamma-beta workflow."""
        gamma = GammaFunction(use_jax=True)
        beta = BetaFunction(use_jax=True)
        
        # Test gamma computation
        result1 = gamma.compute(2.0)
        # JAX returns arrays, so check for JAX array or convert to scalar
        if hasattr(result1, 'item'):  # JAX array
            result_scalar = result1.item()
            assert isinstance(result_scalar, (int, float, np.floating))
        else:
            assert isinstance(result1, (int, float, np.floating))
        
        # Test beta computation
        result2 = beta.compute(2.0, 3.0)
        # JAX returns arrays, so check for JAX array or convert to scalar
        if hasattr(result2, 'item'):  # JAX array
            result_scalar = result2.item()
            assert isinstance(result_scalar, (int, float, np.floating))
        else:
            assert isinstance(result2, (int, float, np.floating))
        
        # Test array computations
        z_vals = np.array([1.0, 2.0, 3.0, 4.0])
        gamma_results = gamma.compute(z_vals)
        assert isinstance(gamma_results, np.ndarray)
        
        a_vals = np.array([1.0, 2.0, 3.0, 4.0])
        b_vals = np.array([1.0, 1.0, 2.0, 2.0])
        beta_results = beta.compute(a_vals, b_vals)
        assert isinstance(beta_results, np.ndarray)
    
    def test_gamma_beta_consistency(self):
        """Test consistency between gamma and beta functions."""
        gamma = GammaFunction()
        beta = BetaFunction()
        
        # Test that B(a,b) = Γ(a)Γ(b)/Γ(a+b)
        a, b = 2.0, 3.0
        
        gamma_a = gamma.compute(a)
        gamma_b = gamma.compute(b)
        gamma_ab = gamma.compute(a + b)
        beta_ab = beta.compute(a, b)
        
        # Check consistency (within numerical precision)
        expected = gamma_a * gamma_b / gamma_ab
        assert abs(beta_ab - expected) < 1e-10
    
    def test_gamma_beta_performance(self):
        """Test gamma and beta function performance scenarios."""
        gamma = GammaFunction(cache_size=1000)
        beta = BetaFunction(cache_size=1000)
        
        # Test many computations
        for i in range(100):
            z = i * 0.1 + 0.1
            a = i * 0.1 + 0.1
            b = i * 0.1 + 0.2
            
            # Gamma computation
            result1 = gamma.compute(z)
            assert isinstance(result1, (int, float, np.floating))
            
            # Beta computation
            result2 = beta.compute(a, b)
            assert isinstance(result2, (int, float, np.floating))


class TestGammaBetaEdgeCases:
    """Test gamma and beta function edge cases and error handling."""
    
    def test_gamma_function_zero_input(self):
        """Test GammaFunction with zero input."""
        gamma = GammaFunction()
        
        # Test with very small positive number
        result = gamma.compute(1e-10)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
    
    def test_gamma_function_negative_input(self):
        """Test GammaFunction with negative input."""
        gamma = GammaFunction()
        
        # Test with negative input (should handle gracefully)
        result = gamma.compute(-0.5)
        assert isinstance(result, (int, float, np.floating))
        # May be NaN or complex, but should not crash
    
    def test_beta_function_zero_inputs(self):
        """Test BetaFunction with zero inputs."""
        beta = BetaFunction()
        
        # Test with very small positive numbers
        result = beta.compute(1e-10, 1e-10)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
    
    def test_beta_function_negative_inputs(self):
        """Test BetaFunction with negative inputs."""
        beta = BetaFunction()
        
        # Test with negative inputs (should handle gracefully)
        result = beta.compute(-0.5, -0.5)
        assert isinstance(result, (int, float, np.floating))
        # May be NaN or complex, but should not crash
    
    def test_gamma_function_very_large_input(self):
        """Test GammaFunction with very large input."""
        gamma = GammaFunction()
        
        # Test with very large input
        result = gamma.compute(100.0)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        assert result > 0
    
    def test_beta_function_very_large_inputs(self):
        """Test BetaFunction with very large inputs."""
        beta = BetaFunction()
        
        # Test with very large inputs
        result = beta.compute(100.0, 50.0)
        assert isinstance(result, (int, float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        assert result > 0
    
    def test_gamma_function_array_edge_cases(self):
        """Test GammaFunction with array edge cases."""
        gamma = GammaFunction()
        
        # Test with mixed array
        z_vals = np.array([0.1, 1.0, 2.0, 10.0, 100.0])
        results = gamma.compute(z_vals)
        assert isinstance(results, np.ndarray)
        assert results.shape == z_vals.shape
        assert not np.any(np.isnan(results))
    
    def test_beta_function_array_edge_cases(self):
        """Test BetaFunction with array edge cases."""
        beta = BetaFunction()
        
        # Test with mixed arrays
        a_vals = np.array([0.1, 1.0, 2.0, 10.0])
        b_vals = np.array([0.1, 1.0, 1.0, 5.0])
        results = beta.compute(a_vals, b_vals)
        assert isinstance(results, np.ndarray)
        assert results.shape == a_vals.shape
        assert not np.any(np.isnan(results))


class TestGammaBetaPerformance:
    """Test gamma and beta function performance scenarios."""
    
    def test_gamma_function_caching_performance(self):
        """Test GammaFunction caching performance."""
        gamma = GammaFunction(cache_size=1000)
        
        # First run - populate cache
        for i in range(100):
            result = gamma.compute(i * 0.1 + 0.1)
            assert isinstance(result, (int, float, np.floating))
        
        # Second run - should use cache
        for i in range(100):
            result = gamma.compute(i * 0.1 + 0.1)
            assert isinstance(result, (int, float, np.floating))
    
    def test_beta_function_caching_performance(self):
        """Test BetaFunction caching performance."""
        beta = BetaFunction(cache_size=1000)
        
        # First run - populate cache
        for i in range(50):
            result = beta.compute(i * 0.1 + 0.1, i * 0.1 + 0.2)
            assert isinstance(result, (int, float, np.floating))
        
        # Second run - should use cache
        for i in range(50):
            result = beta.compute(i * 0.1 + 0.1, i * 0.1 + 0.2)
            assert isinstance(result, (int, float, np.floating))
    
    def test_gamma_beta_memory_usage(self):
        """Test gamma and beta function memory usage."""
        gamma = GammaFunction(cache_size=100)
        beta = BetaFunction(cache_size=100)
        
        # Test that cache doesn't grow indefinitely
        initial_gamma_cache_size = len(gamma._cache) if hasattr(gamma, '_cache') else 0
        initial_beta_cache_size = len(beta._cache) if hasattr(beta, '_cache') else 0
        
        for i in range(200):  # More than cache size
            gamma.compute(i * 0.1 + 0.1)
            beta.compute(i * 0.1 + 0.1, i * 0.1 + 0.2)
        
        # Cache should be limited
        if hasattr(gamma, '_cache'):
            assert len(gamma._cache) <= gamma.cache_size
        if hasattr(beta, '_cache'):
            assert len(beta._cache) <= beta.cache_size


if __name__ == "__main__":
    pytest.main([__file__])

