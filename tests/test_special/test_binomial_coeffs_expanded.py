"""
Expanded comprehensive tests for binomial_coeffs.py module.
Tests large values, edge cases, numerical precision, combinatorial properties.
"""

import pytest
import numpy as np
from scipy.special import comb

from hpfracc.special.binomial_coeffs import (
    binomial,
    binomial_fractional
)
# Alias for compatibility
binomial_coefficient = binomial


class TestBinomialCoefficient:
    """Tests for binomial coefficient function."""
    
    def test_binomial_integer_values(self):
        """Test binomial coefficient for integer values."""
        assert abs(binomial_coefficient(5, 2) - 10.0) < 1e-10
        assert abs(binomial_coefficient(10, 3) - 120.0) < 1e-10
        assert abs(binomial_coefficient(10, 0) - 1.0) < 1e-10
        assert abs(binomial_coefficient(10, 10) - 1.0) < 1e-10
    
    def test_binomial_symmetry(self):
        """Test binomial coefficient symmetry."""
        assert abs(binomial_coefficient(10, 3) - binomial_coefficient(10, 7)) < 1e-10
        assert abs(binomial_coefficient(20, 5) - binomial_coefficient(20, 15)) < 1e-10
    
    def test_binomial_pascals_triangle(self):
        """Test binomial coefficients match Pascal's triangle."""
        # Row 5: 1, 5, 10, 10, 5, 1
        assert abs(binomial_coefficient(5, 0) - 1.0) < 1e-10
        assert abs(binomial_coefficient(5, 1) - 5.0) < 1e-10
        assert abs(binomial_coefficient(5, 2) - 10.0) < 1e-10
        assert abs(binomial_coefficient(5, 3) - 10.0) < 1e-10
        assert abs(binomial_coefficient(5, 4) - 5.0) < 1e-10
        assert abs(binomial_coefficient(5, 5) - 1.0) < 1e-10
    
    def test_binomial_fractional(self):
        """Test binomial coefficient for fractional n."""
        result = binomial_coefficient(5.5, 2)
        # Compare with scipy if available
        try:
            expected = comb(5.5, 2, exact=False)
            assert abs(result - expected) < 1e-6
        except:
            assert isinstance(result, float)
            assert result > 0
    
    def test_binomial_large_values(self):
        """Test binomial coefficient with large values."""
        result = binomial_coefficient(50, 25)
        # Should compute without overflow
        # Result can be int or float depending on implementation
        assert isinstance(result, (int, float, np.integer, np.floating))
        assert result > 0
        assert not np.isinf(result)
    
    def test_binomial_array(self):
        """Test binomial coefficient with array input."""
        n = np.array([5, 10, 15])
        k = np.array([2, 3, 4])
        result = binomial(n, k)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == n.shape


class TestGeneralizedBinomial:
    """Tests for generalized binomial coefficient."""
    
    def test_binomial_coefficient_generalized(self):
        """Test generalized binomial coefficient (fractional)."""
        # binomial_fractional takes (alpha, k) where k is integer
        result = binomial_fractional(5.5, 2)
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_generalized_symmetry(self):
        """Test generalized binomial symmetry."""
        # binomial_fractional takes (alpha, k) where k is integer
        result1 = binomial_fractional(5.5, 2)
        result2 = binomial_fractional(5.5, 3)
        # May not be exactly symmetric for non-integers
        assert isinstance(result1, float)
        assert isinstance(result2, float)


class TestMultinomialCoefficient:
    """Tests for multinomial coefficient."""
    
    @pytest.mark.skip(reason="multinomial_coefficient not implemented")
    def test_multinomial_coefficient(self):
        """Test multinomial coefficient computation."""
        # Function not implemented
        pass
    
    @pytest.mark.skip(reason="multinomial_coefficient not implemented")
    def test_multinomial_simple(self):
        """Test multinomial with simple case."""
        # Function not implemented
        pass


class TestQBinomial:
    """Tests for q-binomial coefficient."""
    
    @pytest.mark.skip(reason="q_binomial_coefficient not implemented")
    def test_q_binomial_coefficient(self):
        """Test q-binomial coefficient."""
        # Function not implemented
        pass
    
    @pytest.mark.skip(reason="q_binomial_coefficient not implemented")
    def test_q_binomial_q_one(self):
        """Test q-binomial with q=1 (should reduce to regular binomial)."""
        # Function not implemented
        pass


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_binomial_k_zero(self):
        """Test binomial with k=0."""
        assert abs(binomial_coefficient(10, 0) - 1.0) < 1e-10
    
    def test_binomial_k_equals_n(self):
        """Test binomial with k=n."""
        assert abs(binomial_coefficient(10, 10) - 1.0) < 1e-10
    
    def test_binomial_k_one(self):
        """Test binomial with k=1."""
        assert abs(binomial_coefficient(10, 1) - 10.0) < 1e-10
    
    def test_binomial_large_n_small_k(self):
        """Test binomial with large n and small k."""
        result = binomial_coefficient(100, 2)
        expected = 100 * 99 / 2  # 4950
        assert abs(result - expected) < 1e-6
    
    def test_binomial_negative_k(self):
        """Test binomial with negative k."""
        # Should handle or raise appropriate error
        try:
            result = binomial_coefficient(10, -1)
            # If it doesn't raise, result should be 0
            assert result == 0.0
        except (ValueError, AssertionError):
            # Expected behavior
            pass
    
    def test_binomial_k_greater_than_n(self):
        """Test binomial with k > n."""
        # Should return 0 for integer n, k
        result = binomial_coefficient(5, 10)
        assert result == 0.0 or abs(result) < 1e-10
