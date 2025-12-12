"""
Expanded comprehensive tests for gamma_beta.py module.
Tests edge cases, special values, numerical stability, large arguments.
"""

import pytest
import numpy as np
from scipy.special import gamma as scipy_gamma, beta as scipy_beta

from hpfracc.special.gamma_beta import (
    gamma,
    beta,
    log_gamma,
    log_beta,
)


class TestGammaFunction:
    """Tests for gamma function."""
    
    def test_gamma_positive_integers(self):
        """Test gamma for positive integers."""
        assert abs(gamma(1) - 1.0) < 1e-10
        assert abs(gamma(2) - 1.0) < 1e-10
        assert abs(gamma(3) - 2.0) < 1e-10
        assert abs(gamma(4) - 6.0) < 1e-10
    
    def test_gamma_half_integers(self):
        """Test gamma for half-integers."""
        assert abs(gamma(0.5) - np.sqrt(np.pi)) < 1e-10
        assert abs(gamma(1.5) - 0.5 * np.sqrt(np.pi)) < 1e-10
    
    def test_gamma_fractional(self):
        """Test gamma for fractional values."""
        result = gamma(0.7)
        expected = scipy_gamma(0.7)
        assert abs(result - expected) < 1e-10
    
    def test_gamma_array(self):
        """Test gamma with array input."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = gamma(x)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape
        assert np.all(result > 0)
    
    def test_gamma_large_values(self):
        """Test gamma with large values."""
        result = gamma(10.0)
        expected = scipy_gamma(10.0)
        assert abs(result - expected) < 1e-6
    
    def test_gamma_small_values(self):
        """Test gamma with small values."""
        result = gamma(0.1)
        expected = scipy_gamma(0.1)
        assert abs(result - expected) < 1e-6


class TestBetaFunction:
    """Tests for beta function."""
    
    def test_beta_known_values(self):
        """Test beta with known values."""
        assert abs(beta(1, 1) - 1.0) < 1e-10
        assert abs(beta(2, 2) - 1.0/6.0) < 1e-10
    
    def test_beta_symmetry(self):
        """Test beta function symmetry."""
        assert abs(beta(2, 3) - beta(3, 2)) < 1e-10
        assert abs(beta(0.5, 0.7) - beta(0.7, 0.5)) < 1e-10
    
    def test_beta_fractional(self):
        """Test beta for fractional values."""
        result = beta(0.5, 0.7)
        expected = scipy_beta(0.5, 0.7)
        assert abs(result - expected) < 1e-10
    
    def test_beta_array(self):
        """Test beta with array input."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 1.0, 1.0])
        result = beta(a, b)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == a.shape


class TestLogarithmicFunctions:
    """Tests for logarithmic gamma and beta functions."""
    
    def test_log_gamma(self):
        """Test log gamma function."""
        result = log_gamma(5.0)
        expected = np.log(scipy_gamma(5.0))
        assert abs(result - expected) < 1e-10
    
    def test_log_gamma_large(self):
        """Test log gamma with large values."""
        result = log_gamma(100.0)
        expected = np.log(scipy_gamma(100.0))
        assert abs(result - expected) < 1e-6
    
    def test_log_beta(self):
        """Test log beta function."""
        result = log_beta(2.0, 3.0)
        expected = np.log(scipy_beta(2.0, 3.0))
        assert abs(result - expected) < 1e-10


class TestIncompleteFunctions:
    """Tests for incomplete gamma and beta functions."""
    
    @pytest.mark.skip(reason="incomplete_gamma function not yet implemented in module")
    def test_incomplete_gamma(self):
        """Test incomplete gamma function."""
        # result = incomplete_gamma(0.5, 1.0)
        # assert isinstance(result, float)
        # assert result > 0
        pass
    
    @pytest.mark.skip(reason="incomplete_beta function not yet implemented in module")
    def test_incomplete_beta(self):
        """Test incomplete beta function."""
        # result = incomplete_beta(0.5, 0.7, 0.5)
        # assert isinstance(result, float)
        # assert 0 <= result <= 1
        pass
    
    @pytest.mark.skip(reason="regularized_incomplete_gamma function not yet implemented in module")
    def test_regularized_incomplete_gamma(self):
        """Test regularized incomplete gamma."""
        # result = regularized_incomplete_gamma(0.5, 1.0)
        # assert isinstance(result, float)
        # assert 0 <= result <= 1
        pass
    
    @pytest.mark.skip(reason="regularized_incomplete_beta function not yet implemented in module")
    def test_regularized_incomplete_beta(self):
        """Test regularized incomplete beta."""
        # result = regularized_incomplete_beta(0.5, 0.7, 0.5)
        # assert isinstance(result, float)
        # assert 0 <= result <= 1
        pass


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_gamma_one(self):
        """Test gamma(1) = 1."""
        assert abs(gamma(1) - 1.0) < 1e-10
    
    def test_gamma_very_small(self):
        """Test gamma with very small values."""
        result = gamma(0.01)
        assert isinstance(result, float)
        assert result > 0
    
    def test_beta_symmetric(self):
        """Test beta function symmetry property."""
        a, b = 2.5, 3.7
        assert abs(beta(a, b) - beta(b, a)) < 1e-10
    
    def test_beta_one_one(self):
        """Test beta(1, 1) = 1."""
        assert abs(beta(1, 1) - 1.0) < 1e-10
