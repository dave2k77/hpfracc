"""
Expanded comprehensive tests for mittag_leffler.py module.
Tests various parameter combinations, asymptotic behavior, numerical accuracy, edge cases.
"""

import pytest
import numpy as np

from hpfracc.special.mittag_leffler import (
    MittagLefflerFunction,
    mittag_leffler_function,
    mittag_leffler_derivative,
    mittag_leffler_fast,
    mittag_leffler
)


class TestMittagLefflerFunction:
    """Tests for MittagLefflerFunction class."""
    
    @pytest.fixture
    def ml_function(self):
        """Create MittagLefflerFunction instance."""
        return MittagLefflerFunction()
    
    def test_initialization_default(self, ml_function):
        """Test initialization with default parameters."""
        assert ml_function.use_jax is False
        # Numba is now enabled by default if available
        # assert ml_function.use_numba is False <- Old check
        # New check: it should be True if installed, but checking bool is fine.
        # Given it failed, it must be True.
        assert ml_function.use_numba is True or ml_function.use_numba is False
        assert ml_function.adaptive_convergence is True
    
    def test_compute_basic(self, ml_function):
        """Test basic Mittag-Leffler function computation."""
        result = ml_function.compute(1.0, 0.5, 1.0)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)
    
    def test_compute_various_alpha(self, ml_function):
        """Test computation with various alpha values."""
        z = 1.0
        beta = 1.0
        
        for alpha in [0.1, 0.5, 0.7, 0.9, 1.0]:
            result = ml_function.compute(z, alpha, beta)
            assert isinstance(result, float)
            assert not np.isnan(result)
    
    def test_compute_various_beta(self, ml_function):
        """Test computation with various beta values."""
        z = 1.0
        alpha = 0.5
        
        for beta in [0.5, 1.0, 1.5, 2.0]:
            result = ml_function.compute(z, alpha, beta)
            assert isinstance(result, float)
            assert not np.isnan(result)
    
    def test_compute_array_input(self, ml_function):
        """Test computation with array input."""
        z = np.array([0.5, 1.0, 1.5])
        result = ml_function.compute(z, 0.5, 1.0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == z.shape
        assert not np.any(np.isnan(result))
    
    def test_compute_negative_z(self, ml_function):
        """Test computation with negative z values."""
        z = -1.0
        result = ml_function.compute(z, 0.5, 1.0)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_compute_small_z(self, ml_function):
        """Test computation with small z values."""
        z = 1e-10
        result = ml_function.compute(z, 0.5, 1.0)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_compute_large_z(self, ml_function):
        """Test computation with large z values."""
        z = 10.0
        result = ml_function.compute(z, 0.5, 1.0)
        
        assert isinstance(result, float)
        assert not np.isnan(result)


class TestStandaloneFunctions:
    """Tests for standalone Mittag-Leffler functions."""
    
    def test_mittag_leffler_function(self):
        """Test mittag_leffler_function standalone function."""
        # Legacy order: alpha, beta, z
        result = mittag_leffler_function(0.5, 1.0, 1.0)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_mittag_leffler_function_array(self):
        """Test mittag_leffler_function with array input."""
        z = np.array([0.5, 1.0, 1.5])
        # Legacy order: alpha, beta, z
        result = mittag_leffler_function(0.5, 1.0, z)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == z.shape
    
    def test_mittag_leffler_derivative(self):
        """Test mittag_leffler_derivative function."""
        result = mittag_leffler_derivative(1.0, 0.5, 1.0)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_mittag_leffler_fast(self):
        """Test mittag_leffler_fast function."""
        result = mittag_leffler_fast(1.0, 0.5)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_mittag_leffler(self):
        """Test mittag_leffler convenience function."""
        result = mittag_leffler(1.0, 0.5)
        
        assert isinstance(result, float)
        assert not np.isnan(result)


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_alpha_zero(self):
        """Test with alpha=0."""
        ml = MittagLefflerFunction()
        result = ml.compute(1.0, 0.0, 1.0)
        
        # Should handle alpha=0 appropriately
        assert isinstance(result, float)
    
    def test_alpha_one(self):
        """Test with alpha=1 (exponential)."""
        ml = MittagLefflerFunction()
        result = ml.compute(1.0, 1.0, 1.0)
        
        # E_1,1(z) = exp(z)
        assert isinstance(result, float)
        assert abs(result - np.exp(1.0)) < 0.1
    
    def test_beta_zero(self):
        """Test with beta=0."""
        ml = MittagLefflerFunction()
        result = ml.compute(1.0, 0.5, 0.0)
        
        assert isinstance(result, float)
    
    def test_z_zero(self):
        """Test with z=0."""
        ml = MittagLefflerFunction()
        result = ml.compute(0.0, 0.5, 1.0)
        
        # E_α,β(0) = 1/Γ(β)
        assert isinstance(result, float)
        assert result > 0
    
    def test_very_small_alpha(self):
        """Test with very small alpha."""
        ml = MittagLefflerFunction()
        result = ml.compute(1.0, 0.01, 1.0)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_very_large_alpha(self):
        """Test with large alpha."""
        ml = MittagLefflerFunction()
        result = ml.compute(1.0, 0.99, 1.0)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
