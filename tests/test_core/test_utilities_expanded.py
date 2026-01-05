"""
Expanded comprehensive tests for utilities.py module.
Tests helper functions, mathematical utilities, validation functions, edge cases.
"""

import pytest
import numpy as np
from scipy.special import gamma

from hpfracc.core.utilities import (
    factorial_fractional,
    binomial_coefficient,
    pochhammer_symbol,
    hypergeometric_series,
    bessel_function_first_kind,
    modified_bessel_function_first_kind,
    validate_fractional_order,
    validate_function,
    validate_tensor_input,
    safe_divide,
    check_numerical_stability,
    vectorize_function,
    normalize_array,
    fractional_power,
    fractional_exponential,
    get_default_precision,
    set_default_precision,
    get_available_methods,
    get_method_properties
)
from hpfracc.core.definitions import FractionalOrder


class TestMathematicalUtilities:
    """Tests for mathematical utility functions."""
    
    def test_factorial_fractional_integer(self):
        """Test factorial for integer values."""
        assert factorial_fractional(5) == 120.0
        assert factorial_fractional(0) == 1.0
        assert factorial_fractional(1) == 1.0
    
    def test_factorial_fractional_float(self):
        """Test factorial for fractional values."""
        result = factorial_fractional(2.5)
        expected = gamma(3.5)
        assert abs(result - expected) < 1e-10
    
    def test_factorial_fractional_negative(self):
        """Test factorial with negative value."""
        with pytest.raises(ValueError):
            factorial_fractional(-1)
    
    def test_factorial_fractional_large(self):
        """Test factorial with very large value."""
        with pytest.raises(OverflowError):
            factorial_fractional(1e7)
    
    def test_binomial_coefficient_integer(self):
        """Test binomial coefficient for integers."""
        assert abs(binomial_coefficient(5, 2) - 10.0) < 1e-10
        assert abs(binomial_coefficient(10, 0) - 1.0) < 1e-10
        assert abs(binomial_coefficient(10, 10) - 1.0) < 1e-10
    
    def test_binomial_coefficient_float(self):
        """Test binomial coefficient for floats."""
        result = binomial_coefficient(5.5, 2.5)
        expected = gamma(6.5) / (gamma(3.5) * gamma(3.0))
        assert abs(result - expected) < 1e-10
    
    def test_binomial_coefficient_negative_k(self):
        """Test binomial coefficient with negative k."""
        with pytest.raises(ValueError, match="k must be non-negative"):
            binomial_coefficient(5, -1)
    
    def test_pochhammer_symbol(self):
        """Test Pochhammer symbol computation."""
        assert abs(pochhammer_symbol(2.0, 3) - 24.0) < 1e-10  # 2 * 3 * 4
        assert abs(pochhammer_symbol(1.0, 0) - 1.0) < 1e-10
    
    def test_hypergeometric_series(self):
        """Test hypergeometric series computation."""
        result = hypergeometric_series(1.0, 1.0, 1.0, 0.5)
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_bessel_function_first_kind(self):
        """Test Bessel function of first kind."""
        result = bessel_function_first_kind(0.5, 1.0)
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_modified_bessel_function_first_kind(self):
        """Test modified Bessel function of first kind."""
        result = modified_bessel_function_first_kind(0.5, 1.0)
        assert isinstance(result, float)
        assert not np.isnan(result)


class TestValidationFunctions:
    """Tests for validation functions."""
    
    def test_validate_fractional_order_valid(self):
        """Test validating valid fractional order."""
        result = validate_fractional_order(0.5)
        assert result is True
    
    def test_validate_fractional_order_float(self):
        """Test validating fractional order as float."""
        result = validate_fractional_order(0.7)
        assert result is True
    
    def test_validate_fractional_order_fractional_order_object(self):
        """Test validating FractionalOrder object."""
        alpha = FractionalOrder(0.5)
        result = validate_fractional_order(alpha)
        assert result is True
    
    def test_validate_fractional_order_negative(self):
        """Test validating negative fractional order."""
        with pytest.raises(ValueError):
            validate_fractional_order(-0.5)
    
    def test_validate_fractional_order_too_large(self):
        """Test validating fractional order that's too large."""
        with pytest.raises(ValueError):
            validate_fractional_order(10.0)
    
    def test_validate_function(self):
        """Test validating function."""
        def f(x):
            return x ** 2
        
        result = validate_function(f, domain=(0, 1))
        assert result is True
    
    def test_validate_function_invalid(self):
        """Test validating invalid function."""
        # Function that raises error
        def f(x):
            if x < 0:
                raise ValueError("Negative x")
            return x ** 2
        
        result = validate_function(f, domain=(-1, 1))
        assert result is False
    
    def test_validate_tensor_input_numpy(self):
        """Test validating numpy array input."""
        x = np.array([1, 2, 3])
        result = validate_tensor_input(x)
        assert result is True
    
    def test_validate_tensor_input_torch(self):
        """Test validating torch tensor input."""
        try:
            import torch
            x = torch.tensor([1, 2, 3])
            result = validate_tensor_input(x)
            assert result is True
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_validate_tensor_input_invalid(self):
        """Test validating invalid tensor input."""
        x = [1, 2, 3]  # List, not array
        result = validate_tensor_input(x)
        assert result is False


class TestNumericalUtilities:
    """Tests for numerical utility functions."""
    
    def test_safe_divide_normal(self):
        """Test safe division with normal values."""
        result = safe_divide(10.0, 2.0)
        assert abs(result - 5.0) < 1e-10
    
    def test_safe_divide_by_zero(self):
        """Test safe division by zero."""
        result = safe_divide(10.0, 0.0)
        assert result == 0.0 or np.isinf(result) or np.isnan(result)
    
    def test_safe_divide_small_denominator(self):
        """Test safe division with very small denominator."""
        result = safe_divide(1.0, 1e-15)
        assert isinstance(result, (int, float, np.ndarray))
    
    def test_check_numerical_stability_stable(self):
        """Test checking numerical stability for stable values."""
        x = np.array([1.0, 2.0, 3.0])
        result = check_numerical_stability(x)
        assert result is True
    
    def test_check_numerical_stability_nan(self):
        """Test checking numerical stability with NaN."""
        x = np.array([1.0, np.nan, 3.0])
        result = check_numerical_stability(x)
        assert result is False
    
    def test_check_numerical_stability_inf(self):
        """Test checking numerical stability with inf."""
        x = np.array([1.0, np.inf, 3.0])
        result = check_numerical_stability(x)
        assert result is False
    
    def test_vectorize_function(self):
        """Test vectorizing a function."""
        def f(x):
            return x ** 2
        
        vectorized = vectorize_function(f, vectorize=True)
        
        x = np.array([1, 2, 3])
        result = vectorized(x)
        
        expected = np.array([1, 4, 9])
        np.testing.assert_array_equal(result, expected)
    
    def test_normalize_array(self):
        """Test normalizing array."""
        x = np.array([1, 2, 3, 4, 5])
        normalized = normalize_array(x, method="standard")
        
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == x.shape
        assert abs(np.mean(normalized)) < 1e-10  # Mean should be ~0
    
    def test_normalize_array_minmax(self):
        """Test normalizing array with min-max method."""
        x = np.array([1, 2, 3, 4, 5])
        normalized = normalize_array(x, method="minmax")
        
        assert isinstance(normalized, np.ndarray)
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 1


class TestFractionalFunctions:
    """Tests for fractional mathematical functions."""
    
    def test_fractional_power(self):
        """Test fractional power computation."""
        result = fractional_power(4.0, 0.5)
        assert abs(result - 2.0) < 1e-10
    
    def test_fractional_power_array(self):
        """Test fractional power with array input."""
        x = np.array([1, 4, 9])
        result = fractional_power(x, 0.5)
        
        expected = np.array([1, 2, 3])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_fractional_exponential(self):
        """Test fractional exponential computation."""
        result = fractional_exponential(1.0, 0.5)
        assert isinstance(result, (int, float, np.ndarray))
        assert not np.isnan(result)
    
    def test_fractional_exponential_array(self):
        """Test fractional exponential with array input."""
        x = np.array([0.5, 1.0, 1.5])
        result = fractional_exponential(x, 0.5)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape
        assert not np.any(np.isnan(result))


class TestConfigurationFunctions:
    """Tests for configuration functions."""
    
    def test_get_default_precision(self):
        """Test getting default precision."""
        precision = get_default_precision()
        assert isinstance(precision, int)
        assert precision > 0
    
    def test_set_default_precision(self):
        """Test setting default precision."""
        original = get_default_precision()
        
        set_default_precision(15)
        assert get_default_precision() == 15
        
        # Restore
        set_default_precision(original)
    
    def test_get_available_methods(self):
        """Test getting available methods."""
        methods = get_available_methods()
        assert isinstance(methods, list)
        assert len(methods) > 0
    
    def test_get_method_properties(self):
        """Test getting method properties."""
        methods = get_available_methods()
        if len(methods) > 0:
            properties = get_method_properties(methods[0])
            assert isinstance(properties, dict)


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_factorial_fractional_zero(self):
        """Test factorial of zero."""
        assert factorial_fractional(0) == 1.0
    
    def test_binomial_coefficient_k_zero(self):
        """Test binomial coefficient with k=0."""
        assert abs(binomial_coefficient(5, 0) - 1.0) < 1e-10
    
    def test_pochhammer_symbol_n_zero(self):
        """Test Pochhammer symbol with n=0."""
        assert abs(pochhammer_symbol(5.0, 0) - 1.0) < 1e-10
    
    def test_safe_divide_nan_numerator(self):
        """Test safe division with NaN numerator."""
        result = safe_divide(np.nan, 2.0)
        assert np.isnan(result) or result == 0.0
    
    def test_normalize_array_constant(self):
        """Test normalizing constant array."""
        x = np.array([5.0, 5.0, 5.0])
        normalized = normalize_array(x, method="standard")
        
        # Should handle constant array
        assert isinstance(normalized, np.ndarray)
    
    def test_fractional_power_negative_base(self):
        """Test fractional power with negative base."""
        result = fractional_power(-4.0, 0.5)
        # May return complex or handle appropriately
        assert isinstance(result, (int, float, complex, np.ndarray))





