"""
Comprehensive edge case and error handling tests for core modules.

This module provides extensive testing for edge cases, boundary conditions,
and error handling across all core fractional calculus modules.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from typing import Union, Callable

from hpfracc.core.definitions import FractionalOrder, DefinitionType
from hpfracc.core.derivatives import BaseFractionalDerivative
from hpfracc.core.fractional_implementations import (
    RiemannLiouvilleDerivative, CaputoDerivative, GrunwaldLetnikovDerivative
)
from hpfracc.core.integrals import (
    RiemannLiouvilleIntegral, CaputoIntegral, WeylIntegral, HadamardIntegral
)
from hpfracc.core.utilities import (
    validate_fractional_order, validate_function, validate_tensor_input
)


class TestFractionalOrderEdgeCases:
    """Test edge cases for FractionalOrder class."""
    
    def test_zero_alpha(self):
        """Test behavior with alpha = 0."""
        alpha = FractionalOrder(0.0)
        assert alpha.alpha == 0.0
        assert alpha.is_integer
        assert alpha.alpha == 0
    
    def test_negative_alpha(self):
        """Test behavior with negative alpha."""
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            FractionalOrder(-0.5)
    
    def test_infinite_alpha(self):
        """Test behavior with infinite alpha."""
        with pytest.raises(ValueError, match="Fractional order must be finite"):
            FractionalOrder(np.inf)
    
    def test_nan_alpha(self):
        """Test behavior with NaN alpha."""
        with pytest.raises(ValueError, match="Fractional order must be finite"):
            FractionalOrder(np.nan)
    
    def test_very_large_alpha(self):
        """Test behavior with very large alpha values."""
        alpha = FractionalOrder(1000.0)
        assert alpha.alpha == 1000.0
        assert alpha.is_integer  # 1000.0 is an integer
    
    def test_very_small_alpha(self):
        """Test behavior with very small alpha values."""
        alpha = FractionalOrder(1e-10)
        assert alpha.alpha == 1e-10
        assert not alpha.is_integer
    
    def test_integer_alpha(self):
        """Test behavior with integer alpha values."""
        alpha = FractionalOrder(2.0)
        assert alpha.alpha == 2.0
        assert alpha.is_integer
        assert alpha.alpha == 2
    
    def test_fractional_alpha(self):
        """Test behavior with fractional alpha values."""
        alpha = FractionalOrder(0.5)
        assert alpha.alpha == 0.5
        assert not alpha.is_integer
        assert alpha.alpha == 0.5


class TestDerivativeEdgeCases:
    """Test edge cases for fractional derivative implementations."""
    
    def test_riemann_liouville_zero_alpha(self):
        """Test Riemann-Liouville derivative with alpha = 0."""
        rl_deriv = RiemannLiouvilleDerivative(0.0)
        assert rl_deriv.alpha.alpha == 0.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl_deriv.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
    
    def test_caputo_alpha_one(self):
        """Test Caputo derivative with alpha = 1 (now valid)."""
        # Caputo now supports alpha = 1.0
        caputo = CaputoDerivative(1.0)
        assert caputo.alpha.alpha == 1.0
    
    def test_caputo_alpha_zero(self):
        """Test Caputo derivative with alpha = 0 (identity operation)."""
        # Alpha = 0 is mathematically valid (identity operation)
        caputo = CaputoDerivative(0.0)
        assert caputo.alpha.alpha == 0.0
    
    def test_grunwald_letnikov_negative_alpha(self):
        """Test Grünwald-Letnikov derivative with negative alpha."""
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            GrunwaldLetnikovDerivative(-0.5)
    
    def test_empty_input_arrays(self):
        """Test behavior with empty input arrays."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([])
        # Empty arrays should raise ValueError
        with pytest.raises(ValueError, match="zero-size array"):
            rl_deriv.compute(test_func, x_vals)
    
    def test_single_element_arrays(self):
        """Test behavior with single element arrays."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0])
        result = rl_deriv.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0  # May return more points due to internal discretization
    
    def test_constant_function(self):
        """Test behavior with constant functions."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        def const_func(x):
            return np.ones_like(x) * 5.0
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl_deriv.compute(const_func, x_vals)
        assert isinstance(result, np.ndarray)
    
    def test_linear_function(self):
        """Test behavior with linear functions."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        def linear_func(x):
            return 2.0 * x + 3.0
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl_deriv.compute(linear_func, x_vals)
        assert isinstance(result, np.ndarray)
    
    def test_very_small_alpha(self):
        """Test behavior with very small alpha values."""
        rl_deriv = RiemannLiouvilleDerivative(1e-6)
        assert rl_deriv.alpha.alpha == 1e-6
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl_deriv.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
    
    def test_very_large_alpha(self):
        """Test behavior with very large alpha values."""
        rl_deriv = RiemannLiouvilleDerivative(10.0)
        assert rl_deriv.alpha.alpha == 10.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl_deriv.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
    
    def test_discontinuous_function(self):
        """Test behavior with discontinuous functions."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        def disc_func(x):
            return np.where(x < 2.0, x**2, x**3)
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl_deriv.compute(disc_func, x_vals)
        assert isinstance(result, np.ndarray)
    
    def test_oscillatory_function(self):
        """Test behavior with oscillatory functions."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        def osc_func(x):
            return np.sin(10 * x)
        
        x_vals = np.array([0.0, 0.1, 0.2])
        result = rl_deriv.compute(osc_func, x_vals)
        assert isinstance(result, np.ndarray)
    
    def test_exponential_function(self):
        """Test behavior with exponential functions."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        def exp_func(x):
            return np.exp(x)
        
        x_vals = np.array([0.0, 1.0, 2.0])
        result = rl_deriv.compute(exp_func, x_vals)
        assert isinstance(result, np.ndarray)
    
    def test_logarithmic_function(self):
        """Test behavior with logarithmic functions."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        def log_func(x):
            return np.log(x + 1)
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl_deriv.compute(log_func, x_vals)
        assert isinstance(result, np.ndarray)
    
    def test_polynomial_function(self):
        """Test behavior with polynomial functions."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        def poly_func(x):
            return x**5 + 2*x**3 + x + 1
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl_deriv.compute(poly_func, x_vals)
        assert isinstance(result, np.ndarray)


class TestIntegralEdgeCases:
    """Test edge cases for fractional integral implementations."""
    
    def test_riemann_liouville_integral_zero_alpha(self):
        """Test Riemann-Liouville integral with alpha = 0."""
        rl_integral = RiemannLiouvilleIntegral(0.0)
        assert rl_integral.alpha.alpha == 0.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl_integral(test_func, x_vals)
        assert isinstance(result, np.ndarray)
    
    def test_caputo_integral_alpha_one(self):
        """Test Caputo integral with alpha = 1."""
        caputo_integral = CaputoIntegral(1.0)
        assert caputo_integral.alpha.alpha == 1.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        # Caputo integral for α ≥ 1 is now implemented (decomposes into integer and fractional parts)
        result = caputo_integral(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_weyl_integral_negative_alpha(self):
        """Test Weyl integral with negative alpha."""
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            WeylIntegral(-0.5)
    
    def test_hadamard_integral_large_alpha(self):
        """Test Hadamard integral with large alpha."""
        hadamard_integral = HadamardIntegral(5.0)
        assert hadamard_integral.alpha.alpha == 5.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([2.0, 3.0, 4.0])  # All x > 1 required
        result = hadamard_integral(test_func, x_vals)
        assert isinstance(result, np.ndarray)
    
    def test_integral_empty_input(self):
        """Test integral behavior with empty input arrays."""
        rl_integral = RiemannLiouvilleIntegral(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([])
        result = rl_integral(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_integral_constant_function(self):
        """Test integral behavior with constant functions."""
        rl_integral = RiemannLiouvilleIntegral(0.5)
        
        def const_func(x):
            return np.ones_like(x) * 3.0
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl_integral(const_func, x_vals)
        assert isinstance(result, np.ndarray)
    
    def test_integral_linear_function(self):
        """Test integral behavior with linear functions."""
        rl_integral = RiemannLiouvilleIntegral(0.5)
        
        def linear_func(x):
            return 2.0 * x + 1.0
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl_integral(linear_func, x_vals)
        assert isinstance(result, np.ndarray)


class TestUtilityEdgeCases:
    """Test edge cases for utility functions."""
    
    def test_validate_fractional_order_edge_cases(self):
        """Test validate_fractional_order with edge cases."""
        # Test with valid alpha
        alpha = validate_fractional_order(0.5)
        assert alpha.alpha == 0.5
        
        # Test with integer alpha
        alpha = validate_fractional_order(2)
        assert alpha.alpha == 2.0
        
        # Test with zero alpha
        alpha = validate_fractional_order(0.0)
        assert alpha.alpha == 0.0
    
    def test_validate_fractional_order_validation(self):
        """Test validate_fractional_order validation."""
        # Test valid alpha
        assert validate_fractional_order(0.5).alpha == 0.5
        
        # Test integer alpha
        assert validate_fractional_order(2).alpha == 2.0
        
        # Test zero alpha
        assert validate_fractional_order(0.0).alpha == 0.0
        
        # Test negative alpha
        with pytest.raises(ValueError):
            validate_fractional_order(-0.5)
        
        # Test infinite alpha
        with pytest.raises(ValueError):
            validate_fractional_order(np.inf)
        
        # Test NaN alpha
        with pytest.raises(ValueError):
            validate_fractional_order(np.nan)
    
    def test_validate_tensor_input_edge_cases(self):
        """Test validate_tensor_input with edge cases."""
        # Test valid inputs
        x_vals = np.array([1.0, 2.0, 3.0])
        assert validate_tensor_input(x_vals) is True
        
        # Test empty array
        x_vals = np.array([])
        assert validate_tensor_input(x_vals) is True
        
        # Test single element
        x_vals = np.array([1.0])
        assert validate_tensor_input(x_vals) is True
        
        # Test with NaN values
        x_vals = np.array([1.0, np.nan, 3.0])
        assert validate_tensor_input(x_vals) is False
        
        # Test with infinite values
        x_vals = np.array([1.0, np.inf, 3.0])
        assert validate_tensor_input(x_vals) is False
    
    def test_validate_function_edge_cases(self):
        """Test validate_function with edge cases."""
        def test_func(x):
            return x**2
        
        # Test with valid function
        assert validate_function(test_func, domain=(0, 10)) is True
        
        # Test with invalid function
        assert validate_function(None, domain=(0, 10)) is False
        
        # Test with non-callable
        assert validate_function("not_a_function", domain=(0, 10)) is False


class TestNumericalStability:
    """Test numerical stability and precision."""
    
    def test_high_precision_calculation(self):
        """Test calculations with high precision requirements."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        def test_func(x):
            return np.sin(x)
        
        x_vals = np.array([0.0, np.pi/4, np.pi/2])
        result = rl_deriv.compute(test_func, x_vals)
        
        # Check that result is finite
        assert np.all(np.isfinite(result))
        assert isinstance(result, np.ndarray)
    
    def test_very_small_values(self):
        """Test calculations with very small values."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        def test_func(x):
            return x**0.1  # Very small power
        
        x_vals = np.array([1e-10, 1e-8, 1e-6])
        result = rl_deriv.compute(test_func, x_vals)
        
        assert np.all(np.isfinite(result))
        assert isinstance(result, np.ndarray)
    
    def test_very_large_values(self):
        """Test calculations with very large values."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        def test_func(x):
            return x**10  # Large power
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl_deriv.compute(test_func, x_vals)
        
        assert np.all(np.isfinite(result))
        assert isinstance(result, np.ndarray)
    
    def test_near_boundary_values(self):
        """Test calculations near boundary values."""
        # Test alpha very close to 0
        rl_deriv = RiemannLiouvilleDerivative(1e-10)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl_deriv.compute(test_func, x_vals)
        
        assert np.all(np.isfinite(result))
        assert isinstance(result, np.ndarray)
    
    def test_near_integer_values(self):
        """Test calculations near integer alpha values."""
        # Test alpha very close to 1
        rl_deriv = RiemannLiouvilleDerivative(0.999999)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl_deriv.compute(test_func, x_vals)
        
        assert np.all(np.isfinite(result))
        assert isinstance(result, np.ndarray)


class TestErrorHandling:
    """Test comprehensive error handling."""
    
    def test_invalid_function_input(self):
        """Test behavior with invalid function inputs."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        # Test with None function
        with pytest.raises((TypeError, AttributeError)):
            rl_deriv.compute(None, np.array([1.0, 2.0, 3.0]))
        
        # Test with non-callable function - should raise TypeError or AttributeError
        with pytest.raises((TypeError, AttributeError, ValueError)):
            rl_deriv.compute("not_a_function", np.array([1.0, 2.0, 3.0]))
    
    def test_invalid_x_input(self):
        """Test behavior with invalid x inputs."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        def test_func(x):
            return x**2
        
        # Test with None x
        with pytest.raises((TypeError, AttributeError)):
            rl_deriv.compute(test_func, None)
        
        # Test with non-array x
        with pytest.raises((TypeError, AttributeError)):
            rl_deriv.compute(test_func, "not_an_array")
    
    def test_mismatched_array_sizes(self):
        """Test behavior with mismatched array sizes."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        def test_func(x):
            return x**2
        
        # Test with mismatched f_values and x_values
        f_values = np.array([1.0, 4.0])  # 2 elements
        x_values = np.array([1.0, 2.0, 3.0])  # 3 elements
        
        # This should either work or raise a clear error
        try:
            result = rl_deriv.compute_numerical(f_values, x_values)
            assert isinstance(result, np.ndarray)
        except (ValueError, IndexError) as e:
            # Expected error for mismatched sizes
            assert "size" in str(e).lower() or "length" in str(e).lower() or "shape" in str(e).lower()
    
    def test_memory_intensive_calculation(self):
        """Test behavior with memory-intensive calculations."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        def test_func(x):
            return x**2
        
        # Test with large array
        x_vals = np.linspace(0, 10, 10000)
        result = rl_deriv.compute(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_concurrent_calculations(self):
        """Test behavior with concurrent calculations."""
        import threading
        import time
        
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        results = []
        
        def compute_derivative():
            result = rl_deriv.compute(test_func, x_vals)
            results.append(result)
        
        # Run multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=compute_derivative)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all results are valid
        assert len(results) == 3
        for result in results:
            assert isinstance(result, np.ndarray)
            assert len(result) > 0
