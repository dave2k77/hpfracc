"""
Expanded comprehensive tests for integrals.py module.
Tests all integral types (RL, Caputo, Weyl, Hadamard, Miller-Ross, Marchaud), edge cases, numerical accuracy.
"""

import pytest
import numpy as np
from scipy.special import gamma

from hpfracc.core.integrals import (
    FractionalIntegral,
    RiemannLiouvilleIntegral,
    CaputoIntegral,
    WeylIntegral,
    HadamardIntegral,
    MillerRossIntegral,
    MarchaudIntegral,
    FractionalIntegralFactory,
    create_fractional_integral,
    create_fractional_integral_factory,
    analytical_fractional_integral,
    trapezoidal_fractional_integral,
    simpson_fractional_integral,
    fractional_integral_properties,
    validate_fractional_integral
)
from hpfracc.core.definitions import FractionalOrder


class TestRiemannLiouvilleIntegralExpanded:
    """Expanded tests for RiemannLiouvilleIntegral."""
    
    @pytest.fixture
    def rl_integral(self):
        """Create Riemann-Liouville integral instance."""
        return RiemannLiouvilleIntegral(0.5)
    
    def test_compute_power_function(self, rl_integral):
        """Test computing integral of power function."""
        def f(x):
            return x ** 2
        
        x = 1.0
        result = rl_integral(f, x)
        
        assert isinstance(result, float)
        assert result >= 0
    
    def test_compute_constant_function(self, rl_integral):
        """Test computing integral of constant function."""
        def f(x):
            return 1.0
        
        x = 1.0
        result = rl_integral(f, x)
        
        # I^α 1 = x^α / Γ(α+1)
        expected = (x ** 0.5) / gamma(1.5)
        assert abs(result - expected) < 0.1  # Allow some numerical error
    
    def test_compute_array(self, rl_integral):
        """Test computing integral for array input."""
        def f(x):
            return x
        
        x = np.array([0.5, 1.0, 1.5])
        result = rl_integral(f, x)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape
        assert np.all(result >= 0)


class TestCaputoIntegralExpanded:
    """Expanded tests for CaputoIntegral."""
    
    @pytest.fixture
    def caputo_integral(self):
        """Create Caputo integral instance."""
        return CaputoIntegral(0.5)
    
    def test_compute_power_function(self, caputo_integral):
        """Test computing Caputo integral of power function."""
        def f(x):
            return x ** 2
        
        x = 1.0
        result = caputo_integral(f, x)
        
        assert isinstance(result, float)
        assert result >= 0
    
    def test_compute_constant_function(self, caputo_integral):
        """Test computing Caputo integral of constant function."""
        def f(x):
            return 1.0
        
        x = 1.0
        result = caputo_integral(f, x)
        
        assert isinstance(result, float)
        assert result >= 0


class TestWeylIntegralExpanded:
    """Expanded tests for WeylIntegral."""
    
    @pytest.fixture
    def weyl_integral(self):
        """Create Weyl integral instance."""
        return WeylIntegral(0.5)
    
    def test_compute_power_function(self, weyl_integral):
        """Test computing Weyl integral of power function."""
        def f(x):
            return x ** 2
        
        x = 1.0
        result = weyl_integral(f, x)
        
        assert isinstance(result, float)
    
    def test_compute_negative_x(self, weyl_integral):
        """Test Weyl integral with negative x."""
        def f(x):
            return x
        
        x = -1.0
        result = weyl_integral(f, x)
        
        # Weyl integral can handle negative values
        assert isinstance(result, float)


class TestHadamardIntegralExpanded:
    """Expanded tests for HadamardIntegral."""
    
    @pytest.fixture
    def hadamard_integral(self):
        """Create Hadamard integral instance."""
        return HadamardIntegral(0.5)
    
    def test_compute_power_function(self, hadamard_integral):
        """Test computing Hadamard integral of power function."""
        def f(x):
            return 1.0 / x if x > 0 else 0
        
        # Hadamard integral requires x > 1
        x = 2.0
        result = hadamard_integral(f, x)
        
        assert isinstance(result, float)
        assert result >= 0
    
    def test_compute_with_log(self, hadamard_integral):
        """Test Hadamard integral with logarithmic function."""
        def f(x):
            return np.log(x) if x > 0 else 0
        
        x = 2.0
        result = hadamard_integral(f, x)
        
        assert isinstance(result, float)


class TestMillerRossIntegralExpanded:
    """Expanded tests for MillerRossIntegral."""
    
    @pytest.fixture
    def miller_ross_integral(self):
        """Create Miller-Ross integral instance."""
        return MillerRossIntegral(0.5)
    
    def test_compute_power_function(self, miller_ross_integral):
        """Test computing Miller-Ross integral of power function."""
        def f(x):
            return x ** 2
        
        x = 1.0
        result = miller_ross_integral(f, x)
        
        assert isinstance(result, float)
        assert result >= 0
    
    def test_compute_zero_x(self, miller_ross_integral):
        """Test Miller-Ross integral with zero x."""
        def f(x):
            return x
        
        x = 0.0
        result = miller_ross_integral(f, x)
        
        assert result == 0.0


class TestMarchaudIntegralExpanded:
    """Expanded tests for MarchaudIntegral."""
    
    @pytest.fixture
    def marchaud_integral(self):
        """Create Marchaud integral instance."""
        return MarchaudIntegral(0.5)
    
    def test_compute_power_function(self, marchaud_integral):
        """Test computing Marchaud integral of power function."""
        def f(x):
            return x ** 2
        
        x = 1.0
        result = marchaud_integral(f, x)
        
        assert isinstance(result, float)
        assert result >= 0
    
    def test_compute_zero_x(self, marchaud_integral):
        """Test Marchaud integral with zero x."""
        def f(x):
            return x
        
        x = 0.0
        result = marchaud_integral(f, x)
        
        assert result == 0.0


class TestIntegralFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_fractional_integral_rl(self):
        """Test creating Riemann-Liouville integral via factory."""
        integral = create_fractional_integral(0.5, "RL")
        assert isinstance(integral, RiemannLiouvilleIntegral)
    
    def test_create_fractional_integral_caputo(self):
        """Test creating Caputo integral via factory."""
        integral = create_fractional_integral(0.5, "Caputo")
        assert isinstance(integral, CaputoIntegral)
    
    def test_create_fractional_integral_weyl(self):
        """Test creating Weyl integral via factory."""
        integral = create_fractional_integral(0.5, "Weyl")
        assert isinstance(integral, WeylIntegral)
    
    def test_create_fractional_integral_hadamard(self):
        """Test creating Hadamard integral via factory."""
        integral = create_fractional_integral(0.5, "Hadamard")
        assert isinstance(integral, HadamardIntegral)
    
    def test_create_fractional_integral_invalid_method(self):
        """Test creating integral with invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            create_fractional_integral(0.5, "InvalidMethod")
    
    def test_create_fractional_integral_factory(self):
        """Test creating integral via factory function."""
        integral = create_fractional_integral_factory("RL", 0.5)
        assert isinstance(integral, FractionalIntegral)


class TestFractionalIntegralFactory:
    """Tests for FractionalIntegralFactory class."""
    
    @pytest.fixture
    def factory(self):
        """Create factory instance."""
        return FractionalIntegralFactory()
    
    def test_register_implementation(self, factory):
        """Test registering custom implementation."""
        # Register an implementation first
        factory.register_implementation("RL", RiemannLiouvilleIntegral)
        # Now create using the registered implementation
        integral = factory.create("RL", 0.5)
        assert isinstance(integral, FractionalIntegral)
    
    def test_get_available_methods(self, factory):
        """Test getting available methods."""
        factory.register_implementation("TEST", RiemannLiouvilleIntegral)
        methods = factory.get_available_methods()
        
        assert isinstance(methods, list)
        assert "TEST" in methods
    
    def test_create_unregistered_method(self, factory):
        """Test creating integral with unregistered method."""
        with pytest.raises(ValueError, match="No implementation registered"):
            factory.create("UNREGISTERED", 0.5)


class TestAnalyticalFunctions:
    """Tests for analytical fractional integral functions."""
    
    def test_analytical_fractional_integral_power(self):
        """Test analytical integral for power function."""
        result = analytical_fractional_integral("power", 0.5, 1.0)
        
        # Should return a function
        assert callable(result)
        
        # Test the function
        integral_value = result(2.0)  # n=2
        assert isinstance(integral_value, (int, float, np.ndarray))
    
    def test_analytical_fractional_integral_exponential(self):
        """Test analytical integral for exponential function."""
        x = np.array([0.5, 1.0, 1.5])
        result = analytical_fractional_integral("exponential", 0.5, x)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape
    
    def test_analytical_fractional_integral_trigonometric(self):
        """Test analytical integral for trigonometric function."""
        x = np.array([0.5, 1.0, 1.5])
        result = analytical_fractional_integral("trigonometric", 0.5, x)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape
    
    def test_analytical_fractional_integral_invalid_type(self):
        """Test analytical integral with invalid function type."""
        with pytest.raises(ValueError, match="Unknown function type"):
            analytical_fractional_integral("invalid", 0.5, 1.0)


class TestNumericalIntegrationMethods:
    """Tests for numerical integration methods."""
    
    def test_trapezoidal_fractional_integral(self):
        """Test trapezoidal fractional integral."""
        def f(x):
            return x ** 2
        
        x = np.array([0.5, 1.0, 1.5])
        result = trapezoidal_fractional_integral(f, x, 0.5, method="RL")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape
        assert np.all(result >= 0)
    
    def test_trapezoidal_fractional_integral_hadamard(self):
        """Test trapezoidal integral with Hadamard method."""
        def f(x):
            # Handle array input properly
            if isinstance(x, np.ndarray):
                return np.where(x > 0, 1.0 / x, 0.0)
            return 1.0 / x if x > 0 else 0
        
        # Hadamard requires x > 1, so use values > 1
        x = np.array([2.0, 3.0, 4.0])
        result = trapezoidal_fractional_integral(f, x, 0.5, method="Hadamard")
        
        assert isinstance(result, np.ndarray)
    
    def test_trapezoidal_fractional_integral_invalid_method(self):
        """Test trapezoidal integral with invalid method."""
        def f(x):
            return x
        
        x = np.array([1.0])
        with pytest.raises(ValueError, match="not supported"):
            trapezoidal_fractional_integral(f, x, 0.5, method="Invalid")
    
    def test_simpson_fractional_integral(self):
        """Test Simpson fractional integral."""
        def f(x):
            return x ** 2
        
        x = np.array([0.5, 1.0, 1.5])
        result = simpson_fractional_integral(f, x, 0.5, method="RL")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_fractional_integral_properties(self):
        """Test getting fractional integral properties."""
        properties = fractional_integral_properties(0.5)
        
        assert isinstance(properties, dict)
        assert 'linearity' in properties
        assert 'composition' in properties
        assert 'semigroup' in properties
    
    def test_validate_fractional_integral(self):
        """Test validating fractional integral."""
        def f(x):
            return x
        
        x = np.array([0.5, 1.0, 1.5, 2.0])
        integral_result = np.array([0.1, 0.3, 0.6, 1.0])  # Increasing
        
        validation = validate_fractional_integral(f, integral_result, x, 0.5)
        
        assert isinstance(validation, dict)
        assert 'linearity_check' in validation
        assert 'monotonicity_check' in validation
        assert 'continuity_check' in validation


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_integral_with_zero_order(self):
        """Test integral with zero order."""
        integral = RiemannLiouvilleIntegral(0.0)
        
        def f(x):
            return x
        
        x = 1.0
        result = integral(f, x)
        
        # I^0 f = f (identity operator)
        assert abs(result - f(x)) < 0.1
    
    def test_integral_with_negative_x(self):
        """Test integral with negative x values."""
        integral = RiemannLiouvilleIntegral(0.5)
        
        def f(x):
            return x ** 2
        
        x = -1.0
        result = integral(f, x)
        
        # Should handle negative values appropriately
        assert isinstance(result, float)
    
    def test_integral_with_array_containing_zeros(self):
        """Test integral with array containing zeros."""
        integral = RiemannLiouvilleIntegral(0.5)
        
        def f(x):
            return x
        
        x = np.array([0.0, 0.5, 1.0])
        result = integral(f, x)
        
        assert isinstance(result, np.ndarray)
        assert result[0] == 0.0  # Integral at 0 should be 0
