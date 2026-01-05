"""
Gamma and Beta functions for fractional calculus.

This module provides optimized implementations of the Gamma and Beta functions,
which are fundamental special functions used throughout fractional calculus.
"""

import numpy as np
from typing import Union
import scipy.special as scipy_special
from scipy.special import gamma as gamma_scipy, beta as beta_scipy
try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gamma as gamma_jax, beta as beta_jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Simple module-level convenience wrappers expected by tests


def gamma_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return scipy_special.gamma(x)


def log_gamma(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Logarithm of the Gamma function."""
    return scipy_special.gammaln(x)


def beta_function(a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Optimized Beta function with caching and special case handling."""
    # Handle edge cases
    if np.isscalar(a) and np.isscalar(b):
        if a <= 0 or b <= 0:
            return np.nan
        # Use SciPy directly for better performance
        return scipy_special.beta(a, b)
    else:
        # For arrays, handle element-wise
        a = np.asarray(a)
        b = np.asarray(b)
        
        # Ensure both arrays have compatible shapes
        if a.shape != b.shape:
            # Broadcast to common shape
            a, b = np.broadcast_arrays(a, b)
        
        result = np.full_like(a, np.nan, dtype=float)
        valid_mask = (a > 0) & (b > 0)
        if np.any(valid_mask):
            result[valid_mask] = scipy_special.beta(
                a[valid_mask], b[valid_mask])
        return result


def log_gamma_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return scipy_special.gammaln(x)


def digamma_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return scipy_special.digamma(x)


# Optional numba import
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Convenience functions for optimized beta function


def beta_function_fast(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    use_numba: bool = True,
    cache_size: int = 1000
) -> Union[float, np.ndarray]:
    """
    Fast Beta function optimized for fractional calculus.

    Args:
        x: First parameter
        y: Second parameter
        use_numba: Use Numba JIT compilation
        cache_size: Size of cache for repeated evaluations

    Returns:
        Beta function value(s)
    """
    beta_func = BetaFunction(use_numba=use_numba, cache_size=cache_size)
    return beta_func.compute_fast(x, y)


# Module-level gamma function for Numba compatibility
@jit(nopython=True)
def _gamma_numba_scalar(z: float) -> float:
    """
    NUMBA-optimized Gamma function for scalar inputs.

    Uses Lanczos approximation for accuracy and performance.
    """
    # Lanczos approximation coefficients
    g = 7.0
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]

    if z < 0.5:
        return np.pi / (np.sin(np.pi * z) * _gamma_numba_scalar(1 - z))

    z -= 1
    x = p[0]
    for i in range(1, len(p)):
        x += p[i] / (z + i)

    t = z + g + 0.5
    return np.sqrt(2 * np.pi) * (t ** (z + 0.5)) * np.exp(-t) * x


class GammaFunction:
    """
    Gamma function implementation with multiple optimization strategies.

    The Gamma function is defined as:
    Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt

    For positive integers n: Γ(n) = (n-1)!
    """

    def __init__(self, use_jax: bool = False, use_numba: bool = True, cache_size: int = 1000):
        """
        Initialize Gamma function calculator.

        Args:
            use_jax: Whether to use JAX implementation for vectorized operations
            use_numba: Whether to use NUMBA JIT compilation for scalar operations
            cache_size: Size of the cache for frequently used values
        """
        self.use_jax = use_jax and JAX_AVAILABLE
        self.use_numba = use_numba
        self.cache_size = cache_size
        self._cache = {}

        if use_jax and JAX_AVAILABLE and jax is not None:
            self._gamma_jax = jax.jit(self._gamma_jax_impl)

    def compute(
        self, z: Union[float, np.ndarray, "jnp.ndarray"]
    ) -> Union[float, np.ndarray, "jnp.ndarray"]:
        """
        Compute the Gamma function.

        Args:
            z: Input value(s), can be scalar or array

        Returns:
            Gamma function value(s)
        """
        if self.use_jax and JAX_AVAILABLE and isinstance(z, (jnp.ndarray, float, int)):
            return self._gamma_jax(z)
        elif self.use_numba and isinstance(z, (float, int)):
            return _gamma_numba_scalar(float(z))
        else:
            return self._gamma_scipy(z)

    @staticmethod
    def _gamma_scipy(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """SciPy implementation for reference and fallback."""
        return scipy_special.gamma(z)

    @staticmethod
    def _gamma_jax_impl(z: "jnp.ndarray") -> "jnp.ndarray":
        """
        JAX implementation of Gamma function.

        Uses JAX's built-in gamma function for vectorized operations.
        """
        return jax.scipy.special.gamma(z)

    def log_gamma(
        self, z: Union[float, np.ndarray, "jnp.ndarray"]
    ) -> Union[float, np.ndarray, "jnp.ndarray"]:
        """
        Compute the natural logarithm of the Gamma function.

        Args:
            z: Input value(s)

        Returns:
            Log Gamma function value(s)
        """
        if self.use_jax and JAX_AVAILABLE and isinstance(z, (jnp.ndarray, float)):
            return jax.scipy.special.gammaln(z)
        else:
            return scipy_special.gammaln(z)


class BetaFunction:
    """
    Beta function implementation with multiple optimization strategies.

    The Beta function is defined as:
    B(x, y) = ∫₀¹ t^(x-1) (1-t)^(y-1) dt = Γ(x)Γ(y)/Γ(x+y)
    """

    def __init__(self, use_jax: bool = False, use_numba: bool = True, cache_size: int = 1000):
        """
        Initialize optimized Beta function calculator.

        Args:
            use_jax: Whether to use JAX implementation for vectorized operations
            use_numba: Whether to use NUMBA JIT compilation for scalar operations
            cache_size: Size of the cache for frequently used values
        """
        self.use_jax = use_jax and JAX_AVAILABLE
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.cache_size = cache_size
        self._cache = {}
        self.gamma = GammaFunction(
            use_jax=use_jax, use_numba=use_numba, cache_size=cache_size)

        # Precompute common values for fractional calculus
        self._common_values = {
            (0.5, 0.5): np.pi,  # B(0.5, 0.5) = π
            (1.0, 1.0): 1.0,     # B(1, 1) = 1
            (2.0, 1.0): 0.5,     # B(2, 1) = 1/2
            (1.0, 2.0): 0.5,     # B(1, 2) = 1/2
            (0.5, 1.0): 2.0,     # B(0.5, 1) = 2
            (1.0, 0.5): 2.0,     # B(1, 0.5) = 2
            (3.0, 1.0): 1.0/3.0,  # B(3, 1) = 1/3
            (1.0, 3.0): 1.0/3.0,  # B(1, 3) = 1/3
        }

        if use_jax and JAX_AVAILABLE and jax is not None:
            self._beta_jax = jax.jit(self._beta_jax_impl)

    def compute(
        self,
        x: Union[float, np.ndarray, "jnp.ndarray"],
        y: Union[float, np.ndarray, "jnp.ndarray"],
    ) -> Union[float, np.ndarray, "jnp.ndarray"]:
        """
        Compute the optimized Beta function with caching and special case handling.

        Args:
            x: First parameter
            y: Second parameter

        Returns:
            Beta function value(s)
        """
        # Handle special cases first (common in fractional calculus)
        if isinstance(x, (float, int)) and isinstance(y, (float, int)):
            # Check for exact matches in common values
            if (x, y) in self._common_values:
                return self._common_values[(x, y)]

            # Check cache for scalar inputs
            cache_key = (x, y)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Choose computation method
        if (
            self.use_jax
            and JAX_AVAILABLE
            and isinstance(x, (jnp.ndarray, float, int))
            and isinstance(y, (jnp.ndarray, float, int))
        ):
            result = self._beta_jax(x, y)
        elif (
            self.use_numba
            and isinstance(x, (float, int))
            and isinstance(y, (float, int))
            and x <= 10 and y <= 10 and (x + y) <= 20
        ):
            # Only use Numba for small values where it might be beneficial
            try:
                result = self._beta_numba_scalar(x, y)
            except Exception:
                result = self._beta_scipy(x, y)
        else:
            # Use SciPy by default (much faster)
            result = self._beta_scipy(x, y)

        # Cache scalar results
        if isinstance(x, (float, int)) and isinstance(y, (float, int)):
            if len(self._cache) < self.cache_size:
                self._cache[cache_key] = result

        return result

    def compute_fast(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Fast Beta function computation optimized for fractional calculus.

        This method is specifically optimized for common use cases in
        fractional calculus, particularly fractional integrals and derivatives.
        """
        # Handle special cases first
        if isinstance(x, (float, int)) and isinstance(y, (float, int)):
            if (x, y) in self._common_values:
                return self._common_values[(x, y)]

            # Check cache
            cache_key = (x, y)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Use optimized computation
        if isinstance(x, (float, int)) and isinstance(y, (float, int)):
            # Use SciPy by default (much faster than Numba gamma function)
            # Only use Numba for very specific cases where it might be beneficial
            if self.use_numba and x <= 10 and y <= 10 and (x + y) <= 20:
                try:
                    result = self._beta_numba_scalar(x, y)
                except Exception:
                    # Fallback to SciPy if Numba fails
                    result = self._beta_scipy(x, y)
            else:
                result = self._beta_scipy(x, y)

            # Cache result
            if len(self._cache) < self.cache_size:
                self._cache[cache_key] = result

            return result
        else:
            # Use SciPy for array inputs
            return self._beta_scipy(x, y)

    @staticmethod
    def _beta_scipy(
        x: Union[float, np.ndarray], y: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """SciPy implementation for reference and fallback."""
        return scipy_special.beta(x, y)

    @staticmethod
    @jit(nopython=True)
    def _beta_numba_scalar(x: float, y: float) -> float:
        """
        NUMBA-optimized Beta function for scalar inputs.

        Uses the relationship B(x,y) = Γ(x)Γ(y)/Γ(x+y)
        """
        gamma_x = _gamma_numba_scalar(x)
        gamma_y = _gamma_numba_scalar(y)
        gamma_sum = _gamma_numba_scalar(x + y)
        return gamma_x * gamma_y / gamma_sum

    @staticmethod
    def _beta_jax_impl(x: "jnp.ndarray", y: "jnp.ndarray") -> "jnp.ndarray":
        """
        JAX implementation of Beta function.

        Uses JAX's built-in beta function for vectorized operations.
        """
        return jax.scipy.special.beta(x, y)

    def log_beta(
        self,
        x: Union[float, np.ndarray, "jnp.ndarray"],
        y: Union[float, np.ndarray, "jnp.ndarray"],
    ) -> Union[float, np.ndarray, "jnp.ndarray"]:
        """
        Compute the natural logarithm of the Beta function.

        Args:
            x: First parameter
            y: Second parameter

        Returns:
            Log Beta function value(s)
        """
        if (
            self.use_jax
            and JAX_AVAILABLE
            and isinstance(x, (jnp.ndarray, float))
            and isinstance(y, (jnp.ndarray, float))
        ):
            return jax.scipy.special.betaln(x, y)
        else:
            return scipy_special.betaln(x, y)


# Note: NUMBA vectorization removed for compatibility
# Use the class methods for optimized computations instead


# Convenience functions
def gamma(x):
    """
    Gamma function that is compatible with both JAX and NumPy/SciPy.
    """
    if JAX_AVAILABLE and isinstance(x, (jnp.ndarray, jax.Array)):
        return gamma_jax(x)
    return gamma_scipy(x)


def beta(x, y):
    """
    Beta function that is compatible with both JAX and NumPy/SciPy.
    """
    if np.any(np.asarray(x) <= 0) or np.any(np.asarray(y) <= 0):
        return np.nan

    if JAX_AVAILABLE and (isinstance(x, (jnp.ndarray, jax.Array)) or isinstance(y, (jnp.ndarray, jax.Array))):
        return beta_jax(x, y)
    return beta_scipy(x, y)


def log_gamma(
    z: Union[float, np.ndarray, "jnp.ndarray"], use_jax: bool = False
) -> Union[float, np.ndarray, "jnp.ndarray"]:
    """
    Convenience function to compute log Gamma function.

    Args:
        z: Input value(s)
        use_jax: Whether to use JAX implementation

    Returns:
        Log Gamma function value(s)
    """
    gamma_func = GammaFunction(use_jax=use_jax, use_numba=False)
    return gamma_func.log_gamma(z)


def log_beta(
    x: Union[float, np.ndarray, "jnp.ndarray"],
    y: Union[float, np.ndarray, "jnp.ndarray"],
    use_jax: bool = False,
) -> Union[float, np.ndarray, "jnp.ndarray"]:
    """
    Convenience function to compute log Beta function.

    Args:
        x: First parameter
        y: Second parameter
        use_jax: Whether to use JAX implementation

    Returns:
        Log Beta function value(s)
    """
    beta_func = BetaFunction(use_jax=use_jax, use_numba=False)
    return beta_func.log_beta(x, y)
