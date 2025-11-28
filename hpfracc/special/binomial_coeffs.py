"""
Binomial coefficients for fractional calculus.

This module provides optimized implementations of binomial coefficients,
which are fundamental in the Grünwald-Letnikov definition of fractional derivatives.
"""

import math
import numpy as np
from typing import Union

# Simplified JAX import
try:
    import jax
    import jax.numpy as jnp
    from jax.config import config
    config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

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
import scipy.special as scipy_special

# Convenience functions for optimized binomial coefficients


def binomial_coefficient_fast(
    n: Union[float, int, np.ndarray],
    k: Union[float, int, np.ndarray],
    use_numba: bool = True,
    cache_size: int = 1000
) -> Union[float, np.ndarray]:
    """
    Fast binomial coefficient computation optimized for fractional calculus.

    Args:
        n: Upper parameter (can be fractional)
        k: Lower parameter (integer)
        use_numba: Use Numba JIT compilation
        cache_size: Size of cache for repeated evaluations

    Returns:
        Binomial coefficient value(s)
    """
    binomial_func = BinomialCoefficients(
        use_numba=use_numba, cache_size=cache_size)
    return binomial_func.compute(n, k)


def binomial_sequence_fast(
    alpha: float,
    max_k: int,
    use_numba: bool = True,
    cache_size: int = 1000,
    sequence_cache_size: int = 100
) -> np.ndarray:
    """
    Fast binomial sequence computation optimized for fractional calculus.

    Args:
        alpha: Fractional parameter
        max_k: Maximum value of k
        use_numba: Use Numba JIT compilation
        cache_size: Size of cache for individual coefficients
        sequence_cache_size: Size of cache for sequences

    Returns:
        Array of binomial coefficients [C(α,0), C(α,1), ..., C(α,max_k)]
    """
    binomial_func = BinomialCoefficients(
        use_numba=use_numba,
        cache_size=cache_size,
        sequence_cache_size=sequence_cache_size
    )
    return binomial_func.compute_sequence(alpha, max_k)


class BinomialCoefficients:
    """
    Binomial coefficients implementation with multiple optimization strategies.

    The binomial coefficient is defined as:
    C(n,k) = n! / (k! * (n-k)!) = Γ(n+1) / (Γ(k+1) * Γ(n-k+1))

    For fractional calculus, we need generalized binomial coefficients:
    C(α,k) = Γ(α+1) / (Γ(k+1) * Γ(α-k+1))
    where α can be any real number.
    """

    def __init__(
            self,
            use_jax: bool = False,
            use_numba: bool = True,
            cache_size: int = 1000,
            sequence_cache_size: int = 100):
        """
        Initialize optimized binomial coefficients calculator.

        Args:
            use_jax: Whether to use JAX implementation for vectorized operations
            use_numba: Whether to use NUMBA JIT compilation for scalar operations
            cache_size: Size of the cache for frequently used coefficients
            sequence_cache_size: Size of the cache for sequences (common in fractional calculus)
        """
        self.use_jax = use_jax and JAX_AVAILABLE
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.cache_size = cache_size
        self.sequence_cache_size = sequence_cache_size
        self._cache = {}
        self._sequence_cache = {}  # Cache for sequences

        # Precompute common fractional values
        self._common_fractional = {
            (0.5, 0): 1.0,    # C(0.5, 0) = 1
            (0.5, 1): 0.5,    # C(0.5, 1) = 0.5
            (0.5, 2): -0.125,  # C(0.5, 2) = -0.125
            (0.5, 3): 0.0625,  # C(0.5, 3) = 0.0625
            (0.25, 0): 1.0,   # C(0.25, 0) = 1
            (0.25, 1): 0.25,  # C(0.25, 1) = 0.25
            (0.75, 0): 1.0,   # C(0.75, 0) = 1
            (0.75, 1): 0.75,  # C(0.75, 1) = 0.75
        }

        if use_jax and JAX_AVAILABLE and jax is not None:
            self._binomial_jax = jax.jit(self._binomial_jax_impl)

    def compute(
        self,
        n: Union[float, int, np.ndarray, "jnp.ndarray"],
        k: Union[float, int, np.ndarray, "jnp.ndarray"],
    ) -> Union[float, np.ndarray, "jnp.ndarray"]:
        """
        Compute the binomial coefficient C(n,k).

        Args:
            n: Upper parameter (can be fractional)
            k: Lower parameter (integer)

        Returns:
            Binomial coefficient value(s)
        """
        # Handle special cases first
        if isinstance(n, (float, int)) and isinstance(k, (float, int)):
            # Check for exact matches in common fractional values
            if (n, k) in self._common_fractional:
                return self._common_fractional[(n, k)]

            # Check cache for scalar inputs
            cache_key = (n, k)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Compute the result
        if self.use_jax:
            try:
                import jax.numpy as jnp
                if (
                    isinstance(n, (jnp.ndarray, float, int))
                    and isinstance(k, (jnp.ndarray, float, int))
                ):
                    result = self._binomial_jax(n, k)
                else:
                    result = self._binomial_jax(n, k)
            except ImportError:
                result = self._binomial_scipy(n, k)
        elif (
            self.use_numba
            and isinstance(n, (float, int))
            and isinstance(k, (float, int))
        ):
            result = self._binomial_numba_scalar(n, k)
        else:
            result = self._binomial_scipy(n, k)

        # Convert to integer if result is a whole number
        if isinstance(result, (int, float)) and result == int(result):
            result = int(result)

        # Cache scalar results
        if isinstance(n, (float, int)) and isinstance(k, (float, int)):
            if len(self._cache) < self.cache_size:
                self._cache[(n, k)] = result

        return result

    @staticmethod
    def _binomial_scipy(
        n: Union[float, np.ndarray], k: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """SciPy implementation for reference and fallback."""
        result = scipy_special.binom(n, k)
        
        # Convert to integer if result is a whole number
        if isinstance(result, (int, float)) and result == int(result):
            return int(result)
        elif isinstance(result, np.ndarray):
            # Handle array case
            integer_mask = (result == result.astype(int))
            result = result.astype(float)
            result[integer_mask] = result[integer_mask].astype(int)
            return result
        
        return result

    @staticmethod
    @jit(nopython=True)
    def _binomial_numba_scalar(n: float, k: float) -> float:
        """
        NUMBA-optimized binomial coefficient for scalar inputs.

        Uses the gamma function relationship for generalized binomial coefficients.
        """
        # Handle special cases
        if k < 0 or k > n:
            return 0.0
        if k == 0 or k == n:
            return 1.0

        # For integer n and k, use the standard formula
        if n == int(n) and k == int(k):
            n_int = int(n)
            k_int = int(k)
            if k_int > n_int // 2:
                k_int = n_int - k_int  # Use symmetry

            result = 1
            for i in range(k_int):
                result = result * (n_int - i) // (i + 1)
            return result

        # For fractional cases, use the gamma function relationship
        # For fractional binomial coefficients: C(n,k) = Γ(n+1) / (Γ(k+1) * Γ(n-k+1))
        # We can compute this using the logarithm to avoid overflow:
        # log(C(n,k)) = log(Γ(n+1)) - log(Γ(k+1)) - log(Γ(n-k+1))
        
        if k == 0:
            return 1.0
        if k == 1:
            return n
        if k == 2:
            return n * (n - 1) / 2.0
        
        # For other fractional cases, use recursive formula for better Numba compatibility
        # C(n,k) = C(n,k-1) * (n-k+1) / k
        # This avoids the need for log-gamma which requires imports
        result = 1.0
        for i in range(1, int(k) + 1):
            result *= (n - i + 1) / i
        
        return result

    @staticmethod
    def _binomial_jax_impl(n: "jnp.ndarray", k: "jnp.ndarray") -> "jnp.ndarray":
        """
        JAX implementation of binomial coefficient using gamma function.

        Uses the gamma function formula: C(n,k) = Γ(n+1) / (Γ(k+1) * Γ(n-k+1))
        """
        return jax.scipy.special.gamma(n + 1) / (jax.scipy.special.gamma(k + 1) * jax.scipy.special.gamma(n - k + 1))

    def compute_fractional(
        self,
        alpha: Union[float, np.ndarray, "jnp.ndarray"],
        k: Union[int, np.ndarray, "jnp.ndarray"],
    ) -> Union[float, np.ndarray, "jnp.ndarray"]:
        """
        Compute the generalized binomial coefficient C(α,k) for fractional α.

        Args:
            alpha: Fractional parameter
            k: Integer parameter

        Returns:
            Generalized binomial coefficient value(s)
        """
        if (
            self.use_jax
            and isinstance(alpha, ("jnp.ndarray", float))
            and isinstance(k, ("jnp.ndarray", int))
        ):
            return self._binomial_fractional_jax(alpha, k)
        elif (
            self.use_numba
            and isinstance(alpha, (float, int))
            and isinstance(k, (int, float))
        ):
            return self._binomial_fractional_numba_scalar(alpha, k)
        else:
            return self._binomial_fractional_scipy(alpha, k)

    @staticmethod
    def _binomial_fractional_scipy(
        alpha: Union[float, np.ndarray], k: Union[int, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """SciPy implementation for fractional binomial coefficients."""
        return scipy_special.binom(alpha, k)

    @staticmethod
    @jit(nopython=True)
    def _binomial_fractional_numba_scalar(alpha: float, k: int) -> float:
        """
        NUMBA-optimized fractional binomial coefficient for scalar inputs.

        Uses the recursive formula for generalized binomial coefficients:
        C(α,k) = C(α,k-1) * (α-k+1) / k
        
        This provides accurate computation for all fractional orders α and integer k.
        
        Args:
            alpha: Fractional parameter (can be any real number)
            k: Integer parameter
            
        Returns:
            Generalized binomial coefficient C(α,k)
        """
        # Handle special cases
        if k < 0:
            return 0.0
        if k == 0:
            return 1.0
        if k == 1:
            return alpha
        if k == 2:
            return alpha * (alpha - 1) / 2.0
        if k == 3:
            return alpha * (alpha - 1) * (alpha - 2) / 6.0
        
        # For other cases, use recursive formula for efficiency
        # C(α,k) = C(α,k-1) * (α-k+1) / k
        result = 1.0
        for i in range(1, int(k) + 1):
            result *= (alpha - i + 1) / i
        
        return result

    @staticmethod
    def _binomial_fractional_jax(
            alpha: "jnp.ndarray",
            k: "jnp.ndarray") -> "jnp.ndarray":
        """
        JAX implementation of fractional binomial coefficient using gamma function.

        Uses the gamma function formula: C(α,k) = Γ(α+1) / (Γ(k+1) * Γ(α-k+1))
        """
        return jax.scipy.special.gamma(alpha + 1) / (jax.scipy.special.gamma(k + 1) * jax.scipy.special.gamma(alpha - k + 1))

    def compute_sequence(self, alpha: float, max_k: int) -> np.ndarray:
        """
        Compute the sequence of binomial coefficients C(α,k) for k = 0, 1, ..., max_k.

        Optimized with caching and recursive computation for efficiency.

        Args:
            alpha: Fractional parameter
            max_k: Maximum value of k

        Returns:
            Array of binomial coefficients [C(α,0), C(α,1), ..., C(α,max_k)]
        """
        # Check sequence cache first
        sequence_key = (alpha, max_k)
        if sequence_key in self._sequence_cache:
            return self._sequence_cache[sequence_key]

        # Use optimized recursive computation
        result = self._compute_sequence_optimized(alpha, max_k)

        # Cache the result
        if len(self._sequence_cache) < self.sequence_cache_size:
            self._sequence_cache[sequence_key] = result

        return result

    def _compute_sequence_optimized(self, alpha: float, max_k: int) -> np.ndarray:
        """
        Optimized sequence computation using recursive formula.

        Uses the recursive relationship: C(α,k+1) = C(α,k) * (α-k)/(k+1)
        This is much more efficient than computing each coefficient individually.
        """
        if self.use_jax and JAX_AVAILABLE:
            k = jnp.arange(max_k + 1)
            return jax.scipy.special.binom(alpha, k)
        else:
            # Use recursive formula for efficiency
            result = np.zeros(max_k + 1)
            result[0] = 1.0  # C(α,0) = 1

            # Use recursive formula: C(α,k+1) = C(α,k) * (α-k)/(k+1)
            for k in range(max_k):
                result[k + 1] = result[k] * (alpha - k) / (k + 1)

            return result

    def compute_alternating_sequence(
            self, alpha: float, max_k: int) -> np.ndarray:
        """
        Compute the alternating sequence of binomial coefficients (-1)^k * C(α,k).

        Args:
            alpha: Fractional parameter
            max_k: Maximum value of k

        Returns:
            Array of alternating binomial coefficients
        """
        coeffs = self.compute_sequence(alpha, max_k)
        signs = (-1) ** np.arange(max_k + 1)
        return coeffs * signs


class GrunwaldLetnikovCoefficients:
    """
    Specialized binomial coefficients for Grünwald-Letnikov fractional derivatives.

    These coefficients appear in the Grünwald-Letnikov definition:
    D^α f(x) = lim_{h→0} h^(-α) * Σ_{k=0}^∞ (-1)^k * C(α,k) * f(x - kh)
    """

    def __init__(self, use_jax: bool = False, use_numba: bool = True, cache_size: int = 1000):
        """
        Initialize Grünwald-Letnikov coefficients calculator.

        Args:
            use_jax: Whether to use JAX implementation
            use_numba: Whether to use NUMBA implementation
            cache_size: Size of the cache for frequently used coefficients
        """
        self.use_jax = use_jax and JAX_AVAILABLE
        self.use_numba = use_numba
        self.cache_size = cache_size
        self._cache = {}
        self.binomial = BinomialCoefficients(
            use_jax=use_jax, use_numba=use_numba, cache_size=cache_size)

    def compute_coefficients(self, alpha: float, max_k: int) -> np.ndarray:
        """
        Compute Grünwald-Letnikov coefficients for fractional order α.

        Args:
            alpha: Fractional order
            max_k: Maximum number of coefficients

        Returns:
            Array of coefficients [w_0, w_1, ..., w_max_k]
        """
        if self.use_jax:
            k = jnp.arange(max_k + 1)
            return (-1) ** k * jax.scipy.special.binom(alpha, k)
        else:
            k = np.arange(max_k + 1)
            return (-1) ** k * scipy_special.binom(alpha, k)

    def compute_weighted_coefficients(
        self, alpha: float, max_k: int, h: float
    ) -> np.ndarray:
        """
        Compute weighted Grünwald-Letnikov coefficients with step size h.

        Args:
            alpha: Fractional order
            max_k: Maximum number of coefficients
            h: Step size

        Returns:
            Array of weighted coefficients
        """
        coeffs = self.compute_coefficients(alpha, max_k)
        return coeffs / (h**alpha)

    def compute(self, alpha: float, max_k: int) -> np.ndarray:
        """
        Compute Grünwald-Letnikov coefficients (alias for compute_coefficients).

        Args:
            alpha: Fractional order
            max_k: Maximum number of coefficients

        Returns:
            Array of coefficients
        """
        return self.compute_coefficients(alpha, max_k)


# Note: NUMBA vectorization removed for compatibility
# Use the class methods for optimized computations instead


# Convenience functions
def binomial(
    n: Union[float, np.ndarray, "jnp.ndarray"],
    k: Union[float, np.ndarray, "jnp.ndarray"],
    use_jax: bool = False,
    use_numba: bool = True,
) -> Union[float, np.ndarray, "jnp.ndarray"]:
    """
    Convenience function to compute binomial coefficient.

    Args:
        n: Upper parameter
        k: Lower parameter
        use_jax: Whether to use JAX implementation
        use_numba: Whether to use NUMBA implementation

    Returns:
        Binomial coefficient value(s)
    """
    binomial_func = BinomialCoefficients(use_jax=use_jax, use_numba=use_numba)
    return binomial_func.compute(n, k)


def binomial_fractional(
    alpha: Union[float, np.ndarray, "jnp.ndarray"],
    k: Union[int, np.ndarray, "jnp.ndarray"],
    use_jax: bool = False,
    use_numba: bool = True,
) -> Union[float, np.ndarray, "jnp.ndarray"]:
    """
    Convenience function to compute fractional binomial coefficient.

    Args:
        alpha: Fractional parameter
        k: Integer parameter
        use_jax: Whether to use JAX implementation
        use_numba: Whether to use NUMBA implementation

    Returns:
        Fractional binomial coefficient value(s)
    """
    binomial_func = BinomialCoefficients(use_jax=use_jax, use_numba=use_numba)
    return binomial_func.compute_fractional(alpha, k)


def grunwald_letnikov_coefficients(
    alpha: float, max_k: int, use_jax: bool = False, use_numba: bool = True
) -> np.ndarray:
    """
    Convenience function to compute Grünwald-Letnikov coefficients.

    Args:
        alpha: Fractional order
        max_k: Maximum number of coefficients
        use_jax: Whether to use JAX implementation
        use_numba: Whether to use NUMBA implementation

    Returns:
        Array of Grünwald-Letnikov coefficients
    """
    gl_coeffs = GrunwaldLetnikovCoefficients(
        use_jax=use_jax, use_numba=use_numba)
    return gl_coeffs.compute_coefficients(alpha, max_k)


def grunwald_letnikov_weighted_coefficients(
        alpha: float,
        max_k: int,
        h: float,
        use_jax: bool = False,
        use_numba: bool = True) -> np.ndarray:
    """
    Convenience function to compute weighted Grünwald-Letnikov coefficients.

    Args:
        alpha: Fractional order
        max_k: Maximum number of coefficients
        h: Step size
        use_jax: Whether to use JAX implementation
        use_numba: Whether to use NUMBA implementation

    Returns:
        Array of weighted Grünwald-Letnikov coefficients
    """
    gl_coeffs = GrunwaldLetnikovCoefficients(
        use_jax=use_jax, use_numba=use_numba)
    return gl_coeffs.compute_weighted_coefficients(alpha, max_k, h)


# Special sequences
def pascal_triangle(
    n: int, use_jax: bool = False, use_numba: bool = True
) -> np.ndarray:
    """
    Generate Pascal's triangle up to row n.

    Args:
        n: Number of rows
        use_jax: Whether to use JAX implementation
        use_numba: Whether to use NUMBA implementation

    Returns:
        Pascal's triangle as a 2D array
    """
    if use_jax:
        triangle = jnp.zeros((n + 1, n + 1))
        for i in range(n + 1):
            for j in range(i + 1):
                triangle = triangle.at[i, j].set(jax.scipy.special.binom(i, j))
        return triangle
    else:
        triangle = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            for j in range(i + 1):
                triangle[i, j] = scipy_special.binom(i, j)
        return triangle


def fractional_pascal_triangle(
    alpha: float, n: int, use_jax: bool = False, use_numba: bool = True
) -> np.ndarray:
    """
    Generate fractional Pascal's triangle for parameter α up to row n.

    Args:
        alpha: Fractional parameter
        n: Number of rows
        use_jax: Whether to use JAX implementation
        use_numba: Whether to use NUMBA implementation

    Returns:
        Fractional Pascal's triangle as a 2D array
    """
    if use_jax:
        triangle = jnp.zeros((n + 1, n + 1))
        for i in range(n + 1):
            for j in range(i + 1):
                triangle = triangle.at[i, j].set(
                    jax.scipy.special.binom(alpha + i, j))
        return triangle
    else:
        triangle = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            for j in range(i + 1):
                triangle[i, j] = scipy_special.binom(alpha + i, j)
        return triangle
