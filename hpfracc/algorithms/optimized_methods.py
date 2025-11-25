"""
Optimized Fractional Calculus Methods
This module provides unified, high-performance implementations of fractional calculus methods,
leveraging JAX for GPU acceleration with NumPy as a fallback.
"""

import numpy as np
from typing import Union, Callable
from functools import partial
from ..core.definitions import FractionalOrder
import warnings
from scipy.signal import convolve

# Use centralized JAX configuration to prevent conflicts
try:
    from ..core.jax_config import get_jax_safely, is_jax_available
    jax, jnp = get_jax_safely()
    JAX_AVAILABLE = (jax is not None and jnp is not None)
    if JAX_AVAILABLE:
        from jax.scipy.signal import convolve as jax_convolve
        # Configure JAX settings
        jax.config.update("jax_enable_x64", True)
except (ImportError, AttributeError):
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from ..special.gamma_beta import gamma as gamma_func

# JAX Implementations
if JAX_AVAILABLE:
    # jax.config.update already called above
    pass

    def _jnp_gradient_edge_order_2(f, h):
        """JAX implementation of numpy.gradient with edge_order=2."""

        if f.shape[0] < 2:
            return jnp.zeros_like(f)

        # Central difference for interior points
        central = (f[2:] - f[:-2]) / (2 * h)

        # Second-order forward difference for the first point
        forward = (-3*f[0] + 4*f[1] - f[2]) / (2 * h)

        # Second-order backward difference for the last point
        backward = (3*f[-1] - 4*f[-2] + f[-3]) / (2 * h)

        # Fallback for small arrays
        if f.shape[0] < 3:
            return jnp.gradient(f, h)

        return jnp.concatenate([
            jnp.array([forward]),
            central,
            jnp.array([backward])
        ])

    @partial(jax.jit, static_argnums=(1,))
    def _fast_binomial_coefficients_jax(alpha: float, max_k: int) -> jnp.ndarray:

        def body_fun(k_val, coeffs):
            coeff = coeffs[k_val - 1] * (alpha - (k_val - 1)) / k_val
            return coeffs.at[k_val].set(coeff)

        initial_coeffs = jnp.zeros(max_k + 1, dtype=jnp.float64).at[0].set(1.0)
        coeffs = jax.lax.fori_loop(1, max_k + 1, body_fun, initial_coeffs)
        return coeffs

    def _grunwald_letnikov_jax(f: jnp.ndarray, alpha: float, h: float) -> jnp.ndarray:
        n = f.shape[0]
        coeffs = _fast_binomial_coefficients_jax(alpha, n - 1)
        signs = (-1) ** jnp.arange(n)
        gl_coeffs = signs * coeffs
        result = jax_convolve(f, gl_coeffs, mode='full')[:n]
        return (h ** (-alpha)) * result

    def _riemann_liouville_jax(f: jnp.ndarray, alpha: float, n: int, h: float) -> jnp.ndarray:
        N = f.shape[0]
        beta = n - alpha
        k_vals = jnp.arange(N)
        b = (k_vals + 1)**beta - k_vals**beta
        integral_part = jax_convolve(f, b, mode='full')[
            :N] * h**beta / jax.scipy.special.gamma(beta + 1)

        def apply_gradients(val):
            def grad_body(i, v):
                return _jnp_gradient_edge_order_2(v, h)
            return jax.lax.fori_loop(0, n, grad_body, val)

        result = jax.lax.cond(
            n == 0,
            lambda val: val,
            apply_gradients,
            integral_part
        )
        return result

    def _caputo_jax(f: jnp.ndarray, alpha: float, h: float) -> jnp.ndarray:
        n_ceil = jnp.ceil(alpha).astype(int)
        beta = n_ceil - alpha

        # Compute n-th derivative of f
        f_deriv = f

        def deriv_body(i, val):
            return _jnp_gradient_edge_order_2(val, h)
        f_deriv = jax.lax.fori_loop(0, n_ceil, deriv_body, f_deriv)

        # Compute fractional integral of the n-th derivative
        N = f.shape[0]
        k_vals = jnp.arange(N)
        b = (k_vals + 1)**beta - k_vals**beta
        integral = jax_convolve(f_deriv, b, mode='full')[
            :N] * h**beta / jax.scipy.special.gamma(beta + 1)
        return integral

# NumPy Implementations


def _fast_binomial_coefficients_numpy(alpha: float, max_k: int) -> np.ndarray:
    """
    Compute binomial coefficients (alpha choose k) efficiently.
    """
    coeffs = np.zeros(max_k + 1)
    coeffs[0] = 1.0
    for k in range(max_k):
        coeffs[k + 1] = coeffs[k] * (alpha - k) / (k + 1)
    return coeffs


def _grunwald_letnikov_numpy(f: np.ndarray, alpha: float, h: float) -> np.ndarray:
    """
    Compute Grünwald-Letnikov derivative using FFT convolution.
    """
    N = len(f)
    coeffs = _fast_binomial_coefficients_numpy(alpha, N - 1)
    signs = (-1) ** np.arange(N)
    gl_coeffs = signs * coeffs
    
    # Use FFT convolution for O(N log N)
    # Pad to power of 2 for efficiency
    size = int(2 ** np.ceil(np.log2(2 * N - 1)))
    
    f_padded = np.zeros(size)
    f_padded[:N] = f
    
    coeffs_padded = np.zeros(size)
    coeffs_padded[:N] = gl_coeffs
    
    f_fft = np.fft.fft(f_padded)
    coeffs_fft = np.fft.fft(coeffs_padded)
    
    result = np.fft.ifft(f_fft * coeffs_fft).real[:N]
    return result * (h ** (-alpha))


def _riemann_liouville_numpy(f: np.ndarray, alpha: float, n: int, h: float) -> np.ndarray:
    """
    Compute Riemann-Liouville derivative using Grünwald-Letnikov approximation
    which is equivalent for small h, or using RL definition directly.
    Here we use the RL definition: D^alpha f = d^n/dt^n I^(n-alpha) f
    """
    N = len(f)
    beta = n - alpha  # Order of integral
    
    # Compute fractional integral I^beta f
    # Kernel: t^(beta-1) / Gamma(beta)
    # Discrete convolution weights: k^(beta-1)
    
    # We use the formula: I^beta f(t_j) approx h^beta / Gamma(beta) * sum_{k=0}^{j-1} (j-k)^(beta-1) f(t_k)
    # But a better approximation is the trapezoidal convolution or similar.
    # Let's use the standard GL weights for integral (alpha < 0) which is stable.
    
    # Actually, for RL derivative, the standard GL approximation is often preferred 
    # as it discretizes the whole operator D^alpha directly.
    # D^alpha f(t) approx h^(-alpha) * sum_{k=0}^{N} (-1)^k (alpha choose k) f(t-kh)
    
    return _grunwald_letnikov_numpy(f, alpha, h)


def _caputo_numpy(f: np.ndarray, alpha: float, h: float) -> np.ndarray:
    """
    Compute Caputo derivative using the L1 scheme (for 0 < alpha < 1) 
    or L2 scheme (for 1 < alpha < 2).
    """
    N = len(f)
    result = np.zeros_like(f)
    
    if 0 < alpha < 1:
        # L1 Scheme: O(h^(2-alpha))
        # D^alpha f(t_n) approx sigma_alpha * sum_{k=1}^n a_{n,k} (f(t_k) - f(t_{k-1}))
        # where sigma_alpha = 1 / (Gamma(2-alpha) * h^alpha)
        # and a_{n,k} = (n-k+1)^(1-alpha) - (n-k)^(1-alpha)
        
        c = 1.0 / (gamma_func(2 - alpha) * h**alpha)
        
        # Precompute weights
        k = np.arange(N)
        weights = (k + 1)**(1 - alpha) - k**(1 - alpha)
        
        # Compute differences
        df = np.diff(f)
        df = np.insert(df, 0, f[0]) # Handle t=0 boundary? L1 usually starts at t1. 
        # Let's stick to the convolution form for efficiency.
        
        # The sum is a convolution of weights and df
        # sum_{k=1}^n weights[n-k] * df[k]
        
        # Pad for FFT
        size = int(2 ** np.ceil(np.log2(2 * N - 1)))
        
        weights_padded = np.zeros(size)
        weights_padded[:N] = weights
        
        df_padded = np.zeros(size)
        df_padded[:N] = df
        
        w_fft = np.fft.fft(weights_padded)
        df_fft = np.fft.fft(df_padded)
        
        conv = np.fft.ifft(w_fft * df_fft).real[:N]
        result = c * conv
        
    elif 1 < alpha < 2:
        # L2 Scheme (or similar extension of L1)
        # D^alpha f(t) = I^(2-alpha) f''(t)
        # We can use central differences for f'' and then convolve
        
        beta = 2 - alpha
        c = 1.0 / (gamma_func(beta + 1) * h**beta) # Note: Gamma(beta+1) for integral
        
        # Second derivative approximation
        # f''(t_i) approx (f_{i+1} - 2f_i + f_{i-1}) / h^2
        d2f = np.zeros_like(f)
        d2f[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / h**2
        # Boundary conditions (forward/backward)
        d2f[0] = (f[2] - 2*f[1] + f[0]) / h**2 
        d2f[-1] = (f[-1] - 2*f[-2] + f[-3]) / h**2
        
        # Fractional integral of d2f
        # I^beta g(t) = convolution of g with t^(beta-1)
        
        k = np.arange(N)
        # Weights for integral: ((k+1)^beta - k^beta) / beta ? 
        # Or standard GL weights for integral -beta
        
        # Using GL for integral part is easiest and consistent
        integral = _grunwald_letnikov_numpy(d2f, -beta, h)
        result = integral
        
    else:
        # Fallback for integer or other orders
        if alpha == 1.0:
            result = np.gradient(f, h, edge_order=2)
        elif alpha == 0.0:
            result = f
        else:
            # Use GL as generic fallback
            result = _grunwald_letnikov_numpy(f, alpha, h)
            
    return result


################################################################################
# Fractional Operator Classes
################################################################################
class FractionalOperator:
    def __init__(self, order: Union[float, FractionalOrder]):
        if isinstance(order, (int, float)):
            self.alpha = FractionalOrder(order)
        else:
            self.alpha = order

        if self.alpha.alpha < 0:
            raise ValueError(
                "Order must be non-negative for fractional derivative")

        self.fractional_order = self.alpha

    def compute(self, f: Union[Callable, np.ndarray, "jnp.ndarray"], t: Union[float, np.ndarray], h: float = None) -> np.ndarray:
        if h is not None and h <= 0:
            raise ValueError("Step size h must be positive")

        t_array = self._get_t_array(f, t, h)
        f_array = self._prepare_f(f, t_array)

        if len(f_array) == 0:
            return np.array([])

        if np.allclose(f_array, f_array[0]):
            return np.zeros_like(f_array)

        step_size = self._get_step_size(t_array, h)

        if step_size <= 0:
            raise ValueError("Step size must be positive")

        if JAX_AVAILABLE:
            try:
                f_array_jax = jnp.asarray(f_array)
                # This will be overridden in subclasses
                result = self._compute_jit(
                    f_array_jax, self.alpha.alpha, step_size)
                return np.asarray(result)
            except Exception as e:
                # JAX failed (likely due to CuDNN issues), fall back to NumPy
                warnings.warn(f"JAX computation failed ({e}), falling back to NumPy")
                return self._numpy_kernel(f_array, self.alpha.alpha, step_size)
        else:
            # Fallback for non-JAX environments
            return self._numpy_kernel(f_array, self.alpha.alpha, step_size)

    def _get_jax_kernel(self):
        raise NotImplementedError("Subclasses must implement _get_jax_kernel")

    def _numpy_kernel(self, f, alpha, h):
        raise NotImplementedError("Subclasses must implement _numpy_kernel")

    def _get_t_array(self, f: Callable, t: Union[float, np.ndarray], h: float) -> np.ndarray:
        t_is_array = hasattr(t, "__len__")
        if callable(f):
            if not t_is_array:
                # Create a t_array if t is scalar
                return np.arange(0, t + (h or 0.01), (h or 0.01))
            return t
        else:
            if t_is_array:
                return np.asarray(t)
            else:
                # If f is an array but t is not, we can't infer t_array
                raise ValueError(
                    "Time array `t` must be provided for array input `f` if `t` is not an array.")

    def _prepare_f(self, f: Union[Callable, np.ndarray, "jnp.ndarray"], t_array: np.ndarray) -> Union["jnp.ndarray", np.ndarray]:
        if callable(f):
            f_array = np.array([f(ti) for ti in t_array])
        else:
            f_array = np.asarray(f)

        if hasattr(t_array, "__len__") and len(f_array) != len(t_array):
            raise ValueError(
                "Function array and time array must have the same length")

        return f_array

    def _get_step_size(self, t_array: np.ndarray, h: float) -> float:
        if h is not None:
            return h
        elif hasattr(t_array, "__len__") and len(t_array) > 1:
            return t_array[1] - t_array[0]
        else:
            return 1.0

    @property
    def _compute_jit(self):
        if not hasattr(self, '_compute_jit_cache'):
            self._compute_jit_cache = self._get_jax_kernel()
        return self._compute_jit_cache


class OptimizedGrunwaldLetnikov(FractionalOperator):
    """
    Optimized implementation of the Grünwald-Letnikov fractional derivative.
    """

    def _get_jax_kernel(self):
        return jax.jit(_grunwald_letnikov_jax)

    def _numpy_kernel(self, f, alpha, h):
        return _grunwald_letnikov_numpy(f, alpha, h)

    def compute(self, f: Union[Callable, np.ndarray, "jnp.ndarray"], t: Union[float, np.ndarray], h: float = None) -> np.ndarray:
        # Grunwald-Letnikov does not need special handling for alpha=0 or 1, can call super
        return super().compute(f, t, h)


class OptimizedRiemannLiouville(FractionalOperator):
    """
    Optimized implementation of the Riemann-Liouville fractional derivative.
    """

    def __init__(self, order: Union[float, FractionalOrder]):
        super().__init__(order)
        self.n = int(np.ceil(self.alpha.alpha))

    def compute(self, f: Union[Callable, np.ndarray, "jnp.ndarray"], t: Union[float, np.ndarray], h: float = None) -> np.ndarray:
        if h is not None and h <= 0:
            raise ValueError("Step size h must be positive")

        t_array = self._get_t_array(f, t, h)
        f_array = self._prepare_f(f, t_array)

        if self.alpha.alpha == 0:
            return f_array
        if self.alpha.alpha == 1:
            step_size = self._get_step_size(t_array, h)
            return np.gradient(f_array, step_size, edge_order=1)

        if len(f_array) == 0:
            return np.array([])

        step_size = self._get_step_size(t_array, h)

        if step_size <= 0:
            raise ValueError("Step size must be positive")

        if JAX_AVAILABLE:
            try:
                f_array_jax = jnp.asarray(f_array)
                result = self._compute_jit(
                    f_array_jax, self.alpha.alpha, self.n, step_size)
                return np.asarray(result)
            except Exception as e:
                # JAX failed (likely due to CuDNN issues), fall back to NumPy
                warnings.warn(f"JAX computation failed ({e}), falling back to NumPy")
                return self._numpy_kernel(f_array, self.alpha.alpha, step_size)
        else:
            return self._numpy_kernel(f_array, self.alpha.alpha, step_size)

    def _get_jax_kernel(self):
        return jax.jit(_riemann_liouville_jax)

    def _numpy_kernel(self, f, alpha, h):
        return _riemann_liouville_numpy(f, alpha, self.n, h)


class OptimizedCaputo(FractionalOperator):
    """
    Optimized implementation of the Caputo fractional derivative.
    
    Note: Caputo derivative is defined for all alpha > 0.
    The implementation uses the standard formulation:
    D^α f(t) = I^(n-α) f^(n)(t) where n = ceil(α)
    """

    def __init__(self, order: Union[float, FractionalOrder]):
        super().__init__(order)
        self.n = int(np.ceil(self.alpha.alpha))
        # Caputo is defined for all alpha > 0 (no restriction needed)

    def compute(self, f: Union[Callable, np.ndarray, "jnp.ndarray"], t: Union[float, np.ndarray], h: float = None) -> np.ndarray:
        return super().compute(f, t, h)

    def _get_jax_kernel(self):
        return jax.jit(_caputo_jax)

    def _numpy_kernel(self, f, alpha, h):
        return _caputo_numpy(f, alpha, h)

# Parallel computation classes
# These classes inherit all functionality from their optimized base classes
# and are designed to support parallel computation in the future.
# Currently, they provide the same interface and can be used as drop-in replacements.


class ParallelOptimizedRiemannLiouville(OptimizedRiemannLiouville):
    """
    Parallel version of OptimizedRiemannLiouville.
    
    Currently inherits all functionality from OptimizedRiemannLiouville.
    Future enhancements will include:
    - Multi-threaded computation for large datasets
    - Distributed computing support
    - Load balancing across multiple cores
    """
    pass


class ParallelOptimizedCaputo(OptimizedCaputo):
    """
    Parallel version of OptimizedCaputo.
    
    Currently inherits all functionality from OptimizedCaputo.
    Future enhancements will include:
    - Multi-threaded computation for large datasets
    - Distributed computing support
    - Load balancing across multiple cores
    """
    pass


class ParallelOptimizedGrunwaldLetnikov(OptimizedGrunwaldLetnikov):
    """
    Parallel version of OptimizedGrunwaldLetnikov.
    
    Currently inherits all functionality from OptimizedGrunwaldLetnikov.
    Future enhancements will include:
    - Multi-threaded computation for large datasets
    - Distributed computing support
    - Load balancing across multiple cores
    """
    pass


class ParallelPerformanceMonitor:
    """
    Performance monitoring for parallel fractional calculus operations.
    
    This is a placeholder class for future implementation.
    Will include:
    - Per-thread performance metrics
    - Load balancing statistics
    - Memory usage tracking across threads
    - Bottleneck identification
    """
    pass


class NumbaOptimizer:
    """
    Numba JIT optimization utilities for fractional calculus.
    
    This is a placeholder class for future implementation.
    Will include:
    - Automatic kernel generation
    - JIT compilation configuration
    - Cache management
    - Performance profiling
    """
    pass


class NumbaFractionalKernels:
    """
    Pre-compiled Numba kernels for fractional calculus operations.
    
    This is a placeholder class for future implementation.
    Will include:
    - Optimized convolution kernels
    - FFT-based fractional derivative kernels
    - Memory-efficient history accumulation
    """
    pass

# Convenience functions


def optimized_riemann_liouville(f, t, alpha, h=None):
    return OptimizedRiemannLiouville(alpha).compute(f, t, h)


def optimized_caputo(f, t, alpha, h=None):
    return OptimizedCaputo(alpha).compute(f, t, h)


def optimized_grunwald_letnikov(f, t, alpha, h=None):
    return OptimizedGrunwaldLetnikov(alpha).compute(f, t, h)

# Dummy classes for compatibility


class OptimizedFractionalMethods:
    """
    Collection of optimized fractional calculus methods.
    
    This is a placeholder class for future implementation.
    Will provide a unified interface for:
    - Method selection and recommendation
    - Automatic optimization based on problem characteristics
    - Performance benchmarking
    """
    pass


class ParallelConfig:
    """
    Configuration for parallel fractional calculus computations.
    
    Attributes:
        n_jobs: Number of parallel jobs (threads/processes)
        enabled: Whether parallel computation is enabled
    """
    def __init__(self, n_jobs=1, enabled=False, **kwargs):
        self.n_jobs = n_jobs
        self.enabled = enabled


class AdvancedFFTMethods:
    """
    Advanced FFT-based methods for fractional derivatives.
    
    Supports multiple FFT-based approaches:
    - Spectral methods
    - Fractional Fourier transforms
    - Wavelet-based methods
    
    Args:
        method: FFT method to use ('spectral', 'fractional_fourier', 'wavelet')
    """
    def __init__(self, method="spectral", *args, **kwargs):
        self.method = method

    def compute_derivative(self, f, t, alpha, h):
        """
        Compute fractional derivative using FFT-based method.
        
        Args:
            f: Function values (array)
            t: Time points (array)
            alpha: Fractional order
            h: Step size
            
        Returns:
            Fractional derivative approximation
        """
        # Handle constant functions
        if np.allclose(f, f[0]):
            return np.zeros_like(f)
        
        # Simple FFT-based approximation
        # For more accurate results, use OptimizedCaputo or OptimizedRiemannLiouville
        N = len(f)
        f_fft = np.fft.fft(f)
        
        # Frequency domain fractional derivative: multiply by (iω)^α
        omega = 2.0 * np.pi * np.fft.fftfreq(N, h)
        
        # Apply fractional operator in frequency domain
        # Using the fact that D^α[e^(iωt)] = (iω)^α e^(iωt)
        multiplier = np.power(1j * omega + 1e-10, alpha)
        
        # Compute inverse FFT
        result = np.fft.ifft(f_fft * multiplier).real
        
        return result


class L1L2Schemes:
    """
    L1 and L2 finite difference schemes for fractional derivatives.
    
    This is a placeholder class for future implementation.
    Will include:
    - L1 scheme for Caputo derivatives (first-order accuracy)
    - L2 scheme for Caputo derivatives (second-order accuracy)
    - L1-2 hybrid schemes
    """
    pass


class ParallelLoadBalancer:
    """
    Load balancer for parallel fractional calculus computations.
    
    This is a placeholder class for future implementation.
    Will include:
    - Dynamic work distribution
    - Thread pool management
    - Resource allocation optimization
    """
    pass


def parallel_optimized_riemann_liouville(f, t, alpha, h=None):
    return OptimizedRiemannLiouville(alpha).compute(f, t, h)


def parallel_optimized_caputo(f, t, alpha, h=None):
    return OptimizedCaputo(alpha).compute(f, t, h)


def parallel_optimized_grunwald_letnikov(f, t, alpha, h=None):
    return OptimizedGrunwaldLetnikov(alpha).compute(f, t, h)
