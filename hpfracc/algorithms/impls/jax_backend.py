"""
JAX backend implementations for fractional calculus operations.
"""

from functools import partial
import warnings

try:
    from ...core.jax_config import get_jax_safely
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

from ...special.binomial_coeffs import BinomialCoefficients


def _check_jax():
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available.")


def _jnp_gradient_edge_order_2(f, h):
    """JAX implementation of numpy.gradient with edge_order=2."""
    _check_jax()
    
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


def _grunwald_letnikov_jax(f, alpha, h):
    _check_jax()
    n = f.shape[0]
    # Use centralized binomial coefficients
    coeffs = BinomialCoefficients(use_jax=True)._binomial_fractional_jax(
        jnp.array(alpha, dtype=jnp.float64), 
        jnp.arange(n, dtype=jnp.float64)
    )
    # Signs
    signs = (-1.0) ** jnp.arange(n)
    gl_coeffs = signs * coeffs
    
    # Convolution
    result = jax_convolve(f, gl_coeffs, mode='full')[:n]
    return (h ** (-alpha)) * result


def _riemann_liouville_jax(f, alpha, n, h):
    _check_jax()
    
    # Handle integer cases
    if alpha == 0.0:
        return f
    if alpha == 1.0:
        return _jnp_gradient_edge_order_2(f, h)
        
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


def _caputo_jax(f, alpha, h):
    _check_jax()
    n_ceil = jnp.ceil(alpha).astype(int)
    beta = n_ceil - alpha
    
    # Handle integers
    if alpha == 0.0:
        return f
    if alpha == 1.0:
        return _jnp_gradient_edge_order_2(f, h)

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
