"""
JAX backend implementations for fractional calculus operations.

Fixed issues:
- Grünwald-Letnikov: Computes fractional binomial coefficients inline using
  jax.scipy.special.gamma to avoid module-level jax=None in BinomialCoefficients.
- Caputo & Riemann-Liouville: Uses jax.lax.cond instead of Python-level if/else
  for integer alpha checks, making them safe under JAX transformations (vmap, grad).
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


def _binomial_fractional_jax_inline(alpha, k):
    """
    Compute generalized fractional binomial coefficient C(alpha, k) using JAX.

    Uses the gamma function formula: C(α, k) = Γ(α+1) / (Γ(k+1) * Γ(α-k+1))

    This is defined inline to guarantee it uses the locally-imported `jax` module
    rather than the potentially-None module-level jax in binomial_coeffs.py.
    """
    return jax.scipy.special.gamma(alpha + 1) / (
        jax.scipy.special.gamma(k + 1) * jax.scipy.special.gamma(alpha - k + 1)
    )


def _grunwald_letnikov_jax(f, alpha, h):
    """
    Grünwald-Letnikov fractional derivative via JAX.

    Uses inline gamma-based binomial coefficients to avoid the NoneType error
    that occurred when BinomialCoefficients.jax was None.
    """
    _check_jax()
    n = f.shape[0]

    alpha_arr = jnp.array(alpha, dtype=jnp.float64)
    k_vals = jnp.arange(n, dtype=jnp.float64)

    # Compute C(alpha, k) inline using gamma function
    coeffs = _binomial_fractional_jax_inline(alpha_arr, k_vals)

    # Alternating signs: (-1)^k
    signs = (-1.0) ** k_vals
    gl_coeffs = signs * coeffs

    # Convolution
    result = jax_convolve(f, gl_coeffs, mode='full')[:n]
    return (h ** (-alpha)) * result


def _fractional_integral(f, beta, h, N):
    """
    Compute the Riemann-Liouville fractional integral of order beta.

    Shared helper used by both RL and Caputo implementations.
    """
    k_vals = jnp.arange(N)
    b = (k_vals + 1)**beta - k_vals**beta
    return jax_convolve(f, b, mode='full')[:N] * h**beta / jax.scipy.special.gamma(beta + 1)


def _riemann_liouville_jax(f, alpha, n, h):
    """
    Riemann-Liouville fractional derivative via JAX.

    Uses jax.lax.cond for integer-alpha checks so it is safe under vmap/grad.
    """
    _check_jax()

    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    n = jnp.asarray(n, dtype=jnp.int32)
    N = f.shape[0]
    beta = n - alpha

    # Compute the fractional integral part (always computed; cheap if unused)
    integral_part = _fractional_integral(f, beta, h, N)

    # Apply n integer-order gradient steps
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
    """
    Caputo fractional derivative via JAX.

    Uses jax.lax.cond instead of Python if/else for the integer-alpha shortcuts,
    making this function safe under JAX transformations (vmap, grad, jit).
    """
    _check_jax()

    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    n_ceil = jnp.ceil(alpha).astype(jnp.int32)
    beta = n_ceil - alpha
    N = f.shape[0]

    # --- General fractional case (always computed) ---
    # Compute n-th integer derivative of f
    def deriv_body(i, val):
        return _jnp_gradient_edge_order_2(val, h)
    f_deriv = jax.lax.fori_loop(0, n_ceil, deriv_body, f)

    # Fractional integral of the n-th derivative
    fractional_result = _fractional_integral(f_deriv, beta, h, N)

    # --- Use lax.cond to select the correct branch ---
    # If alpha == 0 → return f
    # If alpha == 1 → return first-order gradient
    # Otherwise    → return the fractional result

    def _return_identity(_):
        return f

    def _check_alpha_one(_):
        return jax.lax.cond(
            alpha == 1.0,
            lambda _: _jnp_gradient_edge_order_2(f, h),
            lambda _: fractional_result,
            None
        )

    result = jax.lax.cond(
        alpha == 0.0,
        _return_identity,
        _check_alpha_one,
        None
    )
    return result
