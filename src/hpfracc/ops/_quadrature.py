"""Roundoff-stable power-kernel moments for nonuniform product integration."""

from __future__ import annotations

from typing import Any


def _time_difference(left: Any, right: Any) -> Any:
    """Nonnegative integer time differences without signed overflow.

    Integral differences are formed in the corresponding unsigned dtype before
    conversion to floating point. This retains narrow intervals between large
    integer nodes and spans wider than the signed range. Negative integer lags
    belong to masked future history and return zero. Floating paths are unchanged.
    """
    import jax.numpy as jnp

    dtype = jnp.result_type(left, right)
    if jnp.issubdtype(dtype, jnp.integer):
        unsigned = jnp.dtype(f"uint{jnp.iinfo(dtype).bits}")
        difference = left.astype(unsigned) - right.astype(unsigned)
        return jnp.where(left >= right, difference, 0)
    return left - right


def _power_interval_moments(
    upper: Any, lower: Any, width: Any, exponent: Any, valid: Any
) -> tuple[Any, Any, Any]:
    """Integrate a power kernel against constant and linear interval bases.

    For ``A = upper``, ``B = lower``, ``h = width = A - B`` and ``0 < b < 1``,
    return the integrals of ``x**(b-1)`` times ``1``, ``(x-B)/h``, and
    ``(A-x)/h`` over ``[B, A]``. The latter two are the left/right nodal weights
    in forward time. Supply ``h`` from adjacent time nodes, not by subtracting
    the two lags: distant lags may round to the same value on a graded mesh.

    For ``r = h/A <= 1/4``, evaluate the convergent binomial series with
    ``c_k = (1-b)_k / k!``. The dimensionless left/right moments are
    ``sum c_k*r**k/((k+1)*(k+2))`` and ``sum c_k*r**k/(k+2)``. Since
    ``0 < c_k <= 1``, omitting terms from k=26 has relative error at most
    ``2*r**26 / (28*(1-r))`` in either moment, below float64 epsilon.
    This is a roundoff-stable evaluation of the same product integrals, not
    a different quadrature or a history approximation.

    Other intervals use the closed forms with expm1/log power differences.
    Endpoint and masked branches use finite operands throughout, including
    inactive branches, so order gradients are finite at zero lower lag.
    """
    import jax.numpy as jnp

    dtype = jnp.result_type(upper, lower, width, exponent, float)
    # Mask zero-length trial intervals too: completed adaptive scans evaluate
    # their body before discarding the update. Invalid arithmetic there would
    # otherwise poison reverse-mode gradients even through a masked output.
    valid = valid & (jnp.asarray(width) > 0) & (jnp.asarray(upper) > 0)
    a = jnp.where(valid, jnp.asarray(upper, dtype=dtype), 1.0)
    b = jnp.where(valid, jnp.asarray(lower, dtype=dtype), 0.0)
    h = jnp.where(valid, jnp.asarray(width, dtype=dtype), 1.0)
    beta = jnp.asarray(exponent, dtype=dtype)
    r = h / a
    small = r <= 0.25

    # Positive terms avoid the cancellation in the difference of two moments.
    series_r = jnp.where(small, r, 0.0)
    term = jnp.ones_like(r)
    left_series = 0.5 * term
    right_series = 0.5 * term
    for k in range(1, 26):
        term = term * ((k - beta) / k) * series_r
        left_series = left_series + term / ((k + 1) * (k + 2))
        right_series = right_series + term / (k + 2)
    series_scale = h * a ** (beta - 1.0)
    left_series = left_series * series_scale
    right_series = right_series * series_scale

    # Retain B/A directly near the endpoint; 1-r can round to zero even when
    # a small positive B was accurately obtained from adjacent late time nodes.
    ratio = jnp.where(small, 0.0, b / a)
    log_ratio = jnp.log(jnp.where(ratio > 0.0, ratio, 1.0))
    delta = jnp.where(ratio > 0.0, -jnp.expm1(beta * log_ratio), 1.0)
    delta_next = jnp.where(
        ratio > 0.0, -jnp.expm1((beta + 1.0) * log_ratio), 1.0
    )
    power = a**beta
    zeroth = delta / beta
    first = delta_next / (beta + 1.0)
    regular_scale = power / jnp.where(small, 1.0, r)
    left_regular = regular_scale * (first - ratio * zeroth)
    right_regular = regular_scale * (zeroth - first)

    left = jnp.where(small, left_series, left_regular)
    right = jnp.where(small, right_series, right_regular)
    constant = jnp.where(small, left_series + right_series, power * zeroth)
    return (
        jnp.where(valid, constant, 0.0),
        jnp.where(valid, left, 0.0),
        jnp.where(valid, right, 0.0),
    )
