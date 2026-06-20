"""Uniform-grid fractional derivative operators.

The v0.1 implementations intentionally use explicit full-history
discretisations. They are baseline numerical references, not yet optimized
large-scale kernels.
"""

from __future__ import annotations

from typing import Any

from hpfracc.ops.base import OperatorFamily, OperatorInfo, OperatorResult
from hpfracc.ops.orders import as_order, validate_order


def grunwald_letnikov(
    x: Any,
    *,
    dt: float,
    order: float,
    return_info: bool = False,
) -> Any:
    """Approximate a Grunwald-Letnikov fractional derivative.

    Parameters
    ----------
    x:
        Samples with leading time axis ``(time, ...)``.
    dt:
        Uniform timestep. Must be positive.
    order:
        Scalar fractional order satisfying ``0 < order < 1``.

    Returns
    -------
    Any
        Approximation with the same shape as ``x``.

    Notes
    -----
    The implementation uses the full-history GL convolution

    ``dt**(-alpha) * sum_{k=0}^{n} w_k x[n-k]``

    with recurrence ``w_0 = 1`` and
    ``w_k = w_{k-1} * (k - 1 - alpha) / k``.
    """

    alpha = as_order(order)
    _validate_dt(dt)

    jnp = _jnp()
    values = jnp.asarray(x)
    _validate_time_axis(values)

    n_time = values.shape[0]
    weights = _gl_weights(alpha, n_time)
    history = _lower_triangular_history_matrix(weights)
    result = jnp.tensordot(history, values, axes=([1], [0])) / (dt**alpha)
    if return_info:
        return OperatorResult(
            values=result,
            operator_info=_operator_info(
                family=OperatorFamily.GRUNWALD_LETNIKOV,
                method="full_history_convolution",
                alpha=alpha,
                dt=dt,
                n_steps=n_time,
            ),
        )
    return result


def riemann_liouville(
    x: Any,
    *,
    dt: float,
    order: float,
    return_info: bool = False,
) -> Any:
    """Approximate a Riemann-Liouville fractional derivative.

    The Grunwald-Letnikov full-history convolution is, for ``0 < order < 1`` on a
    uniform grid with zero history before ``t = 0``, a first-order ``O(dt)``
    consistent discretisation of the lower-terminal Riemann-Liouville derivative.
    v0.1 therefore computes the RL derivative *through* the GL discretisation;
    the two share an implementation by design, not as an unverified assumption.

    Correctness is validated against the analytic RL reference
    ``hp.ops.riemann_liouville_power_law`` -- including the constant case, where
    the RL derivative is ``t**(-order) / Gamma(1 - order)`` and is nonzero,
    distinguishing it from the Caputo derivative.
    """

    result = grunwald_letnikov(x, dt=dt, order=order)
    if return_info:
        return OperatorResult(
            values=result,
            operator_info=_operator_info(
                family=OperatorFamily.RIEMANN_LIOUVILLE,
                method="grunwald_letnikov_discretisation",
                alpha=validate_order(order),
                dt=dt,
                n_steps=result.shape[0],
                warnings=(
                    "v0.1 Riemann-Liouville is computed through the first-order "
                    "GL full-history uniform-grid discretisation.",
                ),
            ),
        )
    return result


def caputo(
    x: Any,
    *,
    dt: float,
    order: float,
    return_info: bool = False,
) -> Any:
    """Approximate a Caputo fractional derivative with the L1 scheme.

    Parameters
    ----------
    x:
        Samples with leading time axis ``(time, ...)``.
    dt:
        Uniform timestep. Must be positive.
    order:
        Scalar fractional order satisfying ``0 < order < 1``.

    Returns
    -------
    Any
        Approximation with the same shape as ``x``. The first sample is zero
        because the L1 history integral has no elapsed interval at ``t=0``.
    """

    alpha = as_order(order)
    _validate_dt(dt)

    jnp = _jnp()
    values = jnp.asarray(x)
    _validate_time_axis(values)

    n_time = values.shape[0]
    if n_time == 1:
        result = jnp.zeros_like(values)
        if return_info:
            return OperatorResult(
                values=result,
                operator_info=_operator_info(
                    family=OperatorFamily.CAPUTO,
                    method="l1_full_history",
                    alpha=alpha,
                    dt=dt,
                    n_steps=n_time,
                ),
            )
        return result

    increments = values[1:] - values[:-1]
    weights = _l1_weights(alpha, n_time - 1)
    history = _lower_triangular_history_matrix(weights)
    scale = 1.0 / (_gamma()(2.0 - alpha) * (dt**alpha))
    tail = jnp.tensordot(history, increments, axes=([1], [0])) * scale
    result = jnp.concatenate([jnp.zeros_like(values[:1]), tail], axis=0)
    if return_info:
        return OperatorResult(
            values=result,
            operator_info=_operator_info(
                family=OperatorFamily.CAPUTO,
                method="l1_full_history",
                alpha=alpha,
                dt=dt,
                n_steps=n_time,
            ),
        )
    return result


def _jnp() -> Any:
    try:
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        msg = (
            "JAX is required for hpfracc.ops numerical operators. Install the "
            "package with its runtime dependencies before calling this function."
        )
        raise ModuleNotFoundError(msg) from exc
    return jnp


def _gamma() -> Any:
    """Return the JAX gamma function.

    Used instead of ``math.gamma`` so the Caputo L1 normalisation stays
    differentiable with respect to the fractional order ``alpha``.
    """

    try:
        from jax.scipy.special import gamma
    except ModuleNotFoundError as exc:
        msg = (
            "JAX is required for hpfracc.ops numerical operators. Install the "
            "package with its runtime dependencies before calling this function."
        )
        raise ModuleNotFoundError(msg) from exc
    return gamma


def _safe_pow(base: Any, exp: Any) -> Any:
    """``base ** exp`` with a finite gradient where ``base == 0``.

    For a positive exponent the value is ``0`` at ``base == 0``, but plain
    ``base ** exp`` differentiates through ``exp * log(base)`` and yields a NaN
    cotangent there. The double-``where`` keeps the masked branch from poisoning
    the gradient. Assumes ``exp > 0`` (as for the ``1 - alpha`` L1 exponent).
    """

    jnp = _jnp()
    safe_base = jnp.where(base > 0, base, 1.0)
    return jnp.where(base > 0, safe_base**exp, 0.0)


def _validate_dt(dt: float) -> None:
    if not float(dt) > 0.0:
        msg = f"Expected positive uniform timestep dt, got {dt}."
        raise ValueError(msg)


def _validate_time_axis(values: Any) -> None:
    if values.ndim < 1:
        msg = "Expected samples with a leading time axis; got a scalar input."
        raise ValueError(msg)
    if values.shape[0] < 1:
        msg = "Expected at least one time sample."
        raise ValueError(msg)


def _gl_weights(alpha: float, n: int) -> Any:
    jnp = _jnp()
    if n == 1:
        return jnp.ones((1,))
    ks = jnp.arange(1, n, dtype=jnp.result_type(float))
    factors = (ks - 1.0 - alpha) / ks
    return jnp.concatenate([jnp.ones((1,), dtype=factors.dtype), jnp.cumprod(factors)])


def _l1_weights(alpha: float, n: int) -> Any:
    jnp = _jnp()
    ks = jnp.arange(n, dtype=jnp.result_type(float))
    # ks**(1 - alpha) is zero at k=0 but has a NaN d/dalpha there; use _safe_pow
    # so gradients with respect to the fractional order stay finite.
    return _safe_pow(ks + 1.0, 1.0 - alpha) - _safe_pow(ks, 1.0 - alpha)


def _lower_triangular_history_matrix(weights: Any) -> Any:
    jnp = _jnp()
    n = weights.shape[0]
    rows = jnp.arange(n)[:, None]
    cols = jnp.arange(n)[None, :]
    lag = rows - cols
    return jnp.where(lag >= 0, weights[jnp.clip(lag, 0, n - 1)], 0.0)


def _operator_info(
    *,
    family: OperatorFamily,
    method: str,
    alpha: float,
    dt: float,
    n_steps: int,
    warnings: tuple[str, ...] = (),
) -> OperatorInfo:
    return OperatorInfo(
        family=family,
        method=method,
        fractional_order=alpha,
        dt=float(dt),
        n_steps=int(n_steps),
        warnings=warnings,
    )
