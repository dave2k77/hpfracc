"""Uniform-grid fractional derivative operators.

The v0.1 implementations intentionally use explicit full-history
discretisations. They are baseline numerical references, not yet optimized
large-scale kernels.
"""

from __future__ import annotations

from math import gamma
from typing import Any

from hpfracc.ops.base import HistoryMethod, OperatorFamily, OperatorInfo, OperatorResult
from hpfracc.ops.kernels import (
    _gl_weights,
    _history_convolution,
    _l1_weights,
    _soe_convolution,
    _soe_weights,
)
from hpfracc.ops.orders import validate_order


def grunwald_letnikov(
    x: Any,
    *,
    dt: float,
    order: float,
    history: HistoryMethod | str = HistoryMethod.FULL,
    window_steps: int = 64,
    soe_poles: int = 8,
    soe_t_max: float | None = None,
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
    history:
        History convolution strategy.  ``"full"`` (default) uses the dense
        lower-triangular matrix; ``"fft"`` uses an FFT-accelerated causal
        convolution; ``"short_memory"`` truncates history to ``window_steps``;
        ``"soe"`` uses a sum-of-exponentials kernel approximation.  All
        alternatives are opt-in and validated against the ``"full"`` reference.
    window_steps:
        Number of recent time steps retained when ``history="short_memory"``.
    soe_poles:
        Number of exponential poles when ``history="soe"``.
    soe_t_max:
        Physical horizon used for SOE fitting.  Defaults to the signal duration
        ``n_steps * dt``.

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

    alpha = validate_order(order)
    _validate_dt(dt)

    jnp = _jnp()
    values = jnp.asarray(x)
    _validate_time_axis(values)

    n_time = values.shape[0]
    if HistoryMethod(history) is HistoryMethod.SOE:
        t_max = (n_time * dt) if soe_t_max is None else float(soe_t_max)
        weights = _soe_weights(
            alpha, n_time, n_poles=soe_poles, t_max=t_max, kernel="gl"
        )
        conv = _soe_convolution(values, weights)
    else:
        weights = _gl_weights(alpha, n_time)
        conv = _history_convolution(
            values, weights, history=history, window_steps=window_steps
        )
    result = conv / (dt**alpha)
    if return_info:
        return OperatorResult(
            values=result,
            operator_info=_operator_info(
                family=OperatorFamily.GRUNWALD_LETNIKOV,
                method="full_history_convolution",
                alpha=alpha,
                dt=dt,
                n_steps=n_time,
                history=history,
                diagnostics={
                    "history": str(history),
                    "window_steps": window_steps,
                    "soe_poles": soe_poles,
                    "soe_t_max": (
                        (n_time * dt) if soe_t_max is None else float(soe_t_max)
                    ),
                },
            ),
        )
    return result


def riemann_liouville(
    x: Any,
    *,
    dt: float,
    order: float,
    history: HistoryMethod | str = HistoryMethod.FULL,
    window_steps: int = 64,
    soe_poles: int = 8,
    soe_t_max: float | None = None,
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

    result = grunwald_letnikov(
        x,
        dt=dt,
        order=order,
        history=history,
        window_steps=window_steps,
        soe_poles=soe_poles,
        soe_t_max=soe_t_max,
    )
    if return_info:
        return OperatorResult(
            values=result,
            operator_info=_operator_info(
                family=OperatorFamily.RIEMANN_LIOUVILLE,
                method="grunwald_letnikov_discretisation",
                alpha=validate_order(order),
                dt=dt,
                n_steps=result.shape[0],
                history=history,
                warnings=(
                    "v0.1 Riemann-Liouville is computed through the first-order "
                    "GL full-history uniform-grid discretisation.",
                ),
                diagnostics={
                    "history": str(history),
                    "window_steps": window_steps,
                    "soe_poles": soe_poles,
                    "soe_t_max": (
                        (result.shape[0] * dt)
                        if soe_t_max is None
                        else float(soe_t_max)
                    ),
                },
            ),
        )
    return result


def caputo(
    x: Any,
    *,
    dt: float,
    order: float,
    history: HistoryMethod | str = HistoryMethod.FULL,
    window_steps: int = 64,
    soe_poles: int = 8,
    soe_t_max: float | None = None,
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
    history:
        History convolution strategy.  ``"full"`` (default) uses the dense
        lower-triangular matrix; ``"fft"`` uses an FFT-accelerated causal
        convolution on the L1 increments; ``"short_memory"`` truncates history to
        ``window_steps``; ``"soe"`` uses a sum-of-exponentials kernel
        approximation.  All alternatives are opt-in and validated against the
        ``"full"`` reference.
    window_steps:
        Number of recent time steps retained when ``history="short_memory"``.
    soe_poles:
        Number of exponential poles when ``history="soe"``.
    soe_t_max:
        Physical horizon used for SOE fitting.  Defaults to the signal duration
        ``n_steps * dt``.

    Returns
    -------
    Any
        Approximation with the same shape as ``x``. The first sample is zero
        because the L1 history integral has no elapsed interval at ``t=0``.
    """

    alpha = validate_order(order)
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
                    history=history,
                    diagnostics={
                        "history": str(history),
                        "window_steps": window_steps,
                        "soe_poles": soe_poles,
                        "soe_t_max": (
                            (n_time * dt) if soe_t_max is None else float(soe_t_max)
                        ),
                    },
                ),
            )
        return result

    increments = values[1:] - values[:-1]
    if HistoryMethod(history) is HistoryMethod.SOE:
        t_max = (n_time * dt) if soe_t_max is None else float(soe_t_max)
        n_inc = n_time - 1
        weights = _soe_weights(
            alpha, n_inc, n_poles=soe_poles, t_max=t_max, kernel="l1"
        )
        tail = _soe_convolution(increments, weights)
    else:
        weights = _l1_weights(alpha, n_time - 1)
        tail = _history_convolution(
            increments, weights, history=history, window_steps=window_steps
        )
    scale = 1.0 / (gamma(2.0 - alpha) * (dt**alpha))
    tail = tail * scale
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
                history=history,
                diagnostics={
                    "history": str(history),
                    "window_steps": window_steps,
                    "soe_poles": soe_poles,
                    "soe_t_max": (
                        (n_time * dt) if soe_t_max is None else float(soe_t_max)
                    ),
                },
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
    return (ks + 1.0) ** (1.0 - alpha) - ks ** (1.0 - alpha)


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
    history: HistoryMethod | str = HistoryMethod.FULL,
    warnings: tuple[str, ...] = (),
    diagnostics: dict[str, Any] | None = None,
) -> OperatorInfo:
    method_label = method
    if HistoryMethod(history) is HistoryMethod.FFT and "fft" not in method_label:
        method_label = f"{method_label}_fft"
    return OperatorInfo(
        family=family,
        method=method_label,
        fractional_order=alpha,
        dt=float(dt),
        n_steps=int(n_steps),
        history=HistoryMethod(history),
        diagnostics={**(diagnostics or {}), "history": str(history)},
        warnings=warnings,
    )
