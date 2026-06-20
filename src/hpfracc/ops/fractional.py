"""Uniform-grid fractional derivative operators.

The v0.1 implementations intentionally use explicit full-history
discretisations. They are baseline numerical references, not yet optimized
large-scale kernels.
"""

from __future__ import annotations

from typing import Any

from hpfracc.ops.base import HistoryMethod, OperatorFamily, OperatorInfo, OperatorResult
from hpfracc.ops.kernels import (
    _history_convolution,
    _soe_convolution,
    _soe_weights,
)
from hpfracc.ops.orders import as_order

_SOE_VECTOR_MSG = (
    "history='soe' does not support vector / per-state fractional orders in v0.1; "
    "use history='full', 'fft', or 'short_memory'."
)
_NONUNIFORM_NON_CAPUTO_MSG = (
    "non-uniform grids (t=) are supported only for caputo in v0.1; "
    "grunwald_letnikov / riemann_liouville require a uniform dt."
)
_NONUNIFORM_HISTORY_MSG = (
    "non-uniform grids (t=) support only history='full'; the L1 weights on a "
    "non-uniform grid are not Toeplitz, so fft / short_memory / soe do not apply."
)
_NONUNIFORM_VECTOR_MSG = (
    "non-uniform grids (t=) support only a scalar fractional order in v0.1."
)
_DT_XOR_T_MSG = (
    "Provide exactly one of dt (uniform grid) or t (non-uniform node array)."
)


def grunwald_letnikov(
    x: Any,
    *,
    dt: float | None = None,
    t: Any | None = None,
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
    t:
        Not supported: the Grunwald-Letnikov discretisation is a uniform-shift
        operator with no non-uniform-grid analog. Passing ``t`` raises
        ``NotImplementedError``; use a uniform ``dt``.
    order:
        Fractional order(s) satisfying ``0 < order < 1`` elementwise. A scalar
        applies one order to every state component; a per-state order
        broadcastable to the trailing state shape ``x.shape[1:]`` gives each
        component its own order, applied independently. ``"soe"`` history does not
        support per-state orders.
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

    if t is not None:
        raise NotImplementedError(_NONUNIFORM_NON_CAPUTO_MSG)
    alpha = as_order(order)
    _validate_dt(dt)

    jnp = _jnp()
    values = jnp.asarray(x)
    _validate_time_axis(values)

    n_time = values.shape[0]
    alpha_w, alpha_s, is_vector = _prepare_order(alpha, values.shape[1:])
    if HistoryMethod(history) is HistoryMethod.SOE:
        if is_vector:
            raise NotImplementedError(_SOE_VECTOR_MSG)
        t_max = (n_time * dt) if soe_t_max is None else float(soe_t_max)
        weights = _soe_weights(
            alpha, n_time, n_poles=soe_poles, t_max=t_max, kernel="gl"
        )
        conv = _soe_convolution(values, weights)
    else:
        weights = _gl_weights(alpha_w, n_time)
        conv = _history_convolution(
            values, weights, history=history, window_steps=window_steps
        )
    result = conv / (dt**alpha_s)
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
    dt: float | None = None,
    t: Any | None = None,
    order: float,
    history: HistoryMethod | str = HistoryMethod.FULL,
    window_steps: int = 64,
    soe_poles: int = 8,
    soe_t_max: float | None = None,
    return_info: bool = False,
) -> Any:
    """Approximate a Riemann-Liouville fractional derivative.

    Passing ``t`` (a non-uniform node array) raises ``NotImplementedError``: v0.1
    computes RL through the uniform-grid Grunwald-Letnikov discretisation, which
    has no non-uniform analog. Use a uniform ``dt``.

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

    if t is not None:
        raise NotImplementedError(_NONUNIFORM_NON_CAPUTO_MSG)
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
                alpha=as_order(order),
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
    dt: float | None = None,
    t: Any | None = None,
    order: float,
    history: HistoryMethod | str = HistoryMethod.FULL,
    window_steps: int = 64,
    soe_poles: int = 8,
    soe_t_max: float | None = None,
    return_info: bool = False,
) -> Any:
    """Approximate a Caputo fractional derivative with the L1 scheme.

    Provide **exactly one** of ``dt`` (a uniform timestep) or ``t`` (an array of
    time nodes for a non-uniform / graded grid). The non-uniform path uses the L1
    product-integration weights derived from the actual node spacings, reducing
    exactly to the uniform ``b_k`` weights on an equispaced grid. It is
    full-history only (``history='full'``) and supports a scalar order only; the
    fft / short_memory / soe accelerations and per-state orders are uniform-grid
    features.

    Parameters
    ----------
    x:
        Samples with leading time axis ``(time, ...)``.
    dt:
        Uniform timestep. Must be positive. Mutually exclusive with ``t``.
    t:
        Time nodes of shape ``(time,)``, strictly increasing, one per sample in
        ``x``. Selects the non-uniform Caputo L1 path. Mutually exclusive with
        ``dt``.
    order:
        Fractional order(s) satisfying ``0 < order < 1`` elementwise. A scalar
        applies one order to every state component; a per-state order
        broadcastable to the trailing state shape ``x.shape[1:]`` gives each
        component its own order, applied independently. ``"soe"`` history and the
        non-uniform (``t=``) path do not support per-state orders.
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

    alpha = as_order(order)

    jnp = _jnp()
    values = jnp.asarray(x)
    _validate_time_axis(values)
    n_time = values.shape[0]

    if (dt is None) == (t is None):
        raise ValueError(_DT_XOR_T_MSG)
    if t is not None:
        return _caputo_nonuniform_op(
            values,
            t,
            alpha,
            history=history,
            return_info=return_info,
        )

    _validate_dt(dt)
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
                        "grid": "uniform",
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

    alpha_w, alpha_s, is_vector = _prepare_order(alpha, values.shape[1:])
    increments = values[1:] - values[:-1]
    if HistoryMethod(history) is HistoryMethod.SOE:
        if is_vector:
            raise NotImplementedError(_SOE_VECTOR_MSG)
        t_max = (n_time * dt) if soe_t_max is None else float(soe_t_max)
        n_inc = n_time - 1
        weights = _soe_weights(
            alpha, n_inc, n_poles=soe_poles, t_max=t_max, kernel="l1"
        )
        tail = _soe_convolution(increments, weights)
    else:
        weights = _l1_weights(alpha_w, n_time - 1)
        tail = _history_convolution(
            increments, weights, history=history, window_steps=window_steps
        )
    scale = 1.0 / (_gamma()(2.0 - alpha_s) * (dt**alpha_s))
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
                    "grid": "uniform",
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


def _prepare_order(alpha: Any, trail_shape: tuple[int, ...]) -> tuple[Any, Any, bool]:
    """Split a (possibly vector) order into weight-axis and scaling forms.

    Returns ``(alpha_weights, alpha_scale, is_vector)``. For a scalar order both
    forms are the scalar itself and ``is_vector`` is ``False``. For a per-state
    order broadcast to the trailing state shape ``trail_shape``, ``alpha_weights``
    is flattened to ``(m,)`` so it indexes the per-feature weight columns, while
    ``alpha_scale`` keeps the trailing shape so that ``dt ** alpha_scale`` and the
    Gamma normalisation broadcast against ``(time, *trail_shape)`` arrays.
    """

    jnp = _jnp()
    a = jnp.asarray(alpha)
    if a.ndim == 0:
        return a, a, False
    a_trail = jnp.broadcast_to(a, trail_shape)
    return a_trail.reshape(-1), a_trail, True


def _normalize_order_field(alpha: Any) -> Any:
    """Render an order as a JSON-friendly scalar/tuple for ``OperatorInfo``.

    Concrete scalars become ``float``; concrete arrays become a tuple of floats.
    A traced order (under ``jax.grad``/``jit`` with ``return_info=True``) is
    returned unchanged as a best-effort fallback.
    """

    import numpy as np

    try:
        concrete = np.asarray(alpha, dtype=float)
    except (TypeError, ValueError):
        return alpha
    if concrete.ndim == 0:
        return float(concrete)
    return tuple(float(a) for a in concrete.reshape(-1))


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


def _validate_dt(dt: float | None) -> None:
    if dt is None:
        raise ValueError(_DT_XOR_T_MSG)
    if not float(dt) > 0.0:
        msg = f"Expected positive uniform timestep dt, got {dt}."
        raise ValueError(msg)


def _validate_time_nodes(t: Any, n_expected: int) -> None:
    """Validate a non-uniform time-node array ``t`` of shape ``(n_expected,)``.

    Shape checks are static; the strictly-increasing / finite checks run on a
    concrete ``numpy`` view and are skipped under tracing (deferred to the eager
    call site), mirroring ``predictor_corrector._validate_time_grid``.
    """

    if t.ndim != 1:
        msg = "Expected a one-dimensional time-node array t."
        raise ValueError(msg)
    if t.shape[0] != n_expected:
        msg = (
            "Expected t to have one node per time sample "
            f"({n_expected}); got {t.shape[0]}."
        )
        raise ValueError(msg)

    try:
        import numpy as np

        concrete = np.asarray(t)
    except Exception:
        return
    if not np.all(np.isfinite(concrete)):
        msg = "Expected finite time nodes t."
        raise ValueError(msg)
    if concrete.shape[0] >= 2 and not np.all(np.diff(concrete) > 0.0):
        msg = "Expected strictly increasing time nodes t."
        raise ValueError(msg)


def _mean_spacing(t: Any) -> float:
    """Best-effort mean node spacing for the ``OperatorInfo.dt`` summary field."""

    jnp = _jnp()
    spacing = jnp.mean(t[1:] - t[:-1]) if t.shape[0] >= 2 else jnp.asarray(0.0)
    try:
        return float(spacing)
    except (TypeError, ValueError):
        return float("nan")


def _caputo_nonuniform(increments: Any, t: Any, alpha: Any) -> Any:
    """Unscaled non-uniform Caputo L1 history sum ``W @ increments``.

    ``W[n, k] = ((t_n - t_k)^(1-alpha) - (t_n - t_{k+1})^(1-alpha)) / (t_{k+1} - t_k)``
    for ``k < n`` and ``0`` otherwise. Returned shape is ``(n, *trail)`` with the
    ``n = 0`` row zero (empty history). The most-recent increment makes
    ``t_n - t_{k+1} = 0``; ``_safe_pow`` keeps the alpha-gradient finite there.
    """

    jnp = _jnp()
    a = jnp.asarray(alpha)
    n = t.shape[0]
    a_minus = t[:, None] - t[None, :-1]  # (n, n-1): t_n - t_k
    a_plus = t[:, None] - t[None, 1:]  # (n, n-1): t_n - t_{k+1}
    spacing = t[1:] - t[:-1]  # (n-1,)
    rows = jnp.arange(n)[:, None]
    cols = jnp.arange(n - 1)[None, :]
    mask = rows > cols  # increment k contributes to row n iff k < n
    weights = jnp.where(
        mask,
        (_safe_pow(a_minus, 1.0 - a) - _safe_pow(a_plus, 1.0 - a)) / spacing,
        0.0,
    )
    inc_2d = jnp.reshape(increments, (n - 1, -1))
    out_2d = weights @ inc_2d
    return jnp.reshape(out_2d, (n, *increments.shape[1:]))


def _caputo_nonuniform_op(
    values: Any,
    t: Any,
    alpha: Any,
    *,
    history: HistoryMethod | str,
    return_info: bool,
) -> Any:
    """Non-uniform Caputo L1 operator path (scalar order, full history only)."""

    jnp = _jnp()
    t_arr = jnp.asarray(t)
    n_time = values.shape[0]
    _validate_time_nodes(t_arr, n_time)
    if HistoryMethod(history) is not HistoryMethod.FULL:
        raise NotImplementedError(_NONUNIFORM_HISTORY_MSG)
    if jnp.asarray(alpha).ndim > 0:
        raise NotImplementedError(_NONUNIFORM_VECTOR_MSG)

    if n_time == 1:
        result = jnp.zeros_like(values)
    else:
        increments = values[1:] - values[:-1]
        history_sum = _caputo_nonuniform(increments, t_arr, alpha)
        result = history_sum / _gamma()(2.0 - jnp.asarray(alpha))

    if not return_info:
        return result
    return OperatorResult(
        values=result,
        operator_info=_operator_info(
            family=OperatorFamily.CAPUTO,
            method="l1_nonuniform",
            alpha=alpha,
            dt=_mean_spacing(t_arr),
            n_steps=n_time,
            history=HistoryMethod.FULL,
            diagnostics={
                "grid": "nonuniform",
                "history": "full",
                "n_nodes": int(n_time),
            },
        ),
    )


def _validate_time_axis(values: Any) -> None:
    if values.ndim < 1:
        msg = "Expected samples with a leading time axis; got a scalar input."
        raise ValueError(msg)
    if values.shape[0] < 1:
        msg = "Expected at least one time sample."
        raise ValueError(msg)


def _gl_weights(alpha: Any, n: int) -> Any:
    """Grunwald-Letnikov binomial weights along a leading lag axis.

    ``alpha`` may be a scalar (shape ``()``) giving weights of shape ``(n,)`` or a
    1-D array of per-state orders (shape ``(m,)``) giving weights of shape
    ``(n, m)``: a trailing order axis is carried through the recurrence so each
    state component gets its own weight column.
    """

    jnp = _jnp()
    alpha = jnp.asarray(alpha)
    if n == 1:
        return jnp.ones((1, *alpha.shape))
    # Reshape the lag axis to broadcast against the trailing order axis.
    ks = jnp.arange(1, n, dtype=jnp.result_type(float)).reshape(
        (-1, *(1,) * alpha.ndim)
    )
    factors = (ks - 1.0 - alpha) / ks
    ones = jnp.ones((1, *alpha.shape), dtype=factors.dtype)
    return jnp.concatenate([ones, jnp.cumprod(factors, axis=0)], axis=0)


def _l1_weights(alpha: Any, n: int) -> Any:
    """L1 Caputo weights along a leading lag axis (scalar or per-state ``alpha``)."""

    jnp = _jnp()
    alpha = jnp.asarray(alpha)
    ks = jnp.arange(n, dtype=jnp.result_type(float)).reshape((-1, *(1,) * alpha.ndim))
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
        fractional_order=_normalize_order_field(alpha),
        dt=float(dt),
        n_steps=int(n_steps),
        history=HistoryMethod(history),
        diagnostics={**(diagnostics or {}), "history": str(history)},
        warnings=warnings,
    )
