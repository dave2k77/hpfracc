"""Fixed-step Caputo predictor-corrector solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hpfracc.ops.orders import as_order
from hpfracc.solvers.base import SimulationResult, SolverInfo
from hpfracc.typing import DynamicsFn, PyTree


@dataclass(frozen=True, slots=True)
class PredictorCorrector:
    """Configuration for the full-history Caputo PECE solver.

    The implementation targets scalar Caputo initial value problems with
    ``0 < order < 1`` on a positive, uniform timestep grid.
    """

    dt: float
    order: float

    def __post_init__(self) -> None:
        _validate_dt(self.dt)
        # as_order (not validate_order) so a PredictorCorrector can be built with
        # a traced order inside jax.grad/jax.jit; concrete orders are still
        # validated against the open interval.
        as_order(self.order)


def simulate(
    *,
    model: DynamicsFn,
    ts: PyTree,
    solver: PredictorCorrector,
    initial_state: PyTree,
    params: PyTree,
    rng_key: PyTree | None = None,
    inputs: PyTree | None = None,
) -> SimulationResult:
    """Simulate a Caputo FDE with a fixed-step predictor-corrector method.

    The full-history Diethelm predictor-corrector recurrence is evaluated with a
    single ``jax.lax.scan`` over a preallocated history buffer rather than a
    Python loop. This keeps the method ``jax.jit``-traceable and gives a bounded
    autodiff graph whose size does not grow with the number of time steps, while
    reproducing the same numerics as a direct unrolled evaluation. The compute
    cost remains ``O(n**2)`` because each step weights the full history.
    """

    jax = _jax()
    jnp = _jnp()
    times = jnp.asarray(ts)
    _validate_time_grid(times, solver.dt)

    y0 = jnp.asarray(initial_state)
    alpha = as_order(solver.order)
    n_time = int(times.shape[0])
    gamma = _gamma()
    dt_alpha = float(solver.dt) ** alpha
    predictor_scale = dt_alpha / gamma(alpha + 1.0)
    corrector_scale = dt_alpha / gamma(alpha + 2.0)

    indices = jnp.arange(n_time)

    f0 = model(
        times[0],
        y0,
        params,
        rng_key=rng_key,
        inputs=_inputs_at(inputs, 0, n_time),
    )
    history0 = jnp.zeros((n_time, *y0.shape), dtype=f0.dtype).at[0].set(f0)

    def history_dot(weights: PyTree, history: PyTree) -> PyTree:
        return jnp.tensordot(weights, history, axes=([0], [0]))

    def step_fn(history: PyTree, step: PyTree) -> tuple[PyTree, PyTree]:
        valid = indices < step
        # Clamp the lag to >= 1 on masked (future) entries before taking
        # fractional powers, so jnp.where never evaluates a negative base.
        lag = jnp.where(valid, (step - indices).astype(history0.dtype), 1.0)
        step_f = step.astype(history0.dtype)

        predictor_weights = jnp.where(
            valid, lag**alpha - _safe_pow(lag - 1.0, alpha), 0.0
        )
        predicted = y0 + predictor_scale * history_dot(predictor_weights, history)

        predicted_f = model(
            times[step],
            predicted,
            params,
            rng_key=rng_key,
            inputs=_inputs_at(inputs, step, n_time),
        )

        boundary = _safe_pow(step_f - 1.0, alpha + 1.0) - (
            step_f - 1.0 - alpha
        ) * step_f**alpha
        interior = (
            (lag + 1.0) ** (alpha + 1.0)
            + _safe_pow(lag - 1.0, alpha + 1.0)
            - 2.0 * lag ** (alpha + 1.0)
        )
        corrector_weights = jnp.where(
            valid, jnp.where(indices == 0, boundary, interior), 0.0
        )
        corrected = y0 + corrector_scale * (
            history_dot(corrector_weights, history) + predicted_f
        )

        next_f = model(
            times[step],
            corrected,
            params,
            rng_key=rng_key,
            inputs=_inputs_at(inputs, step, n_time),
        )
        return history.at[step].set(next_f), corrected

    _, tail = jax.lax.scan(step_fn, history0, jnp.arange(1, n_time))
    values = jnp.concatenate([y0[jnp.newaxis, ...], tail], axis=0)

    info = SolverInfo(
        name="predictor_corrector",
        method="caputo_pece_full_history",
        fractional_order=alpha,
        step_size=float(solver.dt),
        n_steps=int(n_time - 1),
        diagnostics={"history": "full", "grid": "uniform"},
    )
    return SimulationResult(ts=times, latent_state=values, solver_info=info)


def _inputs_at(inputs: PyTree | None, index: int, n_time: int) -> PyTree | None:
    if inputs is None:
        return None

    jax = _jax()

    def maybe_index(leaf: Any) -> Any:
        value = _jnp().asarray(leaf)
        if value.ndim > 0 and value.shape[0] == n_time:
            return value[index]
        return leaf

    return jax.tree_util.tree_map(maybe_index, inputs)


def _validate_dt(dt: float) -> None:
    if not float(dt) > 0.0:
        msg = f"Expected positive uniform timestep dt, got {dt}."
        raise ValueError(msg)


def _validate_time_grid(times: Any, dt: float) -> None:
    if times.ndim != 1:
        msg = "Expected a one-dimensional time grid."
        raise ValueError(msg)
    if times.shape[0] < 1:
        msg = "Expected at least one time sample."
        raise ValueError(msg)
    if times.shape[0] == 1:
        return

    try:
        import numpy as np

        concrete_times = np.asarray(times)
    except Exception:
        return

    # Tolerate floating-point grid-construction error (a few ULPs of the largest
    # time), scaled to the working dtype, but still reject genuinely non-uniform
    # grids. A fixed atol=1e-8 is below float32 grid resolution and wrongly
    # rejects large uniform grids built in the default single precision.
    dtype = concrete_times.dtype
    eps = float(np.finfo(dtype).eps) if np.issubdtype(dtype, np.floating) else 0.0
    scale = max(abs(float(concrete_times[-1])), 1.0)
    atol = max(16.0 * eps * scale, 1e-12)

    diffs = np.diff(concrete_times)
    if not np.allclose(diffs, diffs[0], rtol=1e-5, atol=atol):
        msg = "Expected a uniform time grid."
        raise ValueError(msg)
    if not np.isclose(float(diffs[0]), float(dt), rtol=1e-5, atol=atol):
        msg = (
            "Expected solver dt to match time-grid spacing; "
            f"got dt={dt} and spacing={float(diffs[0])}."
        )
        raise ValueError(msg)


def _gamma() -> Any:
    """Return the JAX gamma function (differentiable in the fractional order)."""

    try:
        from jax.scipy.special import gamma
    except ModuleNotFoundError as exc:
        msg = (
            "JAX is required for hpfracc.solvers numerical solvers. Install the "
            "package with its runtime dependencies before calling this function."
        )
        raise ModuleNotFoundError(msg) from exc
    return gamma


def _safe_pow(base: Any, exp: Any) -> Any:
    """``base ** exp`` with a finite gradient where ``base == 0``.

    The PECE weights evaluate ``(lag - 1)`` and ``(step - 1)`` powers whose base
    is zero at the most recent lag / first step. The forward value is ``0`` for a
    positive exponent, but plain ``base ** exp`` produces a NaN cotangent there;
    the double-``where`` keeps the gradient finite. Assumes ``exp > 0``.
    """

    jnp = _jnp()
    safe_base = jnp.where(base > 0, base, 1.0)
    return jnp.where(base > 0, safe_base**exp, 0.0)


def _jax() -> Any:
    try:
        import jax
    except ModuleNotFoundError as exc:
        msg = (
            "JAX is required for hpfracc.solvers numerical solvers. Install the "
            "package with its runtime dependencies before calling this function."
        )
        raise ModuleNotFoundError(msg) from exc
    return jax


def _jnp() -> Any:
    try:
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        msg = (
            "JAX is required for hpfracc.solvers numerical solvers. Install the "
            "package with its runtime dependencies before calling this function."
        )
        raise ModuleNotFoundError(msg) from exc
    return jnp
