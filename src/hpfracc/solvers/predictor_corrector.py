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


@dataclass(frozen=True, slots=True)
class ImplicitPredictorCorrector:
    """Configuration for the full-history **implicit** Caputo solver.

    Unlike :class:`PredictorCorrector` (explicit PECE, one corrector pass), this
    solves the fractional Adams-Moulton corrector equation
    ``y_n = C + corrector_scale * f(t_n, y_n)`` at each step with a fixed number of
    Newton iterations, seeded by the explicit predictor. The implicit treatment of
    the most-recent stage gives a much larger stability region on stiff problems.
    It remains full-history, fixed-step, uniform-grid, ``jit``-traceable with a
    bounded autodiff graph, and differentiable.

    Provisional: targets Caputo initial value problems with ``0 < order < 1`` on a
    positive, uniform timestep grid. Adaptive stepping is a future extension.
    """

    dt: float
    order: float
    newton_iterations: int = 3

    def __post_init__(self) -> None:
        _validate_dt(self.dt)
        as_order(self.order)
        if int(self.newton_iterations) < 1:
            msg = (
                "Expected newton_iterations >= 1 for the implicit solver, "
                f"got {self.newton_iterations}."
            )
            raise ValueError(msg)


def simulate(
    *,
    model: DynamicsFn,
    ts: PyTree,
    solver: PredictorCorrector | ImplicitPredictorCorrector,
    initial_state: PyTree,
    params: PyTree,
    rng_key: PyTree | None = None,
    inputs: PyTree | None = None,
) -> SimulationResult:
    """Simulate a Caputo FDE with a fixed-step predictor-corrector method.

    The full-history Diethelm recurrence is evaluated with a single
    ``jax.lax.scan`` over a preallocated history buffer rather than a Python loop.
    This keeps the method ``jax.jit``-traceable and gives a bounded autodiff graph
    whose size does not grow with the number of time steps. The compute cost
    remains ``O(n**2)`` because each step weights the full history.

    With a :class:`PredictorCorrector` the explicit one-pass PECE method is used;
    with an :class:`ImplicitPredictorCorrector` the corrector equation is solved
    implicitly (fixed-iteration Newton) for a much larger stability region on stiff
    problems. Both share the same predictor/corrector history weights.
    """

    if isinstance(solver, ImplicitPredictorCorrector):
        return _simulate_implicit(
            model=model,
            ts=ts,
            solver=solver,
            initial_state=initial_state,
            params=params,
            rng_key=rng_key,
            inputs=inputs,
        )
    return _simulate_explicit(
        model=model,
        ts=ts,
        solver=solver,
        initial_state=initial_state,
        params=params,
        rng_key=rng_key,
        inputs=inputs,
    )


def _setup(
    *,
    model: DynamicsFn,
    times: PyTree,
    solver: PredictorCorrector | ImplicitPredictorCorrector,
    y0: PyTree,
    params: PyTree,
    rng_key: PyTree | None,
    inputs: PyTree | None,
) -> tuple[Any, ...]:
    """Shared predictor-corrector setup: scales, indices, and seeded history."""

    jnp = _jnp()
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
    return alpha, n_time, predictor_scale, corrector_scale, indices, history0


def _pece_weights(
    step: PyTree, indices: PyTree, alpha: Any, dtype: Any
) -> tuple[PyTree, PyTree]:
    """Diethelm predictor (Adams-Bashforth) and corrector (Adams-Moulton) weights.

    Shared by the explicit and implicit paths so the validated weight formulas are
    defined once. Future (masked) lags are clamped to ``>= 1`` before fractional
    powers so ``jnp.where`` never evaluates a negative base.
    """

    jnp = _jnp()
    valid = indices < step
    lag = jnp.where(valid, (step - indices).astype(dtype), 1.0)
    step_f = step.astype(dtype)

    predictor_weights = jnp.where(
        valid, lag**alpha - _safe_pow(lag - 1.0, alpha), 0.0
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
    return predictor_weights, corrector_weights


def _history_dot(weights: PyTree, history: PyTree) -> PyTree:
    return _jnp().tensordot(weights, history, axes=([0], [0]))


def _simulate_explicit(
    *,
    model: DynamicsFn,
    ts: PyTree,
    solver: PredictorCorrector,
    initial_state: PyTree,
    params: PyTree,
    rng_key: PyTree | None = None,
    inputs: PyTree | None = None,
) -> SimulationResult:
    jax = _jax()
    jnp = _jnp()
    times = jnp.asarray(ts)
    _validate_time_grid(times, solver.dt)
    y0 = jnp.asarray(initial_state)
    alpha, n_time, predictor_scale, corrector_scale, indices, history0 = _setup(
        model=model,
        times=times,
        solver=solver,
        y0=y0,
        params=params,
        rng_key=rng_key,
        inputs=inputs,
    )

    def step_fn(history: PyTree, step: PyTree) -> tuple[PyTree, PyTree]:
        predictor_weights, corrector_weights = _pece_weights(
            step, indices, alpha, history0.dtype
        )
        predicted = y0 + predictor_scale * _history_dot(predictor_weights, history)
        predicted_f = model(
            times[step],
            predicted,
            params,
            rng_key=rng_key,
            inputs=_inputs_at(inputs, step, n_time),
        )
        corrected = y0 + corrector_scale * (
            _history_dot(corrector_weights, history) + predicted_f
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


def _simulate_implicit(
    *,
    model: DynamicsFn,
    ts: PyTree,
    solver: ImplicitPredictorCorrector,
    initial_state: PyTree,
    params: PyTree,
    rng_key: PyTree | None = None,
    inputs: PyTree | None = None,
) -> SimulationResult:
    jax = _jax()
    jnp = _jnp()
    times = jnp.asarray(ts)
    _validate_time_grid(times, solver.dt)
    y0 = jnp.asarray(initial_state)
    alpha, n_time, predictor_scale, corrector_scale, indices, history0 = _setup(
        model=model,
        times=times,
        solver=solver,
        y0=y0,
        params=params,
        rng_key=rng_key,
        inputs=inputs,
    )
    iterations = int(solver.newton_iterations)

    def step_fn(history: PyTree, step: PyTree) -> tuple[PyTree, PyTree]:
        predictor_weights, corrector_weights = _pece_weights(
            step, indices, alpha, history0.dtype
        )
        # Explicit predictor seeds the Newton solve; ``known`` is the part of the
        # corrector that depends only on the (already computed) history.
        predicted = y0 + predictor_scale * _history_dot(predictor_weights, history)
        known = y0 + corrector_scale * _history_dot(corrector_weights, history)

        def f_at(state: PyTree) -> PyTree:
            return model(
                times[step],
                state,
                params,
                rng_key=rng_key,
                inputs=_inputs_at(inputs, step, n_time),
            )

        corrected = _newton_corrector(
            f_at, known, corrector_scale, predicted, iterations
        )
        next_f = f_at(corrected)
        return history.at[step].set(next_f), corrected

    _, tail = jax.lax.scan(step_fn, history0, jnp.arange(1, n_time))
    values = jnp.concatenate([y0[jnp.newaxis, ...], tail], axis=0)

    info = SolverInfo(
        name="implicit_predictor_corrector",
        method="caputo_implicit_adams_moulton",
        fractional_order=alpha,
        step_size=float(solver.dt),
        n_steps=int(n_time - 1),
        diagnostics={
            "history": "full",
            "grid": "uniform",
            "scheme": "implicit",
            "newton_iterations": iterations,
        },
    )
    return SimulationResult(ts=times, latent_state=values, solver_info=info)


def _newton_corrector(
    f_at: Any,
    known: PyTree,
    scale: Any,
    guess: PyTree,
    iterations: int,
) -> PyTree:
    """Solve ``y = known + scale * f_at(y)`` by fixed-iteration Newton.

    ``residual(y) = y - known - scale * f_at(y)``; each iteration forms the dense
    state Jacobian, solves the flattened linear system, and updates. The iteration
    count is fixed, so the unrolled graph stays bounded and ``jit``/``grad``-safe.
    For a linear ``f_at`` one iteration is exact.
    """

    jax = _jax()
    jnp = _jnp()
    shape = guess.shape
    size = int(guess.size)

    def residual(state: PyTree) -> PyTree:
        return state - known - scale * f_at(state)

    state = guess
    for _ in range(iterations):
        g = residual(state)
        jac = jax.jacobian(residual)(state).reshape(size, size)
        delta = jnp.linalg.solve(jac, g.reshape(size))
        state = state - delta.reshape(shape)
    return state


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
