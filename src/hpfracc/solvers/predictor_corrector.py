"""Fixed-step Caputo predictor-corrector solvers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from hpfracc.ops._quadrature import _power_interval_moments, _time_difference
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

    ``newton_rtol`` and ``newton_atol`` assess the final corrector residual;
    they do not change the fixed iteration count or update rule. Diagnostics
    report per-step convergence and scaled residuals, including under JIT.

    Provisional: targets Caputo initial value problems with ``0 < order < 1`` on a
    positive, uniform timestep grid. Adaptive stepping is a future extension.
    """

    dt: float
    order: float
    newton_iterations: int = 3
    newton_rtol: float = 1e-6
    newton_atol: float = 1e-8

    def __post_init__(self) -> None:
        _validate_dt(self.dt)
        as_order(self.order)
        if int(self.newton_iterations) < 1:
            msg = (
                "Expected newton_iterations >= 1 for the implicit solver, "
                f"got {self.newton_iterations}."
            )
            raise ValueError(msg)
        for name in ("newton_rtol", "newton_atol"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                msg = f"Expected finite positive {name}, got {value}."
                raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class NonUniformPredictorCorrector:
    """Configuration for the explicit Caputo PECE solver on a **non-uniform** grid.

    Unlike :class:`PredictorCorrector` (uniform timestep ``dt``), this uses the
    variable-step fractional Adams-Bashforth-Moulton weights built from the actual
    node spacings of ``ts``, so any strictly-increasing time grid is admissible. On
    an equispaced grid it reduces exactly to :class:`PredictorCorrector`. Its
    purpose is a-priori graded meshes ``t_j = T (j/N)**r`` clustered near ``t=0``,
    which recover the accuracy a uniform grid loses to the solution's weak
    ``t**order`` singularity at the origin.

    There is no ``dt``: the grid is taken from the ``ts`` passed to
    :func:`simulate`. Provisional; explicit one-pass PECE, full-history,
    ``jit``-traceable with a bounded autodiff graph, and differentiable.
    """

    order: float

    def __post_init__(self) -> None:
        as_order(self.order)


@dataclass(frozen=True, slots=True)
class AdaptivePredictorCorrector:
    """Configuration for the **a-posteriori adaptive-step** Caputo PECE solver.

    This grows/shrinks steps using a **step-doubling indicator** -- one step of
    ``h`` versus two of ``h/2`` -- scaled by ``rtol``/``atol``. Its Richardson-like
    normalization is heuristic, not a calibrated bound for the retained coarse
    step: the quadratic-forcing first-panel audit demonstrates underestimation.
    Both trials reuse older history, whose common error is invisible to their
    difference. See ``docs/validation/adaptive-calibration.md`` for the independent
    identities, tolerance sweeps, and limits of this provisional controller.

    ``error_estimator="subdivision"`` is an experimental opt-in safeguard. It
    compares one full step, two half steps, and four quarter steps using the
    maximum componentwise normalized pair discrepancy, without Richardson
    rescaling. A bounded-factor controller avoids assuming one universal startup
    power. Extra probes can detect pair cancellation, but do not establish a
    local/global error bound or repair common old-history error. The retained
    trajectory is still the coarse PECE path on the chosen mesh.

    The realized mesh is data-dependent, so for a bounded, ``jit``-traceable,
    reverse-differentiable graph the integration runs as a **fixed-length
    ``lax.scan`` of ``max_steps`` masked iterations** (the diffrax pattern), not a
    ``while_loop``. Gradients flow through the solution values on the *realized*
    grid; the step-size controller and accept/reject decisions are treated as
    non-differentiable (mesh positions are ``stop_gradient``-ed), matching
    discretize-then-optimize. It remains a correct full-history ``O(N**2)``
    reference -- rejected steps re-evaluate history; the sum-of-exponentials
    acceleration is a future stage.

    ``simulate``'s ``ts`` supplies only the integration **bounds**
    ``(t0, t_end) = (ts[0], ts[-1])``; the interior nodes are chosen adaptively.

    ``max_steps`` bounds attempted steps, including rejections. Results report
    completion, budget exhaustion, non-finite trials, and step-size limits, with
    a fixed-shape mask identifying the retained output prefix. Numerical status
    remains accessible under JIT. Failed trials are not appended to the history;
    success means completion with finite accepted trials, not a global error bound.

    Provisional: targets Caputo initial value problems with ``0 < order < 1``.
    """

    order: float
    rtol: float = 1e-4
    atol: float = 1e-6
    max_steps: int = 4096
    first_step: float | None = None
    safety: float = 0.9
    min_factor: float = 0.2
    max_factor: float = 5.0
    error_estimator: str = "step_doubling"

    def __post_init__(self) -> None:
        as_order(self.order)
        if self.error_estimator not in ("step_doubling", "subdivision"):
            msg = f"Unknown error_estimator: {self.error_estimator!r}."
            raise ValueError(msg)
        for name in (
            "rtol", "atol", "safety", "min_factor", "max_factor", "first_step"
        ):
            value = getattr(self, name)
            if value is not None and not isfinite(float(value)):
                msg = f"Expected finite {name}, got {value}."
                raise ValueError(msg)
        if not float(self.rtol) > 0.0 or not float(self.atol) > 0.0:
            msg = (
                "Expected positive rtol/atol, got "
                f"rtol={self.rtol}, atol={self.atol}."
            )
            raise ValueError(msg)
        if int(self.max_steps) < 2:
            msg = f"Expected max_steps >= 2, got {self.max_steps}."
            raise ValueError(msg)
        if self.first_step is not None and not float(self.first_step) > 0.0:
            msg = f"Expected positive first_step, got {self.first_step}."
            raise ValueError(msg)
        if not 0.0 < float(self.safety) <= 1.0:
            msg = f"Expected safety in (0, 1], got {self.safety}."
            raise ValueError(msg)
        if not 0.0 < float(self.min_factor) < 1.0 < float(self.max_factor):
            msg = (
                "Expected min_factor < 1 < max_factor, got "
                f"min_factor={self.min_factor}, max_factor={self.max_factor}."
            )
            raise ValueError(msg)


def simulate(
    *,
    model: DynamicsFn,
    ts: PyTree,
    solver: PredictorCorrector
    | ImplicitPredictorCorrector
    | NonUniformPredictorCorrector
    | AdaptivePredictorCorrector,
    initial_state: PyTree,
    params: PyTree,
    rng_key: PyTree | None = None,
    inputs: PyTree | None = None,
) -> SimulationResult:
    """Simulate a Caputo FDE with a predictor-corrector method.

    The full-history Diethelm recurrence is evaluated with a single
    ``jax.lax.scan`` over a preallocated history buffer rather than a Python loop.
    This keeps the method ``jax.jit``-traceable and gives a bounded autodiff graph
    whose size does not grow with the number of time steps. The compute cost
    remains ``O(n**2)`` because each step weights the full history.

    With a :class:`PredictorCorrector` the explicit one-pass PECE method is used;
    with an :class:`ImplicitPredictorCorrector` the corrector equation is solved
    implicitly (fixed-iteration Newton) for a much larger stability region on stiff
    problems; with a :class:`NonUniformPredictorCorrector` the variable-step
    fractional Adams weights are built from the (non-uniform) ``ts`` node spacings;
    with an :class:`AdaptivePredictorCorrector` the step size is chosen on the fly
    from a step-doubling local-error estimate (``ts`` then supplies only the
    integration bounds ``(ts[0], ts[-1])``).
    """

    if isinstance(solver, AdaptivePredictorCorrector):
        return _simulate_adaptive(
            model=model,
            ts=ts,
            solver=solver,
            initial_state=initial_state,
            params=params,
            rng_key=rng_key,
            inputs=inputs,
        )
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
    if isinstance(solver, NonUniformPredictorCorrector):
        return _simulate_nonuniform(
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
        residual = corrected - known - corrector_scale * next_f
        tolerance = solver.newton_atol + solver.newton_rtol * jnp.maximum(
            jnp.abs(corrected), jnp.abs(known)
        )
        scaled = jnp.max(jnp.abs(residual) / tolerance)
        finite = (
            jnp.all(jnp.isfinite(corrected))
            & jnp.all(jnp.isfinite(known))
            & jnp.all(jnp.isfinite(next_f))
            & jnp.all(jnp.isfinite(tolerance))
            & jnp.isfinite(scaled)
        )
        # Diagnostics neither alter Newton updates nor add gradient paths.
        scaled = jax.lax.stop_gradient(jnp.where(finite, scaled, jnp.inf))
        return history.at[step].set(next_f), (corrected, scaled, finite)

    _, (tail, residuals, finite_steps) = jax.lax.scan(
        step_fn, history0, jnp.arange(1, n_time)
    )
    values = jnp.concatenate([y0[jnp.newaxis, ...], tail], axis=0)

    converged_steps = residuals <= 1.0
    failure_count = jnp.sum(~converged_steps)
    failures = _maybe_concrete_int(failure_count)
    warnings = ()
    if failures is not None and failures > 0:
        warnings = (
            f"Implicit Newton corrector failed residual tolerances on {failures} "
            "step(s); inspect the diagnostics before using this trajectory. "
            "The solver retained its fixed iteration count and did not retry.",
        )

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
            "newton_rtol": float(solver.newton_rtol),
            "newton_atol": float(solver.newton_atol),
            "newton_converged": _diagnostic_value(jnp.all(converged_steps)),
            "newton_failure_count": _diagnostic_value(failure_count),
            "newton_nonfinite_count": _diagnostic_value(jnp.sum(~finite_steps)),
            "newton_step_converged": _diagnostic_value(converged_steps),
            "newton_scaled_residuals": _diagnostic_value(residuals),
            "newton_max_scaled_residual": _diagnostic_value(
                jnp.max(residuals, initial=0.0)
            ),
        },
        warnings=warnings,
    )
    return SimulationResult(ts=times, latent_state=values, solver_info=info)


def _diagnostic_value(value: Any) -> Any:
    """Export scalar/vector diagnostics eagerly, preserving arrays under JIT.

    Non-finite floating values become JSON null eagerly; their corresponding
    failure flags remain explicit. Traced values retain their fixed shape and
    dtype so callers can return diagnostics from compiled simulations.
    """
    import numpy as np

    try:
        concrete = np.asarray(value).tolist()
    except (TypeError, ValueError):
        return value

    def finite_or_none(item: Any) -> Any:
        return None if isinstance(item, float) and not isfinite(item) else item

    if isinstance(concrete, list):
        return [finite_or_none(item) for item in concrete]
    return finite_or_none(concrete)


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


def _nonuniform_pece_weights(
    step: PyTree, times: PyTree, indices: PyTree, alpha: Any, dtype: Any
) -> tuple[PyTree, PyTree, PyTree]:
    """Variable-step fractional Adams predictor/corrector weights at node ``step``.

    Returns ``(predictor_weights, corrector_weights, diagonal_weight)``, each
    already scaled by ``1 / Gamma(alpha)``. ``predictor_weights`` and
    ``corrector_weights`` are length-``n_time`` arrays. The predictor is zero at
    indices ``>= step``; the corrector includes the diagonal at ``step`` and is
    zero beyond it. Callers keep history at ``step`` zero and apply the separately
    returned ``diagonal_weight`` to ``f(t_n, y_predicted)``. Moment checks must
    therefore use ``corrector_weights[:step]`` plus the diagonal, not count it
    twice. On a uniform grid these collapse to the fixed-step
    Diethelm weights. Derived from product integration of the Volterra form with a
    piecewise-constant predictor and piecewise-linear corrector; the singular
    ``(t-s)**(alpha-1)`` integrates to ``**alpha`` / ``**(alpha+1)`` powers.
    Stable interval moments avoid cancellation on graded meshes and preserve
    finite order gradients at the most-recent interval's zero lower lag.
    """

    jnp = _jnp()
    # Keep normalization and powers in the history's numerical dtype even when
    # x64 is enabled and the order was supplied as a Python scalar.
    alpha = jnp.asarray(alpha, dtype=dtype)
    t_n = times[step]
    t_next = jnp.concatenate([times[1:], times[-1:]])  # t_{j+1}; last is masked
    a_upper = _time_difference(t_n, times).astype(dtype)  # A_j = t_n - t_j
    b_upper = _time_difference(t_n, t_next).astype(dtype)  # B_j = t_n - t_{j+1}
    spacing = _time_difference(t_next, times).astype(dtype)  # h_j (last masked)

    valid = indices < step  # interval j is used iff j < step
    predictor, i0, i1 = _power_interval_moments(
        a_upper, b_upper, spacing, alpha, valid
    )

    # Node j collects I0[j] (left endpoint of interval j) and I1[j-1] (right
    # endpoint of interval j-1). The diagonal node ``step`` keeps I1[step-1].
    i1_shifted = jnp.concatenate([jnp.zeros_like(i1[:1]), i1[:-1]])
    corrector = i0 + i1_shifted  # already masked: i0/i1 are zero where invalid
    diagonal = i1[step - 1]

    inv_gamma = 1.0 / _gamma()(alpha)
    return predictor * inv_gamma, corrector * inv_gamma, diagonal * inv_gamma


def _simulate_nonuniform(
    *,
    model: DynamicsFn,
    ts: PyTree,
    solver: NonUniformPredictorCorrector,
    initial_state: PyTree,
    params: PyTree,
    rng_key: PyTree | None = None,
    inputs: PyTree | None = None,
) -> SimulationResult:
    jax = _jax()
    jnp = _jnp()
    times = jnp.asarray(ts)
    _validate_increasing_grid(times)
    y0 = jnp.asarray(initial_state)
    alpha = as_order(solver.order)
    n_time = int(times.shape[0])
    indices = jnp.arange(n_time)

    f0 = model(
        times[0],
        y0,
        params,
        rng_key=rng_key,
        inputs=_inputs_at(inputs, 0, n_time),
    )
    history0 = jnp.zeros((n_time, *y0.shape), dtype=f0.dtype).at[0].set(f0)

    def step_fn(history: PyTree, step: PyTree) -> tuple[PyTree, PyTree]:
        predictor_weights, corrector_weights, diagonal = _nonuniform_pece_weights(
            step, times, indices, alpha, history0.dtype
        )
        predicted = y0 + _history_dot(predictor_weights, history)
        predicted_f = model(
            times[step],
            predicted,
            params,
            rng_key=rng_key,
            inputs=_inputs_at(inputs, step, n_time),
        )
        corrected = (
            y0 + _history_dot(corrector_weights, history) + diagonal * predicted_f
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
        name="nonuniform_predictor_corrector",
        method="caputo_pece_nonuniform",
        fractional_order=alpha,
        step_size=None,
        n_steps=int(n_time - 1),
        diagnostics={"history": "full", "grid": "nonuniform"},
    )
    return SimulationResult(ts=times, latent_state=values, solver_info=info)


def _adaptive_pc_at(
    *,
    model: DynamicsFn,
    y0: PyTree,
    t_target: Any,
    step: Any,
    times_eff: PyTree,
    f_hist_eff: PyTree,
    indices: PyTree,
    alpha: Any,
    dtype: Any,
    params: PyTree,
    rng_key: PyTree | None,
    inputs: PyTree | None,
    n_eval: int,
) -> PyTree:
    """One full PECE step targeting ``t_target`` placed at index ``step``.

    ``times_eff`` / ``f_hist_eff`` are the (masked) history buffers with the trial
    node(s) already written in, so the validated :func:`_nonuniform_pece_weights`
    is reused verbatim -- the adaptive solver introduces no new weight formulas.
    The mesh positions are frozen for autodiff by the caller (``stop_gradient``);
    gradients flow through ``y0`` / ``f_hist_eff`` only.
    """

    predictor_weights, corrector_weights, diagonal = _nonuniform_pece_weights(
        step, times_eff, indices, alpha, dtype
    )
    predicted = y0 + _history_dot(predictor_weights, f_hist_eff)
    predicted_f = model(
        t_target,
        predicted,
        params,
        rng_key=rng_key,
        inputs=_inputs_at(inputs, n_eval, n_eval + 1),
    )
    return y0 + _history_dot(corrector_weights, f_hist_eff) + diagonal * predicted_f


def _simulate_adaptive(
    *,
    model: DynamicsFn,
    ts: PyTree,
    solver: AdaptivePredictorCorrector,
    initial_state: PyTree,
    params: PyTree,
    rng_key: PyTree | None = None,
    inputs: PyTree | None = None,
) -> SimulationResult:
    jax = _jax()
    jnp = _jnp()
    times_in = jnp.asarray(ts)
    _validate_increasing_grid(times_in)
    y0 = jnp.asarray(initial_state)
    # Mesh positions and the controller must share a floating dtype throughout
    # lax.scan. Promote integer bounds and retain the precision of either input.
    time_dtype = jnp.result_type(times_in.dtype, y0.real.dtype, float)
    requested_t_end = times_in[-1]
    bounds_exact = jnp.asarray(True)
    if jnp.issubdtype(times_in.dtype, jnp.integer):
        original_bounds = jnp.stack([times_in[0], times_in[-1]])
        floating_bounds = original_bounds.astype(time_dtype)
        bounds_exact = jnp.all(
            floating_bounds.astype(times_in.dtype) == original_bounds
        )
        integer_info = jnp.iinfo(times_in.dtype)
        if integer_info.bits > jnp.finfo(time_dtype).nmant + 1:
            # Float-to-int casts saturate at the maximum integer. A round trip
            # alone can therefore falsely certify a rounded-up maximum bound.
            maximum = jnp.asarray(integer_info.max, dtype=times_in.dtype)
            bounds_exact = bounds_exact & jnp.all(original_bounds != maximum)
    times_in = times_in.astype(time_dtype)
    alpha = as_order(solver.order)
    t0 = times_in[0]
    t_end = times_in[-1]
    span = t_end - t0

    max_steps = int(solver.max_steps)
    subdivision = solver.error_estimator == "subdivision"
    # Trial-only slots must also fit on the last allowed attempt. Public output
    # retains max_steps+2 slots in either mode; extra quarter nodes are discarded.
    buf_len = max_steps + (4 if subdivision else 2)
    ctrl_dtype = time_dtype
    rtol = jnp.asarray(solver.rtol, dtype=ctrl_dtype)
    atol = jnp.asarray(solver.atol, dtype=rtol.dtype)
    safety = jnp.asarray(solver.safety, dtype=rtol.dtype)
    min_factor = jnp.asarray(solver.min_factor, dtype=rtol.dtype)
    max_factor = jnp.asarray(solver.max_factor, dtype=rtol.dtype)
    # Heuristic normalization exponent; not a calibrated retained-coarse bound.
    order_q = jnp.asarray(1.0 + alpha, dtype=ctrl_dtype)
    richardson = 2.0**order_q - 1.0

    if solver.first_step is None:
        h0 = jnp.asarray(0.01, dtype=rtol.dtype) * span
    else:
        h0 = jnp.asarray(solver.first_step, dtype=rtol.dtype)
    h0 = jnp.minimum(h0, span)
    # Relative to the integration span, not a unit-scale floor. Representability
    # is checked separately before accepting any full/half-step trial.
    h_min = jnp.asarray(1e-12, dtype=rtol.dtype) * jnp.abs(span)

    f0 = model(
        t0, y0, params, rng_key=rng_key, inputs=_inputs_at(inputs, 0, 1)
    )
    # A lower-precision dynamics result must not narrow the stored state.
    dtype = jnp.result_type(y0.dtype, f0.dtype, float)
    node_times0 = jnp.zeros((buf_len,), dtype=times_in.dtype).at[0].set(t0)
    y_hist0 = jnp.zeros((buf_len, *y0.shape), dtype=dtype).at[0].set(y0)
    f_hist0 = jnp.zeros((buf_len, *y0.shape), dtype=dtype).at[0].set(f0)
    indices = jnp.arange(buf_len)

    def weighted_error(y_big: PyTree, y_small: PyTree) -> Any:
        err = jnp.abs(y_big - y_small) / richardson
        scale = atol + rtol * jnp.maximum(jnp.abs(y_big), jnp.abs(y_small))
        normalized = jnp.sqrt(jnp.mean((err / scale) ** 2))
        valid_scale = jnp.all(jnp.isfinite(scale) & (scale > 0))
        return jnp.where(valid_scale, normalized, jnp.inf)

    def attempt(carry: Any) -> Any:
        (n_acc, t_cur, h, node_times, y_hist, f_hist, n_rej,
         done, nonfinite, underflow) = carry
        sg = jax.lax.stop_gradient
        # Mesh / controller scalars are frozen for autodiff (diffrax pattern).
        t_cur_f = sg(t_cur)
        h_f = sg(h)
        t_trial = jnp.minimum(t_cur_f + h_f, t_end)
        h_act = t_trial - t_cur_f
        t_mid = t_cur_f + 0.5 * h_act
        resolved = (t_cur_f < t_mid) & (t_mid < t_trial)
        m = n_acc

        # Big step (one step of h) -- frozen mesh positions for the weights.
        times_big = node_times.at[m + 1].set(t_trial)
        y_big = _adaptive_pc_at(
            model=model, y0=y0, t_target=t_trial, step=m + 1,
            times_eff=sg(times_big), f_hist_eff=f_hist, indices=indices,
            alpha=alpha, dtype=dtype, params=params, rng_key=rng_key,
            inputs=inputs, n_eval=0,
        )
        # Two steps of h/2.
        times_mid = node_times.at[m + 1].set(t_mid)
        y_mid = _adaptive_pc_at(
            model=model, y0=y0, t_target=t_mid, step=m + 1,
            times_eff=sg(times_mid), f_hist_eff=f_hist, indices=indices,
            alpha=alpha, dtype=dtype, params=params, rng_key=rng_key,
            inputs=inputs, n_eval=0,
        )
        f_mid = model(
            t_mid, y_mid, params, rng_key=rng_key, inputs=_inputs_at(inputs, 0, 1)
        )
        times_small = node_times.at[m + 1].set(t_mid).at[m + 2].set(t_trial)
        f_hist_small = f_hist.at[m + 1].set(f_mid)
        y_small = _adaptive_pc_at(
            model=model, y0=y0, t_target=t_trial, step=m + 2,
            times_eff=sg(times_small), f_hist_eff=f_hist_small, indices=indices,
            alpha=alpha, dtype=dtype, params=params, rng_key=rng_key,
            inputs=inputs, n_eval=0,
        )

        quarter_finite = jnp.asarray(True)
        if subdivision:
            quarter_times, quarter_history = node_times, f_hist
            previous_target = t_cur_f
            for part in range(1, 5):
                target = t_trial if part == 4 else t_cur_f + (part / 4) * h_act
                resolved = resolved & (previous_target < target)
                quarter_times = quarter_times.at[m + part].set(target)
                quarter_value = _adaptive_pc_at(
                    model=model, y0=y0, t_target=target, step=m + part,
                    times_eff=sg(quarter_times), f_hist_eff=quarter_history,
                    indices=indices, alpha=alpha, dtype=dtype, params=params,
                    rng_key=rng_key, inputs=inputs, n_eval=0,
                )
                quarter_f = model(
                    target, quarter_value, params, rng_key=rng_key,
                    inputs=_inputs_at(inputs, 0, 1),
                )
                quarter_finite = (
                    quarter_finite & jnp.all(jnp.isfinite(quarter_value))
                    & jnp.all(jnp.isfinite(quarter_f))
                )
                quarter_history = quarter_history.at[m + part].set(quarter_f)
                previous_target = target
            difference = jnp.maximum(
                jnp.abs(y_big - y_small),
                jnp.maximum(jnp.abs(y_big - quarter_value),
                            jnp.abs(y_small - quarter_value)),
            )
            scale = atol + rtol * jnp.maximum(
                jnp.maximum(jnp.abs(y_big), jnp.abs(y_small)),
                jnp.abs(quarter_value),
            )
            valid_scale = jnp.all(jnp.isfinite(scale) & (scale > 0))
            err = sg(jnp.where(valid_scale, jnp.max(difference / scale), jnp.inf))
        else:
            err = sg(weighted_error(y_big, y_small))
        # min(..., t_end) clips overshoots exactly: an absolute endpoint tolerance
        # would incorrectly declare tiny integration spans complete too early.
        reached = t_trial >= t_end

        # No local extrapolation: advance with the single full step ``y_big``.
        # The indicator is not a calibrated bound on its error. The half-step
        # ``y_small`` is used only for the error estimate. This makes the realized
        # trajectory identical to the non-uniform solver run on the realized mesh,
        # so gradients are the exact frozen-mesh sensitivity.
        idx = m + 1
        new_f = model(
            t_trial, y_big, params, rng_key=rng_key, inputs=_inputs_at(inputs, 0, 1)
        )
        finite = (
            jnp.all(jnp.isfinite(y_big))
            & jnp.all(jnp.isfinite(y_mid))
            & jnp.all(jnp.isfinite(y_small))
            & jnp.all(jnp.isfinite(f_mid))
            & jnp.all(jnp.isfinite(new_f))
            & jnp.isfinite(err)
            & quarter_finite
        )
        nonfinite = nonfinite | (resolved & ~finite)
        underflow = underflow | ~resolved | (finite & (err > 1.0) & (h_act <= h_min))
        accept = resolved & finite & (err <= 1.0)
        node_times = node_times.at[idx].set(
            jnp.where(accept, t_trial, node_times[idx])
        )
        y_hist = y_hist.at[idx].set(jnp.where(accept, y_big, y_hist[idx]))
        f_hist = f_hist.at[idx].set(jnp.where(accept, new_f, f_hist[idx]))
        n_acc = n_acc + jnp.where(accept, 1, 0)
        n_rej = n_rej + jnp.where(accept, 0, 1)
        t_cur = jnp.where(accept, t_trial, t_cur)
        done = done | (accept & reached)

        # Integral step-size controller; frozen w.r.t. autodiff. ``factor`` grows
        # the step on accept and shrinks it on reject; clamped both sides.
        safe_err = jnp.maximum(err, 1e-16)
        if subdivision:
            # Experimental bounded proposals, not an assumed universal power:
            # reject -> half; comfortably accepted -> double; otherwise hold,
            # all multiplied by safety and constrained by the configured limits.
            proposal = jnp.where(err > 1, 0.5, jnp.where(err < 0.25, 2.0, 1.0))
            factor = jnp.clip(safety * proposal, min_factor, max_factor)
        else:
            factor = jnp.clip(
                safety * safe_err ** (-1.0 / (order_q + 1.0)), min_factor, max_factor
            )
        h_new = jnp.minimum(h_f * factor, jnp.maximum(t_end - t_cur, h_min))
        # Higher-precision dynamics may promote the error estimate and factor;
        # round the proposed step back to the mesh dtype at the carry boundary.
        h = jnp.where(done | nonfinite | underflow, h, h_new).astype(ctrl_dtype)

        return (n_acc, t_cur, h, node_times, y_hist, f_hist, n_rej,
                done, nonfinite, underflow)

    def iteration(carry: Any, _: Any) -> tuple[Any, Any]:
        done, nonfinite, underflow = carry[-3:]
        # The scan remains bounded, but no trials occur after a terminal status.
        # This also avoids NaN gradient paths through unused post-completion work.
        active = ~(done | nonfinite | underflow)
        return jax.lax.cond(active, attempt, lambda state: state, carry), None

    initial_finite = (
        jnp.all(jnp.isfinite(y_hist0[0]))
        & jnp.all(jnp.isfinite(f_hist0[0]))
        & jnp.isfinite(span)
    )
    zero_duration = times_in.shape[0] == 1
    initial_underflow = initial_finite & (
        ~bounds_exact | ((span <= 0) & ~jnp.asarray(zero_duration))
    )
    init = (
        jnp.asarray(0, dtype=jnp.int32),
        t0,
        h0,
        node_times0,
        y_hist0,
        f_hist0,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(zero_duration) & initial_finite & bounds_exact,
        ~initial_finite,
        initial_underflow,
    )
    final, _ = jax.lax.scan(iteration, init, None, length=max_steps)
    (n_acc, t_cur, _h, node_times, y_hist, f_hist, n_rej,
     done, nonfinite, underflow) = final

    # Trim to the realized mesh: pad the unused tail with the final node / state so
    # ``ts`` stays monotone non-decreasing and ``latent_state`` is well-defined.
    valid = indices <= n_acc
    final_t = node_times[n_acc]
    final_y = y_hist[n_acc]
    node_times = jnp.where(valid, node_times, final_t)
    y_hist = jnp.where(valid.reshape((-1,) + (1,) * y0.ndim), y_hist, final_y)
    if subdivision:
        node_times = node_times[:max_steps + 2]
        y_hist = y_hist[:max_steps + 2]
        valid = valid[:max_steps + 2]

    exhausted = ~(done | nonfinite | underflow)
    warnings = ()
    if _maybe_concrete_bool(nonfinite) is True:
        warnings = ("Adaptive integration stopped on non-finite state, dynamics, "
                    "or error normalization; inspect the retained trajectory.",)
    elif _maybe_concrete_bool(underflow) is True:
        warnings = ("Adaptive integration cannot resolve a tolerable step at the "
                    "minimum step size or time precision; integration is incomplete.",)
    elif _maybe_concrete_bool(exhausted) is True:
        warnings = ("Adaptive integration exhausted max_steps before reaching "
                    "the requested endpoint; integration is incomplete.",)

    info = SolverInfo(
        name="adaptive_predictor_corrector",
        method="caputo_adaptive_pece",
        fractional_order=alpha,
        step_size=None,
        n_steps=_diagnostic_value(n_acc),
        diagnostics={
            "history": "full",
            "grid": "adaptive",
            "estimator": solver.error_estimator,
            "estimator_stability": "experimental" if subdivision else "provisional",
            "error_norm": "max" if subdivision else "rms",
            "trial_substeps": 4 if subdivision else 2,
            "rtol": float(solver.rtol),
            "atol": float(solver.atol),
            "max_steps": max_steps,
            "n_nodes": _diagnostic_value(n_acc + 1),
            "n_accepted": _diagnostic_value(n_acc),
            "n_rejected": _diagnostic_value(n_rej),
            "n_attempted": _diagnostic_value(n_acc + n_rej),
            "reached_t_end": _diagnostic_value(done),
            "success": _diagnostic_value(done),
            "max_steps_reached": _diagnostic_value(exhausted),
            "nonfinite_detected": _diagnostic_value(nonfinite),
            "step_size_underflow": _diagnostic_value(underflow),
            "valid_mask": _diagnostic_value(valid),
            "last_time": _diagnostic_value(final_t),
            "requested_t_end": _diagnostic_value(requested_t_end),
            "time_bounds_representable": _diagnostic_value(bounds_exact),
            "minimum_step": _diagnostic_value(h_min),
        },
        warnings=warnings,
    )
    return SimulationResult(ts=node_times, latent_state=y_hist, solver_info=info)


def _maybe_concrete_int(value: Any) -> int | None:
    """``int(value)`` when concrete, else ``None`` (e.g. a tracer under ``jit``)."""

    try:
        return int(value)
    except (TypeError, ValueError, _jax().errors.ConcretizationTypeError):
        return None


def _maybe_concrete_bool(value: Any) -> bool | None:
    try:
        return bool(value)
    except (TypeError, ValueError, _jax().errors.ConcretizationTypeError):
        return None


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
    if not isfinite(float(dt)) or not float(dt) > 0.0:
        msg = f"Expected finite positive uniform timestep dt, got {dt}."
        raise ValueError(msg)


def _validate_increasing_grid(times: Any) -> None:
    """Validate a (possibly non-uniform) time grid: 1-D, strictly increasing.

    Static shape checks always run; the strictly-increasing / finite checks run on
    a concrete ``numpy`` view and are skipped under tracing, mirroring
    :func:`_validate_time_grid`.
    """

    if times.ndim != 1:
        msg = "Expected a one-dimensional time grid."
        raise ValueError(msg)
    if times.shape[0] < 1:
        msg = "Expected at least one time sample."
        raise ValueError(msg)

    try:
        import numpy as np

        concrete = np.asarray(times)
    except (TypeError, ValueError):
        return
    if not np.all(np.isfinite(concrete)):
        msg = "Expected finite time nodes."
        raise ValueError(msg)
    if not np.all(concrete[1:] > concrete[:-1]):
        msg = "Expected a strictly increasing time grid."
        raise ValueError(msg)


def _validate_time_grid(times: Any, dt: float) -> None:
    _validate_increasing_grid(times)
    if times.shape[0] == 1:
        return

    try:
        import numpy as np

        concrete_times = np.asarray(times)
    except (TypeError, ValueError):
        return

    # Tolerate floating-point grid-construction error (a few ULPs of the largest
    # time), scaled to the working dtype, but still reject genuinely non-uniform
    # grids. A fixed atol=1e-8 is below float32 grid resolution and wrongly
    # rejects large uniform grids built in the default single precision.
    dtype = concrete_times.dtype
    eps = float(np.finfo(dtype).eps) if np.issubdtype(dtype, np.floating) else 0.0
    # No unit-scale or absolute floor: on tiny grids either would admit large
    # relative spacing errors. Include both ends for grids starting below zero.
    scale = max(
        abs(float(concrete_times[0])), abs(float(concrete_times[-1])), abs(float(dt))
    )
    atol = 16.0 * eps * scale

    if np.issubdtype(dtype, np.integer):
        # Python integers prevent subtraction overflow without rounding the
        # original nodes to floats before their (potentially tiny) differences.
        diffs = np.diff(concrete_times.astype(object)).astype(float)
    else:
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
