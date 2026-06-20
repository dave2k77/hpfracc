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

    Unlike the fixed-grid solvers, this estimates the local error as it integrates
    and grows/shrinks the step to track ``rtol``/``atol``, clustering nodes where
    the solution is hard to resolve. The local error is a **step-doubling
    (Richardson)** estimate -- one step of ``h`` versus two of ``h/2`` -- which is
    the correct *local* estimator for a full-history fractional method (the
    predictor/corrector difference, by contrast, behaves like the *global* error
    because every step re-sums the whole history, so it cannot be used here).

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

    def __post_init__(self) -> None:
        as_order(self.order)
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


def _nonuniform_pece_weights(
    step: PyTree, times: PyTree, indices: PyTree, alpha: Any, dtype: Any
) -> tuple[PyTree, PyTree, PyTree]:
    """Variable-step fractional Adams predictor/corrector weights at node ``step``.

    Returns ``(predictor_weights, corrector_weights, diagonal_weight)``, each
    already scaled by ``1 / Gamma(alpha)``. ``predictor_weights`` and
    ``corrector_weights`` are length-``n_time`` history weights (zero for indices
    ``>= step``); ``diagonal_weight`` is the implicit-stage coefficient applied to
    ``f(t_n, y_predicted)``. On a uniform grid these collapse to the fixed-step
    Diethelm weights. Derived from product integration of the Volterra form with a
    piecewise-constant predictor and piecewise-linear corrector; the singular
    ``(t-s)**(alpha-1)`` integrates to ``**alpha`` / ``**(alpha+1)`` powers, and the
    most-recent interval makes the upper base zero, so ``_safe_pow`` keeps the
    alpha-gradient finite.
    """

    jnp = _jnp()
    t_n = times[step]
    t_next = jnp.concatenate([times[1:], times[-1:]])  # t_{j+1}; last is masked
    a_upper = (t_n - times).astype(dtype)  # A_j = t_n - t_j
    b_upper = (t_n - t_next).astype(dtype)  # B_j = t_n - t_{j+1}
    spacing = (t_next - times).astype(dtype)  # h_j (last entry bogus, masked)
    spacing = jnp.where(spacing > 0, spacing, 1.0)

    valid = indices < step  # interval j is used iff j < step
    a1 = alpha
    ap1 = alpha + 1.0
    pow_a_upper = _safe_pow(a_upper, a1)
    pow_b_upper = _safe_pow(b_upper, a1)
    pow_a_ap1 = _safe_pow(a_upper, ap1)
    pow_b_ap1 = _safe_pow(b_upper, ap1)
    diff_a = pow_a_upper - pow_b_upper  # A_j**a - B_j**a
    diff_ap1 = pow_a_ap1 - pow_b_ap1  # A_j**(a+1) - B_j**(a+1)

    predictor = jnp.where(valid, diff_a / a1, 0.0)
    # Per-interval hat integrals for the piecewise-linear corrector.
    i0 = jnp.where(valid, (diff_ap1 / ap1 - b_upper * diff_a / a1) / spacing, 0.0)
    i1 = jnp.where(valid, (a_upper * diff_a / a1 - diff_ap1 / ap1) / spacing, 0.0)

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
    alpha = as_order(solver.order)
    t0 = times_in[0]
    t_end = times_in[-1]
    span = t_end - t0

    max_steps = int(solver.max_steps)
    # Two extra buffer slots so the step-doubling "small" step can place its mid
    # and trial nodes at indices n_accepted+1, n_accepted+2 without overflowing.
    buf_len = max_steps + 2
    ctrl_dtype = y0.dtype if jnp.issubdtype(y0.dtype, jnp.floating) else jnp.float32
    rtol = jnp.asarray(solver.rtol, dtype=ctrl_dtype)
    atol = jnp.asarray(solver.atol, dtype=rtol.dtype)
    safety = jnp.asarray(solver.safety, dtype=rtol.dtype)
    min_factor = jnp.asarray(solver.min_factor, dtype=rtol.dtype)
    max_factor = jnp.asarray(solver.max_factor, dtype=rtol.dtype)
    # Local order of the corrector pair (~ 1 + alpha for 0 < alpha < 1).
    order_q = 1.0 + alpha
    richardson = 2.0**order_q - 1.0

    if solver.first_step is None:
        h0 = jnp.asarray(0.01, dtype=rtol.dtype) * span
    else:
        h0 = jnp.asarray(solver.first_step, dtype=rtol.dtype)
    h0 = jnp.minimum(h0, span)
    h_min = jnp.asarray(1e-12, dtype=rtol.dtype) * jnp.maximum(jnp.abs(t_end), 1.0)
    end_eps = jnp.asarray(1e-12, dtype=rtol.dtype) * jnp.maximum(jnp.abs(t_end), 1.0)

    f0 = model(
        t0, y0, params, rng_key=rng_key, inputs=_inputs_at(inputs, 0, 1)
    )
    dtype = f0.dtype
    node_times0 = jnp.zeros((buf_len,), dtype=times_in.dtype).at[0].set(t0)
    y_hist0 = jnp.zeros((buf_len, *y0.shape), dtype=dtype).at[0].set(y0)
    f_hist0 = jnp.zeros((buf_len, *y0.shape), dtype=dtype).at[0].set(f0)
    indices = jnp.arange(buf_len)

    def weighted_error(y_big: PyTree, y_small: PyTree) -> Any:
        err = jnp.abs(y_big - y_small) / richardson
        scale = atol + rtol * jnp.maximum(jnp.abs(y_big), jnp.abs(y_small))
        return jnp.sqrt(jnp.mean((err / scale) ** 2))

    def iteration(carry: Any, _: Any) -> tuple[Any, Any]:
        (n_acc, t_cur, h, node_times, y_hist, f_hist, n_rej, done) = carry
        sg = jax.lax.stop_gradient
        # Mesh / controller scalars are frozen for autodiff (diffrax pattern).
        t_cur_f = sg(t_cur)
        h_f = sg(h)
        t_trial = jnp.minimum(t_cur_f + h_f, t_end)
        h_act = t_trial - t_cur_f
        t_mid = t_cur_f + 0.5 * h_act
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

        err = sg(weighted_error(y_big, y_small))
        accept = ((err <= 1.0) | (h_f <= h_min)) & (~done)
        reached = t_trial >= t_end - end_eps

        # No local extrapolation: advance with the single full step ``y_big`` (the
        # solution whose error we estimated and are controlling). The half-step
        # ``y_small`` is used only for the error estimate. This makes the realized
        # trajectory identical to the non-uniform solver run on the realized mesh,
        # so gradients are the exact frozen-mesh sensitivity.
        idx = m + 1
        new_f = model(
            t_trial, y_big, params, rng_key=rng_key, inputs=_inputs_at(inputs, 0, 1)
        )
        node_times = node_times.at[idx].set(
            jnp.where(accept, t_trial, node_times[idx])
        )
        y_hist = y_hist.at[idx].set(jnp.where(accept, y_big, y_hist[idx]))
        f_hist = f_hist.at[idx].set(jnp.where(accept, new_f, f_hist[idx]))
        n_acc = n_acc + jnp.where(accept, 1, 0)
        n_rej = n_rej + jnp.where(accept | done, 0, 1)
        t_cur = jnp.where(accept, t_trial, t_cur)
        done = done | (accept & reached)

        # Integral step-size controller; frozen w.r.t. autodiff. ``factor`` grows
        # the step on accept and shrinks it on reject; clamped both sides.
        safe_err = jnp.maximum(err, 1e-16)
        factor = jnp.clip(
            safety * safe_err ** (-1.0 / (order_q + 1.0)), min_factor, max_factor
        )
        h_new = jnp.minimum(h_f * factor, jnp.maximum(t_end - t_cur, h_min))
        h = jnp.where(done, h, h_new)

        carry = (n_acc, t_cur, h, node_times, y_hist, f_hist, n_rej, done)
        return carry, None

    init = (
        jnp.asarray(0, dtype=jnp.int32),
        t0,
        h0,
        node_times0,
        y_hist0,
        f_hist0,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
    )
    (n_acc, t_cur, _h, node_times, y_hist, f_hist, n_rej, done), _ = jax.lax.scan(
        iteration, init, None, length=max_steps
    )

    # Trim to the realized mesh: pad the unused tail with the final node / state so
    # ``ts`` stays monotone non-decreasing and ``latent_state`` is well-defined.
    valid = indices <= n_acc
    final_t = node_times[n_acc]
    final_y = y_hist[n_acc]
    node_times = jnp.where(valid, node_times, final_t)
    y_hist = jnp.where(valid.reshape((-1,) + (1,) * y0.ndim), y_hist, final_y)

    # Concrete diagnostics when called eagerly; ``None`` under tracing (jit), where
    # these counts are data-dependent traced values and cannot be Python ints.
    n_acc_i = _maybe_concrete_int(n_acc)
    info = SolverInfo(
        name="adaptive_predictor_corrector",
        method="caputo_adaptive_pece",
        fractional_order=alpha,
        step_size=None,
        n_steps=n_acc_i,
        diagnostics={
            "history": "full",
            "grid": "adaptive",
            "estimator": "step_doubling",
            "rtol": float(solver.rtol),
            "atol": float(solver.atol),
            "max_steps": max_steps,
            "n_nodes": (n_acc_i + 1) if n_acc_i is not None else None,
            "n_rejected": _maybe_concrete_int(n_rej),
            "reached_t_end": _maybe_concrete_bool(done),
        },
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
    if not float(dt) > 0.0:
        msg = f"Expected positive uniform timestep dt, got {dt}."
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
    if times.shape[0] == 1:
        return

    try:
        import numpy as np

        concrete = np.asarray(times)
    except Exception:
        return
    if not np.all(np.isfinite(concrete)):
        msg = "Expected finite time nodes."
        raise ValueError(msg)
    if not np.all(np.diff(concrete) > 0.0):
        msg = "Expected a strictly increasing time grid."
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
