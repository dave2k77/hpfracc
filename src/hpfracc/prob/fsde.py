"""Experimental stochastic fractional differential equation helpers."""

from __future__ import annotations

from math import gamma
from typing import Any

from hpfracc.ops import validate_order
from hpfracc.solvers import PredictorCorrector, SimulationResult, SolverInfo
from hpfracc.solvers.predictor_corrector import _inputs_at, _validate_time_grid
from hpfracc.typing import DynamicsFn, PRNGKey, PyTree


def simulate_stochastic(
    *,
    model: DynamicsFn,
    diffusion: DynamicsFn,
    ts: PyTree,
    solver: PredictorCorrector,
    initial_state: PyTree,
    params: PyTree,
    rng_key: PRNGKey,
    inputs: PyTree | None = None,
) -> SimulationResult:
    """Simulate an additive-noise Caputo FSDE with a fractional Euler-Maruyama step.

    Solves ``D_C^alpha y(t) = f(t, y) + g(t, y) xi(t)`` for ``1/2 < alpha < 1``,
    where ``xi`` is Gaussian white noise. In integral form the stochastic term is
    the Riemann-Liouville fractional integral of the noise,

    ``(1 / Gamma(alpha)) * integral_0^t (t - s)^(alpha - 1) g(s, y) dW(s)``,

    which is discretised with the exact per-interval kernel variance

    ``w_{j,n} = integral_{t_j}^{t_{j+1}} (t_n - s)^(2 alpha - 2) ds``

    so each Brownian increment is weighted by ``sqrt(w_{j,n})``. The drift uses
    the same full-history Diethelm predictor-corrector recurrence as the
    deterministic solver, and the recurrence is evaluated with ``jax.lax.scan``.

    The scheme is validated against the analytic variance of the linear additive
    FSDE (a Monte-Carlo check), and its sample mean reproduces the deterministic
    Mittag-Leffler solution. It is restricted to ``alpha > 1/2`` because the
    stochastic fractional integral of white noise is singular for
    ``alpha <= 1/2``.
    """

    jax = _jax()
    jnp = _jnp()
    times = jnp.asarray(ts)
    _validate_time_grid(times, solver.dt)

    y0 = jnp.asarray(initial_state)
    alpha = validate_order(solver.order)
    if not alpha > 0.5:
        msg = (
            "Additive-noise Caputo FSDE requires order > 0.5: the stochastic "
            "fractional integral of white noise is singular for order <= 0.5. "
            f"Got order={solver.order}."
        )
        raise ValueError(msg)

    n_time = int(times.shape[0])
    h = float(solver.dt)
    predictor_scale = h**alpha / gamma(alpha + 1.0)
    corrector_scale = h**alpha / gamma(alpha + 2.0)
    inv_gamma_alpha = 1.0 / gamma(alpha)
    two_alpha_minus_one = 2.0 * alpha - 1.0

    indices = jnp.arange(n_time)
    # One standard-normal Brownian increment per interval, drawn once and shared
    # across all steps so the simulated path is consistent (and reproducible for
    # a fixed key).
    increments = jax.random.normal(rng_key, (n_time, *y0.shape))

    inputs_0 = _inputs_at(inputs, 0, n_time)
    f0 = model(times[0], y0, params, rng_key=None, inputs=inputs_0)
    g0 = jnp.asarray(diffusion(times[0], y0, params, rng_key=None, inputs=inputs_0))
    drift_history0 = jnp.zeros((n_time, *y0.shape), dtype=f0.dtype).at[0].set(f0)
    diffusion_history0 = jnp.zeros((n_time, *y0.shape), dtype=g0.dtype).at[0].set(g0)

    def history_dot(weights: PyTree, history: PyTree) -> PyTree:
        return jnp.tensordot(weights, history, axes=([0], [0]))

    def state_broadcast(vector: PyTree) -> PyTree:
        return vector.reshape((n_time, *([1] * (y0.ndim))))

    def step_fn(carry: PyTree, step: PyTree) -> tuple[PyTree, PyTree]:
        drift_history, diffusion_history = carry
        valid = indices < step
        lag = jnp.where(valid, (step - indices).astype(f0.dtype), 1.0)
        step_f = step.astype(f0.dtype)

        predictor_weights = jnp.where(valid, lag**alpha - (lag - 1.0) ** alpha, 0.0)
        predicted = y0 + predictor_scale * history_dot(predictor_weights, drift_history)
        predicted_f = model(
            times[step], predicted, params, rng_key=None,
            inputs=_inputs_at(inputs, step, n_time),
        )

        boundary = (step_f - 1.0) ** (alpha + 1.0) - (
            step_f - 1.0 - alpha
        ) * step_f**alpha
        interior = (
            (lag + 1.0) ** (alpha + 1.0)
            + (lag - 1.0) ** (alpha + 1.0)
            - 2.0 * lag ** (alpha + 1.0)
        )
        corrector_weights = jnp.where(
            valid, jnp.where(indices == 0, boundary, interior), 0.0
        )
        drift = y0 + corrector_scale * (
            history_dot(corrector_weights, drift_history) + predicted_f
        )

        # Exact per-interval kernel variance w_{j,n}, then sqrt for the increment
        # weight. The lag is clamped to >= 1 on masked entries before the power.
        interval_variance = (
            h**two_alpha_minus_one
            * (lag**two_alpha_minus_one - (lag - 1.0) ** two_alpha_minus_one)
            / two_alpha_minus_one
        )
        increment_weights = jnp.where(valid, jnp.sqrt(interval_variance), 0.0)
        stochastic = inv_gamma_alpha * jnp.sum(
            state_broadcast(increment_weights) * diffusion_history * increments,
            axis=0,
        )
        corrected = drift + stochastic

        next_f = model(
            times[step], corrected, params, rng_key=None,
            inputs=_inputs_at(inputs, step, n_time),
        )
        next_g = jnp.asarray(
            diffusion(
                times[step], corrected, params, rng_key=None,
                inputs=_inputs_at(inputs, step, n_time),
            )
        )
        carry = (
            drift_history.at[step].set(next_f),
            diffusion_history.at[step].set(next_g),
        )
        return carry, corrected

    _, tail = jax.lax.scan(
        step_fn, (drift_history0, diffusion_history0), jnp.arange(1, n_time)
    )
    values = jnp.concatenate([y0[jnp.newaxis, ...], tail], axis=0)

    info = SolverInfo(
        name="stochastic_predictor_corrector",
        method="caputo_fractional_euler_maruyama",
        fractional_order=alpha,
        step_size=h,
        n_steps=int(n_time - 1),
        diagnostics={"history": "full", "grid": "uniform", "noise": "additive"},
        warnings=(
            "Experimental additive-noise FSDE for order > 0.5; validated against "
            "the analytic linear-FSDE variance but not a general stochastic "
            "numerics guarantee.",
        ),
    )
    return SimulationResult(ts=times, latent_state=values, solver_info=info)


def _jax() -> Any:
    try:
        import jax
    except ModuleNotFoundError as exc:
        msg = (
            "JAX is required for hpfracc.prob stochastic simulations. Install "
            "the package with its runtime dependencies before calling this "
            "function."
        )
        raise ModuleNotFoundError(msg) from exc
    return jax


def _jnp() -> Any:
    try:
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        msg = (
            "JAX is required for hpfracc.prob stochastic simulations. Install "
            "the package with its runtime dependencies before calling this "
            "function."
        )
        raise ModuleNotFoundError(msg) from exc
    return jnp
