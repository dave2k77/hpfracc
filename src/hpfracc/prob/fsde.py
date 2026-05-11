"""Experimental stochastic fractional differential equation helpers."""

from __future__ import annotations

from math import gamma
from typing import Any

from hpfracc.ops import validate_order
from hpfracc.solvers import PredictorCorrector, SimulationResult, SolverInfo
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
    """Simulate an experimental additive-noise Caputo FSDE baseline.

    This is a small Euler-Maruyama-style stochastic extension of the full-history
    Caputo predictor-corrector solver. It is intended for reproducible research
    prototypes, not as a validated stochastic numerics claim.
    """

    jax = _jax()
    jnp = _jnp()
    times = jnp.asarray(ts)
    _validate_time_grid(times, solver.dt)

    y0 = jnp.asarray(initial_state)
    alpha = validate_order(solver.order)
    dt = float(solver.dt)
    dt_alpha = dt**alpha
    predictor_scale = dt_alpha / gamma(alpha + 1.0)
    corrector_scale = dt_alpha / gamma(alpha + 2.0)
    noise_scale = dt**0.5

    key = rng_key
    trajectory = [y0]
    f_history = [
        model(
            times[0],
            y0,
            params,
            rng_key=key,
            inputs=_inputs_at(inputs, 0, times.shape[0]),
        )
    ]

    for step in range(1, int(times.shape[0])):
        predictor_history = jnp.zeros_like(y0)
        for history_index, f_value in enumerate(f_history):
            lag = step - history_index
            weight = (lag**alpha) - ((lag - 1) ** alpha)
            predictor_history = predictor_history + weight * f_value
        predicted = y0 + predictor_scale * predictor_history

        predicted_f = model(
            times[step],
            predicted,
            params,
            rng_key=key,
            inputs=_inputs_at(inputs, step, times.shape[0]),
        )

        corrector_history = jnp.zeros_like(y0)
        for history_index, f_value in enumerate(f_history):
            weight = _corrector_history_weight(alpha, step, history_index)
            corrector_history = corrector_history + weight * f_value

        deterministic = y0 + corrector_scale * (corrector_history + predicted_f)
        key, noise_key = jax.random.split(key)
        sigma = diffusion(
            times[step],
            deterministic,
            params,
            rng_key=noise_key,
            inputs=_inputs_at(inputs, step, times.shape[0]),
        )
        noise = jax.random.normal(noise_key, shape=deterministic.shape)
        corrected = deterministic + jnp.asarray(sigma) * noise_scale * noise
        trajectory.append(corrected)
        f_history.append(
            model(
                times[step],
                corrected,
                params,
                rng_key=key,
                inputs=_inputs_at(inputs, step, times.shape[0]),
            )
        )

    values = jnp.stack(trajectory, axis=0)
    info = SolverInfo(
        name="stochastic_predictor_corrector",
        method="caputo_pece_additive_noise",
        fractional_order=alpha,
        step_size=dt,
        n_steps=int(times.shape[0] - 1),
        diagnostics={"history": "full", "grid": "uniform", "noise": "additive"},
        warnings=(
            "Experimental stochastic baseline; not a validated stochastic "
            "fractional numerical method.",
        ),
    )
    return SimulationResult(ts=times, latent_state=values, solver_info=info)


def _corrector_history_weight(alpha: float, step: int, history_index: int) -> float:
    if history_index == 0:
        return (step - 1) ** (alpha + 1.0) - (
            step - 1.0 - alpha
        ) * step**alpha

    lag = step - history_index
    return (
        (lag + 1) ** (alpha + 1.0)
        + (lag - 1) ** (alpha + 1.0)
        - 2.0 * lag ** (alpha + 1.0)
    )


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


def _validate_time_grid(times: Any, dt: float) -> None:
    if times.ndim != 1:
        msg = "Expected a one-dimensional time grid."
        raise ValueError(msg)
    if times.shape[0] < 1:
        msg = "Expected at least one time sample."
        raise ValueError(msg)
    if times.shape[0] == 1:
        return

    import numpy as np

    concrete_times = np.asarray(times)
    diffs = np.diff(concrete_times)
    if not np.allclose(diffs, diffs[0], rtol=1e-5, atol=1e-8):
        msg = "Expected a uniform time grid."
        raise ValueError(msg)
    if not np.isclose(float(diffs[0]), float(dt), rtol=1e-5, atol=1e-8):
        msg = (
            "Expected solver dt to match time-grid spacing; "
            f"got dt={dt} and spacing={float(diffs[0])}."
        )
        raise ValueError(msg)


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
