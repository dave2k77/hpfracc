from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import hpfracc as hp


def linear_dynamics(
    t: jax.Array,
    state: jax.Array,
    params: dict[str, jax.Array],
    *,
    rng_key: jax.Array | None = None,
    inputs: jax.Array | None = None,
) -> jax.Array:
    del t, rng_key, inputs
    return params["rate"] * state


def constant_diffusion(
    t: jax.Array,
    state: jax.Array,
    params: dict[str, jax.Array],
    *,
    rng_key: jax.Array | None = None,
    inputs: jax.Array | None = None,
) -> jax.Array:
    del t, state, rng_key, inputs
    return jnp.asarray(params["noise"])


def make_model() -> hp.nn.NeuralFODE:
    ts = jnp.linspace(0.0, 0.5, 11)
    solver = hp.solvers.PredictorCorrector(dt=float(ts[1] - ts[0]), order=0.7)
    return hp.nn.NeuralFODE(dynamics=linear_dynamics, solver=solver)


def test_stochastic_simulation_is_reproducible_for_same_key() -> None:
    ts = jnp.linspace(0.0, 0.5, 11)
    solver = hp.solvers.PredictorCorrector(dt=float(ts[1] - ts[0]), order=0.7)
    params = {"rate": jnp.asarray(-0.5), "noise": jnp.asarray(0.1)}
    key = jax.random.PRNGKey(0)

    first = hp.prob.simulate_stochastic(
        model=linear_dynamics,
        diffusion=constant_diffusion,
        ts=ts,
        solver=solver,
        initial_state=jnp.asarray(1.0),
        params=params,
        rng_key=key,
    )
    second = hp.prob.simulate_stochastic(
        model=linear_dynamics,
        diffusion=constant_diffusion,
        ts=ts,
        solver=solver,
        initial_state=jnp.asarray(1.0),
        params=params,
        rng_key=key,
    )

    assert jnp.allclose(first.latent_state, second.latent_state)
    assert first.solver_info is not None
    assert first.solver_info.warnings


def test_stochastic_simulation_changes_with_key() -> None:
    ts = jnp.linspace(0.0, 0.5, 11)
    solver = hp.solvers.PredictorCorrector(dt=float(ts[1] - ts[0]), order=0.7)
    params = {"rate": jnp.asarray(-0.5), "noise": jnp.asarray(0.1)}

    first = hp.prob.simulate_stochastic(
        model=linear_dynamics,
        diffusion=constant_diffusion,
        ts=ts,
        solver=solver,
        initial_state=jnp.asarray(1.0),
        params=params,
        rng_key=jax.random.PRNGKey(0),
    )
    second = hp.prob.simulate_stochastic(
        model=linear_dynamics,
        diffusion=constant_diffusion,
        ts=ts,
        solver=solver,
        initial_state=jnp.asarray(1.0),
        params=params,
        rng_key=jax.random.PRNGKey(1),
    )

    assert not jnp.allclose(first.latent_state, second.latent_state)


def test_gaussian_log_likelihood_prefers_closer_predictions() -> None:
    observed = jnp.asarray([1.0, 0.8, 0.6])
    close = jnp.asarray([1.0, 0.81, 0.59])
    far = jnp.asarray([1.3, 1.1, 0.9])

    assert hp.prob.gaussian_log_likelihood(
        close,
        observed,
        noise_scale=0.1,
    ) > hp.prob.gaussian_log_likelihood(far, observed, noise_scale=0.1)


def test_grid_calibration_recovers_nearby_scalar_parameter() -> None:
    model = make_model()
    ts = jnp.linspace(0.0, 0.5, 11)
    observed = model(
        ts=ts,
        initial_state=jnp.asarray(1.0),
        params={"rate": jnp.asarray(-0.8)},
    ).latent_state
    grid = jnp.asarray([-1.2, -0.8, -0.4, 0.0])

    result = hp.prob.grid_calibrate_scalar(
        model,
        ts=ts,
        initial_state=jnp.asarray(1.0),
        observations=observed,
        parameter_name="rate",
        parameter_grid=grid,
        noise_scale=0.05,
    )

    assert result.best_index == 1
    assert jnp.isclose(result.best_params["rate"], -0.8)
    assert jnp.isclose(jnp.sum(result.posterior_weights), 1.0)


def test_posterior_predictive_returns_weighted_summary() -> None:
    model = make_model()
    ts = jnp.linspace(0.0, 0.5, 11)
    grid = jnp.asarray([-1.2, -0.8, -0.4])
    weights = jnp.asarray([0.1, 0.8, 0.1])

    result = hp.prob.posterior_predictive(
        model,
        ts=ts,
        initial_state=jnp.asarray(1.0),
        parameter_name="rate",
        parameter_grid=grid,
        weights=weights,
    )

    assert result.trajectories.shape == (3, 11)
    assert result.mean.shape == (11,)
    assert result.lower.shape == (11,)
    assert result.upper.shape == (11,)
    assert jnp.isclose(jnp.sum(result.weights), 1.0)


def test_weighted_quantile_matches_known_discrete_cases() -> None:
    two = jnp.asarray([0.0, 10.0])
    # The weighted median follows the mass: 0.9 on the low point -> low point.
    assert float(hp.prob.weighted_quantile(two, jnp.asarray([0.9, 0.1]), 0.5)) == 0.0
    assert float(hp.prob.weighted_quantile(two, jnp.asarray([0.1, 0.9]), 0.5)) == 10.0

    four = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    uniform = jnp.ones(4)
    assert float(hp.prob.weighted_quantile(four, uniform, 0.05)) == 0.0
    assert float(hp.prob.weighted_quantile(four, uniform, 0.5)) == 1.0
    assert float(hp.prob.weighted_quantile(four, uniform, 0.95)) == 3.0


def test_weighted_quantile_is_vectorized_over_columns() -> None:
    # Leading axis is the weighted (grid) axis; each column resolved independently.
    stacked = jnp.asarray([[0.0, 0.0], [1.0, 5.0], [2.0, 9.0]])
    lower = hp.prob.weighted_quantile(stacked, jnp.ones(3), 0.05)
    upper = hp.prob.weighted_quantile(stacked, jnp.ones(3), 0.95)
    assert jnp.allclose(lower, jnp.asarray([0.0, 0.0]))
    assert jnp.allclose(upper, jnp.asarray([2.0, 9.0]))


def test_weighted_quantile_rejects_out_of_range_q() -> None:
    with pytest.raises(ValueError, match="quantile level"):
        hp.prob.weighted_quantile(jnp.asarray([0.0, 1.0]), jnp.ones(2), 1.5)


def _final_band(weights: jax.Array) -> tuple[float, float, float]:
    model = make_model()
    ts = jnp.linspace(0.0, 0.5, 11)
    grid = jnp.asarray([-1.2, -0.8, -0.4])
    result = hp.prob.posterior_predictive(
        model,
        ts=ts,
        initial_state=jnp.asarray(1.0),
        parameter_name="rate",
        parameter_grid=grid,
        weights=weights,
    )
    return float(result.lower[-1]), float(result.mean[-1]), float(result.upper[-1])


def test_posterior_predictive_interval_respects_weights() -> None:
    # Regression guard for the original bug, where lower/upper used an unweighted
    # quantile and were identical across different posteriors. The grid is
    # [-1.2, -0.8, -0.4]; faster decay (-1.2) gives the lowest final value.
    low_lower, _, low_upper = _final_band(jnp.asarray([0.98, 0.01, 0.01]))
    high_lower, _, high_upper = _final_band(jnp.asarray([0.01, 0.01, 0.98]))

    # Different posteriors must yield different bands (the bug made them equal).
    assert not jnp.isclose(low_lower, high_lower)
    assert not jnp.isclose(low_upper, high_upper)
    # A posterior concentrated on fast decay sits entirely below one concentrated
    # on slow decay.
    assert low_upper < high_lower


def test_posterior_predictive_band_stays_within_trajectory_support() -> None:
    model = make_model()
    ts = jnp.linspace(0.0, 0.5, 11)
    grid = jnp.asarray([-1.2, -0.8, -0.4])
    weights = jnp.asarray([0.2, 0.5, 0.3])
    result = hp.prob.posterior_predictive(
        model,
        ts=ts,
        initial_state=jnp.asarray(1.0),
        parameter_name="rate",
        parameter_grid=grid,
        weights=weights,
    )
    per_time_min = jnp.min(result.trajectories, axis=0)
    per_time_max = jnp.max(result.trajectories, axis=0)
    assert jnp.all(result.lower >= per_time_min - 1e-6)
    assert jnp.all(result.upper <= per_time_max + 1e-6)
    assert jnp.all(result.upper >= result.lower)


def test_posterior_predictive_concentrated_weight_collapses_band() -> None:
    # 90% interval of a posterior with ~all mass on one grid point is that point.
    lower, _, upper = _final_band(jnp.asarray([0.001, 0.998, 0.001]))
    assert jnp.isclose(lower, upper, atol=1e-6)


def test_posterior_predictive_uniform_weight_brackets_mean() -> None:
    # For a symmetric uniform grid weighting the mean lies inside the band.
    lower, mean, upper = _final_band(jnp.asarray([1.0, 1.0, 1.0]))
    assert lower <= mean <= upper
