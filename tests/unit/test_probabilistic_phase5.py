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
