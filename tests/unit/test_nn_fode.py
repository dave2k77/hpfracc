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


def make_model() -> hp.nn.NeuralFODE:
    solver = hp.solvers.PredictorCorrector(dt=0.05, order=0.7)
    return hp.nn.NeuralFODE(dynamics=linear_dynamics, solver=solver)


def test_neural_fode_returns_simulation_result() -> None:
    model = make_model()
    ts = jnp.linspace(0.0, 0.5, 11)

    result = model(
        ts=ts,
        initial_state=jnp.asarray(1.0),
        params={"rate": jnp.asarray(-0.5)},
    )

    assert result.latent_state.shape == (11,)
    assert result.observed is None
    assert result.solver_info is not None
    assert result.solver_info.name == "predictor_corrector"


def test_neural_fode_observation_transform_is_recorded() -> None:
    solver = hp.solvers.PredictorCorrector(dt=0.05, order=0.7)
    model = hp.nn.NeuralFODE(
        dynamics=linear_dynamics,
        solver=solver,
        observe=lambda latent: latent[..., None],
    )
    ts = jnp.linspace(0.0, 0.5, 11)

    result = model(
        ts=ts,
        initial_state=jnp.asarray(1.0),
        params={"rate": jnp.asarray(-0.5)},
    )

    assert result.latent_state.shape == (11,)
    assert result.observed.shape == (11, 1)


def test_trajectory_mse_supports_parameter_and_initial_state_gradients() -> None:
    model = make_model()
    ts = jnp.linspace(0.0, 0.5, 11)
    target = model(
        ts=ts,
        initial_state=jnp.asarray(1.2),
        params={"rate": jnp.asarray(-0.7)},
    ).latent_state

    def objective(initial_state: jax.Array, rate: jax.Array) -> jax.Array:
        return hp.nn.trajectory_mse(
            model,
            ts=ts,
            initial_state=initial_state,
            params={"rate": rate},
            target=target,
        )

    grad_initial, grad_rate = jax.grad(objective, argnums=(0, 1))(
        jnp.asarray(1.0),
        jnp.asarray(-0.4),
    )

    assert jnp.isfinite(grad_initial)
    assert jnp.isfinite(grad_rate)


def test_sgd_step_updates_parameter_pytrees() -> None:
    params = {"rate": jnp.asarray(-0.2), "bias": jnp.asarray([1.0, -1.0])}
    grads = {"rate": jnp.asarray(0.5), "bias": jnp.asarray([0.25, -0.25])}

    actual = hp.nn.sgd_step(params, grads, learning_rate=0.1)

    assert jnp.allclose(actual["rate"], -0.25)
    assert jnp.allclose(actual["bias"], jnp.asarray([0.975, -0.975]))


def test_synthetic_rate_recovery_reduces_loss() -> None:
    model = make_model()
    ts = jnp.linspace(0.0, 0.5, 11)
    target = model(
        ts=ts,
        initial_state=jnp.asarray(1.0),
        params={"rate": jnp.asarray(-0.8)},
    ).latent_state

    def loss(params: dict[str, jax.Array]) -> jax.Array:
        return hp.nn.trajectory_mse(
            model,
            ts=ts,
            initial_state=jnp.asarray(1.0),
            params=params,
            target=target,
        )

    params = {"rate": jnp.asarray(0.1)}
    initial_loss = loss(params)
    for _ in range(20):
        grads = jax.grad(loss)(params)
        params = hp.nn.sgd_step(params, grads, learning_rate=0.5)

    assert loss(params) < initial_loss
    assert params["rate"] < 0.1
