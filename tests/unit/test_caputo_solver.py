from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import hpfracc as hp


def linear_model(
    t: jax.Array,
    state: jax.Array,
    params: jax.Array,
    *,
    rng_key: jax.Array | None = None,
    inputs: jax.Array | None = None,
) -> jax.Array:
    del t, rng_key, inputs
    return params * state


def mittag_leffler(alpha: float, z: jax.Array, *, terms: int = 80) -> jax.Array:
    total = jnp.zeros_like(z)
    for k in range(terms):
        total = total + (z**k) / math.gamma(alpha * k + 1.0)
    return total


def solve_linear(*, n_steps: int, order: float, rate: float) -> hp.solvers.SimulationResult:
    ts = jnp.linspace(0.0, 1.0, n_steps)
    solver = hp.solvers.PredictorCorrector(
        dt=float(ts[1] - ts[0]),
        order=order,
    )
    return hp.solvers.simulate(
        model=linear_model,
        ts=ts,
        solver=solver,
        initial_state=jnp.asarray(1.0),
        params=jnp.asarray(rate),
    )


def test_predictor_corrector_rejects_nonuniform_time_grid() -> None:
    solver = hp.solvers.PredictorCorrector(dt=0.1, order=0.7)
    ts = jnp.asarray([0.0, 0.1, 0.25])

    with pytest.raises(ValueError, match="uniform time grid"):
        hp.solvers.simulate(
            model=linear_model,
            ts=ts,
            solver=solver,
            initial_state=jnp.asarray(1.0),
            params=jnp.asarray(-1.0),
        )


def test_predictor_corrector_rejects_mismatched_dt() -> None:
    solver = hp.solvers.PredictorCorrector(dt=0.2, order=0.7)
    ts = jnp.asarray([0.0, 0.1, 0.2])

    with pytest.raises(ValueError, match="dt to match"):
        hp.solvers.simulate(
            model=linear_model,
            ts=ts,
            solver=solver,
            initial_state=jnp.asarray(1.0),
            params=jnp.asarray(-1.0),
        )


def test_predictor_corrector_preserves_trailing_state_shape() -> None:
    solver = hp.solvers.PredictorCorrector(dt=0.1, order=0.6)
    ts = jnp.arange(6) * solver.dt
    y0 = jnp.ones((2, 3))

    result = hp.solvers.simulate(
        model=linear_model,
        ts=ts,
        solver=solver,
        initial_state=y0,
        params=jnp.asarray(-0.5),
    )

    assert result.latent_state.shape == (6, 2, 3)
    assert result.solver_info is not None
    assert result.solver_info.method == "caputo_pece_full_history"


def test_predictor_corrector_refines_against_linear_reference() -> None:
    order = 0.7
    rate = -0.8
    coarse = solve_linear(n_steps=21, order=order, rate=rate)
    fine = solve_linear(n_steps=81, order=order, rate=rate)

    coarse_expected = mittag_leffler(order, rate * coarse.ts**order)
    fine_expected = mittag_leffler(order, rate * fine.ts**order)
    coarse_error = jnp.max(jnp.abs(coarse.latent_state - coarse_expected))
    fine_error = jnp.max(jnp.abs(fine.latent_state - fine_expected))

    assert fine_error < coarse_error


def test_predictor_corrector_is_jit_consistent_for_initial_state() -> None:
    order = 0.6
    ts = jnp.linspace(0.0, 0.5, 11)
    solver = hp.solvers.PredictorCorrector(dt=float(ts[1] - ts[0]), order=order)

    def run(initial_state: jax.Array) -> jax.Array:
        return hp.solvers.simulate(
            model=linear_model,
            ts=ts,
            solver=solver,
            initial_state=initial_state,
            params=jnp.asarray(-0.25),
        ).latent_state

    actual = jax.jit(run)(jnp.asarray(1.0))
    expected = run(jnp.asarray(1.0))
    assert jnp.allclose(actual, expected)


def test_predictor_corrector_supports_initial_state_and_parameter_gradients() -> None:
    ts = jnp.linspace(0.0, 0.5, 11)
    solver = hp.solvers.PredictorCorrector(dt=float(ts[1] - ts[0]), order=0.6)

    def objective(initial_state: jax.Array, rate: jax.Array) -> jax.Array:
        result = hp.solvers.simulate(
            model=linear_model,
            ts=ts,
            solver=solver,
            initial_state=initial_state,
            params=rate,
        )
        return result.latent_state[-1]

    grad_initial, grad_rate = jax.grad(objective, argnums=(0, 1))(
        jnp.asarray(1.0),
        jnp.asarray(-0.25),
    )

    assert jnp.isfinite(grad_initial)
    assert jnp.isfinite(grad_rate)

