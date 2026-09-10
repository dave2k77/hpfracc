"""Regression coverage for eager solver validation and adaptive time dtypes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import hpfracc as hp

_SOLVERS = ("explicit", "implicit", "nonuniform", "adaptive", "stochastic")
_UNIFORM_SOLVERS = ("explicit", "implicit", "stochastic")


def _linear_model(t, state, params, *, rng_key=None, inputs=None):
    del t, rng_key, inputs
    return params * state


def _diffusion(t, state, params, *, rng_key=None, inputs=None):
    del t, params, rng_key, inputs
    return jnp.ones_like(state) * 0.01


def _simulate(kind, ts, *, dt=1e-7):
    if kind == "implicit":
        solver = hp.solvers.ImplicitPredictorCorrector(dt=dt, order=0.7)
    elif kind == "nonuniform":
        solver = hp.solvers.NonUniformPredictorCorrector(order=0.7)
    elif kind == "adaptive":
        solver = hp.solvers.AdaptivePredictorCorrector(order=0.7, max_steps=16)
    else:
        solver = hp.solvers.PredictorCorrector(dt=dt, order=0.7)
    kwargs = {
        "model": _linear_model,
        "ts": ts,
        "solver": solver,
        "initial_state": jnp.asarray(1.0),
        "params": jnp.asarray(-0.1),
    }
    if kind == "stochastic":
        return hp.prob.simulate_stochastic(
            **kwargs, diffusion=_diffusion, rng_key=jax.random.PRNGKey(0)
        )
    return hp.solvers.simulate(**kwargs)


@pytest.mark.parametrize("kind", _SOLVERS)
@pytest.mark.parametrize("nodes", ([0.0, 0.0, 0.0], [2e-7, 1e-7, 0.0]))
def test_solver_rejects_non_increasing_grid(kind, nodes):
    with pytest.raises(ValueError, match="strictly increasing"):
        _simulate(kind, jnp.asarray(nodes, dtype=jnp.float32))


@pytest.mark.parametrize("kind", _SOLVERS)
@pytest.mark.parametrize(
    "nodes",
    ([float("nan")], [float("inf")], [-float("inf")], [0.0, float("inf")]),
)
def test_solver_rejects_nonfinite_grid_including_single_node(kind, nodes):
    with pytest.raises(ValueError, match="finite time nodes"):
        _simulate(kind, jnp.asarray(nodes))


@pytest.mark.parametrize(
    "solver_type",
    (hp.solvers.PredictorCorrector, hp.solvers.ImplicitPredictorCorrector),
)
@pytest.mark.parametrize("dt", (0.0, -1.0, float("nan"), float("inf")))
def test_uniform_solver_requires_finite_positive_dt(solver_type, dt):
    with pytest.raises(ValueError, match="positive uniform timestep"):
        solver_type(dt=dt, order=0.7)


@pytest.mark.parametrize("kind", _UNIFORM_SOLVERS)
@pytest.mark.parametrize("dt", (1e-7, 1e-14))
def test_uniform_solver_rejects_small_mismatched_spacing(kind, dt):
    # The rounding tolerance must scale with the grid, not an absolute unit floor.
    ts = jnp.arange(3, dtype=jnp.float32) * (2.0 * dt)
    with pytest.raises(ValueError, match="dt to match"):
        _simulate(kind, ts, dt=dt)


@pytest.mark.parametrize("kind", _UNIFORM_SOLVERS)
def test_uniform_solver_rejects_small_nonuniform_spacing(kind):
    ts = jnp.asarray([0.0, 1e-7, 3e-7], dtype=jnp.float32)
    with pytest.raises(ValueError, match="uniform time grid"):
        _simulate(kind, ts)


@pytest.mark.parametrize("kind", _UNIFORM_SOLVERS)
@pytest.mark.parametrize("dt", (1e-7, 1e-14))
def test_uniform_solver_accepts_small_valid_spacing(kind, dt):
    ts = jnp.arange(3, dtype=jnp.float32) * dt
    result = _simulate(kind, ts, dt=dt)
    assert jnp.array_equal(result.ts, ts)
    assert jnp.all(jnp.isfinite(result.latent_state))


@pytest.mark.parametrize("time_dtype", (jnp.int32, jnp.float32, jnp.float64))
@pytest.mark.parametrize("state_dtype", (jnp.float32, jnp.float64))
@pytest.mark.filterwarnings(
    "error:scatter inputs have incompatible types:FutureWarning"
)
def test_adaptive_normalizes_time_dtype(enable_x64, time_dtype, state_dtype):
    solver = hp.solvers.AdaptivePredictorCorrector(
        order=0.7, rtol=1e-2, atol=1e-4, max_steps=32
    )
    times = jnp.asarray([0, 1], dtype=time_dtype)
    y0 = jnp.asarray(1.0, dtype=state_dtype)
    rate = jnp.asarray(-0.1, dtype=state_dtype)
    time_dtype_expected = jnp.result_type(times.dtype, y0.dtype, float)

    def run(ts, initial_state):
        return hp.solvers.simulate(
            model=_linear_model,
            ts=ts,
            solver=solver,
            initial_state=initial_state,
            params=rate,
        )

    result = run(times, y0)
    reference = run(times.astype(time_dtype_expected), y0)
    assert result.ts.dtype == time_dtype_expected
    assert result.solver_info.diagnostics["reached_t_end"] is True
    assert jnp.allclose(result.ts, reference.ts)
    assert jnp.allclose(result.latent_state, reference.latent_state)
    assert jnp.all(jnp.isfinite(result.latent_state))

    # Dynamic bounds must work under jit as well as on the eager path.
    compiled = jax.jit(lambda ts, state: run(ts, state).latent_state)(times, y0)
    assert jnp.allclose(compiled, result.latent_state, rtol=1e-5, atol=1e-6)


def test_adaptive_integer_bounds_preserve_state_gradient():
    solver = hp.solvers.AdaptivePredictorCorrector(
        order=0.7, rtol=1e-2, atol=1e-4, max_steps=32
    )

    def endpoint(y0, ts):
        return hp.solvers.simulate(
            model=_linear_model,
            ts=ts,
            solver=solver,
            initial_state=y0,
            params=jnp.asarray(-0.1),
        ).latent_state[-1]

    y0 = jnp.asarray(1.0)
    integer = jax.grad(endpoint)(y0, jnp.asarray([0, 1]))
    floating = jax.grad(endpoint)(y0, jnp.asarray([0.0, 1.0]))
    assert jnp.isfinite(integer)
    assert jnp.allclose(integer, floating)


def test_adaptive_controller_accepts_higher_precision_dynamics(enable_x64):
    solver = hp.solvers.AdaptivePredictorCorrector(
        order=0.7, rtol=1e-2, atol=1e-4, max_steps=32
    )
    times = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    y0 = jnp.asarray(1.0, dtype=jnp.float32)

    def run(rate):
        return hp.solvers.simulate(
            model=_linear_model,
            ts=times,
            solver=solver,
            initial_state=y0,
            params=rate,
        )

    # A float64 parameter promotes the dynamics/history, not the time carry.
    rate = jnp.asarray(-0.1, dtype=jnp.float64)
    result = run(rate)
    assert result.ts.dtype == jnp.float32
    assert result.latent_state.dtype == jnp.float64
    assert result.solver_info.diagnostics["reached_t_end"] is True
    assert jnp.all(jnp.isfinite(result.latent_state))
    compiled = jax.jit(lambda p: run(p).latent_state)(rate)
    assert jnp.allclose(compiled, result.latent_state)
