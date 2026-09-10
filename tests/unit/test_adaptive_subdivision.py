"""Opt-in subdivision indicator: cancellation, failure, and JAX contracts."""

from __future__ import annotations

from math import gamma

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hpfracc as hp


def _linear(t, state, params, *, rng_key=None, inputs=None):
    del t, rng_key, inputs
    return params * state


def _constant(t, state, params, *, rng_key=None, inputs=None):
    del t, params, rng_key, inputs
    return jnp.ones_like(state)


def _zero(t, state, params, *, rng_key=None, inputs=None):
    del t, params, rng_key, inputs
    return jnp.zeros_like(state)


def _cubic(t, state, params, *, rng_key=None, inputs=None):
    del state, params, rng_key, inputs
    return (t - t * (2 * t - 1) * (t - 1)) / 3


def _run(model=_linear, *, mode="subdivision", budget=128, first=0.01,
         tol=1e-3, y0=1.0, rate=-0.5, ts=None):
    return hp.solvers.simulate(
        model=model, ts=jnp.asarray([0.0, 1.0]) if ts is None else ts,
        solver=hp.solvers.AdaptivePredictorCorrector(
            order=0.5, rtol=tol, atol=tol * 0.01, max_steps=budget,
            first_step=first, error_estimator=mode,
        ), initial_state=jnp.asarray(y0), params=jnp.asarray(rate),
    )


def test_unknown_estimator_rejected():
    with pytest.raises(ValueError, match="error_estimator"):
        hp.solvers.AdaptivePredictorCorrector(order=0.5, error_estimator="unknown")


def test_default_remains_step_doubling():
    assert hp.solvers.AdaptivePredictorCorrector(order=0.5).error_estimator == (
        "step_doubling"
    )


def test_cubic_cancellation_trial_is_rejected(enable_x64):
    legacy = _run(_cubic, mode="step_doubling", budget=2, first=1, tol=1e-8, y0=0.)
    guarded = _run(_cubic, budget=2, first=1, tol=1e-8, y0=0.)
    assert legacy.solver_info.diagnostics["n_rejected"] == 0
    assert guarded.solver_info.diagnostics["n_rejected"] > 0
    assert not guarded.solver_info.diagnostics["success"]


def test_cubic_refines_to_a_more_accurate_complete_trajectory(enable_x64):
    result = _run(_cubic, first=1, budget=512, tol=1e-4, y0=0.)
    d = result.solver_info.diagnostics
    assert d["success"]
    t = np.asarray(result.ts[:d["n_nodes"]])
    truth = 2 * t**2.5 / gamma(3.5) - 4 * t**3.5 / gamma(4.5)
    assert np.max(np.abs(np.asarray(result.latent_state[:len(t)]) - truth)) < 1e-3


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_constant_forcing_and_last_attempt_keep_public_shape(dtype, enable_x64):
    result = _run(_constant, budget=2, first=.5, y0=jnp.zeros((2, 3), dtype=dtype),
                  ts=jnp.asarray([0., 1.], dtype=dtype))
    d = result.solver_info.diagnostics
    assert d["success"] and d["n_accepted"] == 2
    assert result.latent_state.shape == (4, 2, 3)
    assert result.latent_state.dtype == dtype
    np.testing.assert_array_equal(d["valid_mask"], [True, True, True, False])
    assert np.allclose(result.latent_state[-1], 1 / gamma(1.5), rtol=2e-6)
    assert d["estimator"] == "subdivision"
    assert d["estimator_stability"] == "experimental"
    assert d["error_norm"] == "max"


def test_quarter_only_nonfinite_trial_fails_without_acceptance():
    def model(t, state, params, *, rng_key=None, inputs=None):
        del params, rng_key, inputs
        return jnp.where(t == .25, jnp.full_like(state, jnp.nan), 0.)

    result = _run(model, first=1, budget=2)
    d = result.solver_info.diagnostics
    assert d["nonfinite_detected"] and not d["success"]
    assert d["n_attempted"] == d["n_rejected"] == 1
    assert d["n_accepted"] == 0
    assert np.all(np.isfinite(result.latent_state))


def test_quarter_nodes_require_stronger_time_resolution():
    start = jnp.asarray(1., dtype=jnp.float32)
    mid = jnp.nextafter(start, jnp.asarray(2., dtype=jnp.float32))
    end = jnp.nextafter(mid, jnp.asarray(2., dtype=jnp.float32))
    result = _run(_zero, first=float(end - start), budget=2,
                  ts=jnp.stack([start, end]), y0=jnp.asarray(1., dtype=jnp.float32))
    assert result.solver_info.diagnostics["step_size_underflow"]
    assert result.solver_info.diagnostics["n_accepted"] == 0


def test_inactive_components_do_not_dilute_the_indicator():
    def model(t, state, params, *, rng_key=None, inputs=None):
        return _cubic(t, state, params) * (jnp.arange(state.size) == 0)

    result = _run(model, first=1, budget=2, tol=1e-3, y0=jnp.zeros(1024))
    assert result.solver_info.diagnostics["n_rejected"] > 0


def test_completed_singleton_skips_quarter_probes():
    result = _run(_zero, ts=jnp.asarray([2.]), budget=2)
    d = result.solver_info.diagnostics
    assert d["success"] and d["n_attempted"] == 0
    np.testing.assert_array_equal(d["valid_mask"], [True, False, False, False])


def test_initial_state_and_order_gradients_are_finite(enable_x64):
    def objective(initial, order):
        result = hp.solvers.simulate(
            model=_linear, ts=jnp.asarray([0., 1.]),
            solver=hp.solvers.AdaptivePredictorCorrector(
                order=order, rtol=1e-3, atol=1e-5, max_steps=128,
                error_estimator="subdivision",
            ), initial_state=initial, params=jnp.asarray(-.5),
        )
        return result.latent_state[-1]

    gradients = jax.jit(jax.grad(objective, argnums=(0, 1)))(
        jnp.asarray(1.), jnp.asarray(.5),
    )
    assert all(np.isfinite(value) for value in gradients)


def test_jit_status_and_frozen_mesh_parameter_gradient(enable_x64):
    def solve(rate):
        result = _run(rate=rate)
        d = result.solver_info.diagnostics
        return result.latent_state, d["success"], d["valid_mask"]

    eager = solve(jnp.asarray(-.5))
    compiled = jax.jit(solve)(jnp.asarray(-.5))
    assert bool(compiled[1])
    for actual, expected in zip(compiled, eager, strict=True):
        assert np.allclose(actual, expected, rtol=1e-10, atol=1e-12)
    result = _run()
    mesh = result.ts[:result.solver_info.diagnostics["n_nodes"]]

    def frozen(rate):
        return hp.solvers.simulate(
            model=_linear, ts=mesh,
            solver=hp.solvers.NonUniformPredictorCorrector(order=0.5),
            initial_state=jnp.asarray(1.), params=rate,
        ).latent_state[-1]

    value = jnp.asarray(-.5)
    actual = jax.jit(jax.grad(lambda rate: solve(rate)[0][-1]))(value)
    expected = jax.grad(frozen)(value)
    fd = (frozen(value + 1e-5) - frozen(value - 1e-5)) / 2e-5
    assert np.isfinite(actual)
    assert np.allclose(actual, expected, rtol=1e-8, atol=1e-10)
    assert np.allclose(actual, fd, rtol=1e-6, atol=1e-8)
