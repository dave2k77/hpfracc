"""Adaptive terminal status, failure handling, and compiled output masks."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hpfracc as hp


def _zero(t, state, params, *, rng_key=None, inputs=None):
    del t, params, rng_key, inputs
    return jnp.zeros_like(state)


def _linear(t, state, params, *, rng_key=None, inputs=None):
    del t, rng_key, inputs
    return params * state


def _fails_later(t, state, params, *, rng_key=None, inputs=None):
    del params, rng_key, inputs
    return jnp.where(t > 0.1, jnp.full_like(state, jnp.nan), jnp.zeros_like(state))


def _run(solver, *, model=_zero, ts=None, y0=1.0, rate=-1.0):
    return hp.solvers.simulate(
        model=model,
        ts=jnp.asarray([0.0, 1.0]) if ts is None else ts,
        solver=solver,
        initial_state=jnp.asarray(y0),
        params=jnp.asarray(rate),
    )


def _check_counts_and_padding(result, budget):
    diag = result.solver_info.diagnostics
    n = diag["n_nodes"]
    mask = np.asarray(diag["valid_mask"])
    assert mask.shape == (budget + 2,)
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, np.arange(budget + 2) < n)
    assert n == diag["n_accepted"] + 1
    assert result.solver_info.n_steps == diag["n_accepted"]
    assert diag["n_attempted"] == diag["n_accepted"] + diag["n_rejected"]
    assert diag["n_attempted"] <= budget
    assert bool(jnp.all(jnp.diff(result.ts[:n]) > 0))
    assert bool(jnp.all(result.ts[n:] == result.ts[n - 1]))
    assert bool(jnp.all(result.latent_state[n:] == result.latent_state[n - 1]))
    assert diag["last_time"] == float(result.ts[n - 1])
    json.dumps(result.solver_info.to_dict(), allow_nan=False)


@pytest.mark.parametrize("first_step,success", ((0.5, True), (0.01, False)))
def test_completion_on_last_attempt_is_not_budget_failure(first_step, success):
    solver = hp.solvers.AdaptivePredictorCorrector(
        order=0.7, first_step=first_step, max_steps=2
    )
    result = _run(solver)
    diag = result.solver_info.diagnostics
    assert diag["success"] is success
    assert diag["reached_t_end"] is success
    assert diag["max_steps_reached"] is (not success)
    assert diag["nonfinite_detected"] is False
    assert diag["step_size_underflow"] is False
    assert diag["n_attempted"] == 2
    assert bool(result.solver_info.warnings) is (not success)
    _check_counts_and_padding(result, 2)


@pytest.mark.parametrize("first_step", (0.5, 0.01))
def test_adaptive_status_and_mask_are_available_under_jit(first_step):
    solver = hp.solvers.AdaptivePredictorCorrector(
        order=0.7, first_step=first_step, max_steps=2
    )

    def run(y0):
        result = _run(solver, y0=y0)
        d = result.solver_info.diagnostics
        return (
            result.latent_state, result.solver_info.n_steps,
            d["success"], d["reached_t_end"], d["max_steps_reached"],
            d["n_nodes"], d["n_rejected"], d["valid_mask"],
        )

    eager = run(1.0)
    compiled = jax.jit(run)(1.0)
    assert all(value is not None for value in compiled)
    for actual, expected in zip(compiled, eager, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_nonfinite_trial_preserves_last_accepted_prefix():
    solver = hp.solvers.AdaptivePredictorCorrector(
        order=0.7, first_step=0.05, max_steps=16
    )
    result = _run(solver, model=_fails_later)
    diag = result.solver_info.diagnostics
    assert diag["success"] is False
    assert diag["nonfinite_detected"] is True
    assert diag["max_steps_reached"] is False
    assert diag["step_size_underflow"] is False
    assert diag["n_accepted"] == 1
    assert diag["n_rejected"] == 1
    assert diag["n_attempted"] == 2
    assert diag["last_time"] == pytest.approx(0.05)
    assert jnp.all(jnp.isfinite(result.latent_state))
    assert result.solver_info.warnings
    _check_counts_and_padding(result, 16)

    compiled = jax.jit(
        lambda y: _run(solver, model=_fails_later, y0=y)
        .solver_info.diagnostics["nonfinite_detected"]
    )(1.0)
    assert bool(compiled)


@pytest.mark.parametrize("bad", ("state", "dynamics"))
def test_nonfinite_initial_data_stops_before_any_attempt(bad):
    solver = hp.solvers.AdaptivePredictorCorrector(order=0.7, max_steps=4)
    result = _run(
        solver, model=_linear,
        y0=float("nan") if bad == "state" else 1.0,
        rate=float("nan") if bad == "dynamics" else -1.0,
    )
    diag = result.solver_info.diagnostics
    assert diag["success"] is False
    assert diag["nonfinite_detected"] is True
    assert diag["n_attempted"] == 0
    assert diag["n_accepted"] == 0
    assert diag["n_nodes"] == 1
    assert result.solver_info.warnings


def test_unrepresentable_midpoint_reports_step_size_underflow():
    start = jnp.asarray(1.0, dtype=jnp.float32)
    end = jnp.nextafter(start, jnp.asarray(2.0, dtype=jnp.float32))
    solver = hp.solvers.AdaptivePredictorCorrector(
        order=0.7, first_step=float(end - start), max_steps=4
    )
    result = _run(solver, ts=jnp.stack([start, end]))
    diag = result.solver_info.diagnostics
    assert diag["success"] is False
    assert diag["step_size_underflow"] is True
    assert diag["nonfinite_detected"] is False
    assert diag["max_steps_reached"] is False
    assert diag["n_attempted"] == 1
    assert diag["n_accepted"] == 0
    _check_counts_and_padding(result, 4)


def test_minimum_step_does_not_force_accept_failed_tolerance(enable_x64):
    solver = hp.solvers.AdaptivePredictorCorrector(
        order=0.4, first_step=1e-14, rtol=1e-30, atol=1e-30, max_steps=4
    )
    result = _run(solver, model=_linear)
    diag = result.solver_info.diagnostics
    assert diag["success"] is False
    assert diag["step_size_underflow"] is True
    assert diag["n_accepted"] == 0
    assert diag["n_attempted"] == 1
    assert result.solver_info.warnings


def test_tiny_horizon_must_reach_the_actual_endpoint():
    solver = hp.solvers.AdaptivePredictorCorrector(order=0.7, max_steps=8)
    ts = jnp.asarray([0.0, 1e-14])
    result = _run(solver, ts=ts)
    assert result.solver_info.diagnostics["reached_t_end"] is True
    assert float(result.ts[-1]) == float(ts[-1])
    _check_counts_and_padding(result, 8)


def test_single_node_completes_without_attempts():
    solver = hp.solvers.AdaptivePredictorCorrector(order=0.7, max_steps=4)
    result = _run(solver, ts=jnp.asarray([2.0]))
    diag = result.solver_info.diagnostics
    assert diag["success"] is True
    assert diag["n_attempted"] == 0
    assert diag["n_nodes"] == 1
    assert result.solver_info.warnings == ()
    _check_counts_and_padding(result, 4)


def test_rejected_trials_consume_the_attempt_budget():
    solver = hp.solvers.AdaptivePredictorCorrector(
        order=0.4, first_step=1e-3, rtol=1e-10, atol=1e-12, max_steps=3
    )
    result = _run(solver, model=_linear)
    diag = result.solver_info.diagnostics
    assert diag["max_steps_reached"] is True
    assert diag["n_accepted"] == 0
    assert diag["n_rejected"] == 3
    assert diag["n_attempted"] == 3
    _check_counts_and_padding(result, 3)


def test_completed_adaptive_solve_retains_compiled_gradient():
    solver = hp.solvers.AdaptivePredictorCorrector(
        order=0.7, first_step=0.5, max_steps=8
    )

    def endpoint(y0):
        return _run(solver, y0=y0).latent_state[-1]

    # Zero dynamics leave y0 unchanged, including through the inactive scan tail.
    assert float(jax.jit(jax.grad(endpoint))(1.0)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "integer_dtype,state_dtype,start,end",
    (
        (jnp.int32, jnp.float32, 2**24, 2**24 + 1),
        (jnp.int64, jnp.float64, 2**53, 2**53 + 1),
        (jnp.int32, jnp.float32, 0, 2**24 + 1),
        (jnp.int32, jnp.float32, 0, 2**31 - 1),
        (jnp.int64, jnp.float64, 0, 2**63 - 1),
        (jnp.uint64, jnp.float64, 0, 2**64 - 1),
    ),
)
def test_lossy_integer_bounds_cannot_report_success(
    enable_x64, integer_dtype, state_dtype, start, end
):
    solver = hp.solvers.AdaptivePredictorCorrector(order=0.7, max_steps=8)
    ts = jnp.asarray([start, end], dtype=integer_dtype)

    def run(y0):
        result = _run(solver, ts=ts, y0=y0)
        diag = result.solver_info.diagnostics
        return (
            diag["success"], diag["step_size_underflow"],
            diag["n_attempted"], diag["requested_t_end"],
        )

    y0 = jnp.asarray(1.0, dtype=state_dtype)
    for actual in (run(y0), jax.jit(run)(y0)):
        assert not bool(actual[0])
        assert bool(actual[1])
        assert int(actual[2]) == 0
        assert int(actual[3]) == end


@pytest.mark.parametrize("nodes", ([0.0], [0.0, 1.0]))
@pytest.mark.parametrize("weak_state", (False, True))
def test_narrow_dynamics_cannot_overflow_stored_initial_state(
    enable_x64, nodes, weak_state
):
    solver = hp.solvers.AdaptivePredictorCorrector(order=0.7, max_steps=8)

    def narrow_zero(t, state, params, *, rng_key=None, inputs=None):
        del t, params, rng_key, inputs
        return jnp.zeros_like(state, dtype=jnp.float32)

    ts = jnp.asarray(nodes)
    y0 = jnp.asarray(1e100) if weak_state else jnp.asarray(1e100, dtype=jnp.float64)

    def run(initial):
        result = _run(solver, model=narrow_zero, ts=ts, y0=initial)
        return result.latent_state, result.solver_info.diagnostics["success"]

    for states, success in (run(y0), jax.jit(run)(y0)):
        assert bool(success)
        assert states.dtype == jnp.float64
        assert bool(jnp.all(jnp.isfinite(states)))
        assert bool(jnp.all(states == y0))


@pytest.mark.parametrize("field", ("rtol", "atol", "first_step", "max_factor"))
def test_adaptive_configuration_rejects_infinite_values(field):
    with pytest.raises(ValueError, match=field):
        hp.solvers.AdaptivePredictorCorrector(order=0.7, **{field: float("inf")})
