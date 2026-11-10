"""Residual-based diagnostics for the fixed-iteration implicit corrector."""

from __future__ import annotations

import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hpfracc as hp


def _cubic(t, state, params, *, rng_key=None, inputs=None):
    del t, rng_key, inputs
    return params * state**3


def _linear(t, state, params, *, rng_key=None, inputs=None):
    del t, rng_key, inputs
    return params * state


def _solve(solver, *, model=_cubic, y0=None, rate=-2.0, ts=None):
    return hp.solvers.simulate(
        model=model,
        ts=jnp.asarray([0.0, solver.dt]) if ts is None else ts,
        solver=solver,
        initial_state=jnp.asarray(1.0) if y0 is None else y0,
        params=jnp.asarray(rate),
    )


def _first_step_scaled_residual(solver, y, y0, rate):
    scale = solver.dt**solver.order / math.gamma(solver.order + 2)
    known = y0 + solver.order * scale * rate * y0**3
    residual = y - known - scale * rate * y**3
    tolerance = solver.newton_atol + solver.newton_rtol * np.maximum(
        np.abs(y), np.abs(known)
    )
    return float(np.max(np.abs(residual) / tolerance))


def test_underiterated_nonlinear_corrector_reports_failure(enable_x64):
    solver = hp.solvers.ImplicitPredictorCorrector(
        dt=0.5, order=0.7, newton_iterations=1
    )
    result = _solve(solver)
    info = result.solver_info
    diag = info.diagnostics
    assert diag["newton_converged"] is False
    assert diag["newton_failure_count"] == 1
    assert diag["newton_nonfinite_count"] == 0
    assert diag["newton_step_converged"] == [False]
    expected = _first_step_scaled_residual(
        solver, float(result.latent_state[-1]), 1.0, -2.0
    )
    assert expected > 1
    assert diag["newton_scaled_residuals"] == pytest.approx([expected], rel=1e-10)
    assert diag["newton_max_scaled_residual"] == pytest.approx(expected, rel=1e-10)
    assert any("Newton" in warning and "1" in warning for warning in info.warnings)
    json.dumps(info.to_dict(), allow_nan=False)


def test_more_newton_iterations_resolve_nonlinear_residual(enable_x64):
    solver = hp.solvers.ImplicitPredictorCorrector(
        dt=0.5, order=0.7, newton_iterations=8
    )
    result = _solve(solver)
    diag = result.solver_info.diagnostics
    assert diag["newton_converged"] is True
    assert diag["newton_failure_count"] == 0
    assert diag["newton_nonfinite_count"] == 0
    assert diag["newton_step_converged"] == [True]
    assert diag["newton_max_scaled_residual"] <= 1
    assert result.solver_info.warnings == ()


def test_linear_newton_reports_all_steps_converged():
    solver = hp.solvers.ImplicitPredictorCorrector(
        dt=0.1, order=0.7, newton_iterations=1
    )
    result = _solve(solver, model=_linear, ts=jnp.arange(6) * 0.1)
    diag = result.solver_info.diagnostics
    assert diag["newton_converged"] is True
    assert diag["newton_step_converged"] == [True] * 5
    assert len(diag["newton_scaled_residuals"]) == 5
    assert diag["newton_max_scaled_residual"] <= 1
    assert result.solver_info.warnings == ()


def test_residual_criterion_checks_every_state_component(enable_x64):
    solver = hp.solvers.ImplicitPredictorCorrector(
        dt=0.5, order=0.7, newton_iterations=1,
        newton_atol=0.1, newton_rtol=1e-12,
    )
    y0 = jnp.asarray([[1.0, 0.0], [0.0, 0.0]])
    result = _solve(solver, y0=y0)
    expected = _first_step_scaled_residual(
        solver, np.asarray(result.latent_state[-1]), np.asarray(y0), -2.0
    )
    # Only one component has a residual. An RMS over all four would hide it.
    assert 1 < expected < 2
    diag = result.solver_info.diagnostics
    assert diag["newton_converged"] is False
    assert diag["newton_max_scaled_residual"] == pytest.approx(expected, rel=1e-10)


def test_tolerances_only_change_diagnostics_not_updates_or_gradients(enable_x64):
    strict = hp.solvers.ImplicitPredictorCorrector(
        dt=0.5, order=0.7, newton_iterations=1,
        newton_rtol=1e-8, newton_atol=1e-10,
    )
    loose = hp.solvers.ImplicitPredictorCorrector(
        dt=0.5, order=0.7, newton_iterations=1,
        newton_rtol=1.0, newton_atol=1.0,
    )
    a, b = _solve(strict), _solve(loose)
    assert a.solver_info.diagnostics["newton_converged"] is False
    assert b.solver_info.diagnostics["newton_converged"] is True
    np.testing.assert_array_equal(a.latent_state, b.latent_state)
    grad_a = jax.grad(lambda y: _solve(strict, y0=y).latent_state[-1])(1.0)
    grad_b = jax.grad(lambda y: _solve(loose, y0=y).latent_state[-1])(1.0)
    assert jnp.isfinite(grad_a)
    np.testing.assert_array_equal(grad_a, grad_b)


@pytest.mark.parametrize("iterations", (1, 8))
def test_newton_diagnostics_remain_available_under_jit(enable_x64, iterations):
    solver = hp.solvers.ImplicitPredictorCorrector(
        dt=0.5, order=0.7, newton_iterations=iterations
    )

    def run(rate):
        result = _solve(solver, rate=rate)
        diag = result.solver_info.diagnostics
        return (
            result.latent_state,
            diag["newton_converged"],
            diag["newton_failure_count"],
            diag["newton_step_converged"],
            diag["newton_scaled_residuals"],
        )

    actual = jax.jit(run)(-2.0)
    expected = run(-2.0)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)
    assert bool(actual[1]) is (iterations == 8)


def test_nonfinite_newton_is_failure_with_strict_json_export():
    solver = hp.solvers.ImplicitPredictorCorrector(dt=0.1, order=0.7)
    result = _solve(solver, rate=float("nan"), ts=jnp.asarray([0.0, 0.1, 0.2]))
    diag = result.solver_info.diagnostics
    assert diag["newton_converged"] is False
    assert diag["newton_failure_count"] == 2
    assert diag["newton_nonfinite_count"] == 2
    assert diag["newton_step_converged"] == [False, False]
    assert diag["newton_scaled_residuals"] == [None, None]
    assert diag["newton_max_scaled_residual"] is None
    assert result.solver_info.warnings
    json.dumps(result.solver_info.to_dict(), allow_nan=False)

    def compiled_status(rate):
        info = _solve(
            solver, rate=rate, ts=jnp.asarray([0.0, 0.1, 0.2])
        ).solver_info
        return (
            info.diagnostics["newton_converged"],
            info.diagnostics["newton_nonfinite_count"],
            info.diagnostics["newton_scaled_residuals"],
        )

    converged, count, residuals = jax.jit(compiled_status)(float("nan"))
    assert not bool(converged)
    assert int(count) == 2
    assert jnp.all(jnp.isinf(residuals))


def test_singular_newton_jacobian_is_reported():
    from hpfracc.ops._special import _rgamma1p

    # Make the corrector coefficient exactly one at working precision. With
    # f(y)=y, the residual Jacobian is zero and the nonzero RHS is inconsistent.
    normalization = _rgamma1p(0.5)
    dt = float(1.5 / normalization) ** 2
    assert float(dt**0.5 * normalization / 1.5) == 1.0
    solver = hp.solvers.ImplicitPredictorCorrector(
        dt=dt, order=0.5, newton_iterations=1
    )
    result = _solve(solver, model=_linear, rate=1.0)
    diag = result.solver_info.diagnostics
    assert not jnp.all(jnp.isfinite(result.latent_state))
    assert diag["newton_converged"] is False
    assert diag["newton_nonfinite_count"] == 1
    assert diag["newton_failure_count"] == 1
    assert result.solver_info.warnings
    json.dumps(result.solver_info.to_dict(), allow_nan=False)


def test_single_node_has_no_failed_newton_steps():
    solver = hp.solvers.ImplicitPredictorCorrector(dt=0.1, order=0.7)
    result = _solve(solver, ts=jnp.asarray([0.0]))
    diag = result.solver_info.diagnostics
    assert diag["newton_converged"] is True
    assert diag["newton_failure_count"] == 0
    assert diag["newton_nonfinite_count"] == 0
    assert diag["newton_scaled_residuals"] == []
    assert diag["newton_step_converged"] == []
    assert diag["newton_max_scaled_residual"] == 0
    assert result.solver_info.warnings == ()


@pytest.mark.parametrize("field", ("newton_rtol", "newton_atol"))
@pytest.mark.parametrize("value", (0.0, -1.0, float("nan"), float("inf")))
def test_newton_tolerances_require_finite_positive_values(field, value):
    with pytest.raises(ValueError, match=field):
        hp.solvers.ImplicitPredictorCorrector(dt=0.1, order=0.7, **{field: value})
