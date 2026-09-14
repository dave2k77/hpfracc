"""Independent coupled oracles, frozen-mesh sensitivities, and audit checks."""

from dataclasses import replace
from math import gamma

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from benchmarks.numerical import adaptive_coupled as audit
from hpfracc.solvers import (
    AdaptivePredictorCorrector,
    NonUniformPredictorCorrector,
    simulate,
)


@pytest.mark.parametrize("alpha", audit.ORDERS)
def test_manufactured_forcing_against_continuous_caputo(alpha, enable_x64):
    scale = jnp.asarray(audit.SCALES["mixed_scale"])
    for t in (.1, .7, 1.):
        # Weighted quadrature independently integrates the derivative of u(t).
        expected = np.asarray([
            quad(lambda s, k=k: k * s ** (k - 1), 0, t,
                 weight="alg", wvar=(0, -alpha), epsabs=1e-12)[0] / gamma(1 - alpha)
            for k in (2, 3)
        ])
        actual = audit.dynamics(
            t, scale * audit.reference(t), (scale, alpha, 1.),
        ) / scale
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-13)
    # Both off-diagonal feedback terms must actually affect off-oracle states.
    exact = scale * audit.reference(.5)
    jac = jax.jacfwd(lambda y: audit.dynamics(.5, y, (scale, alpha, 1.)))(exact)
    assert jac[0, 1] > 0 and jac[1, 0] > 0


@pytest.mark.parametrize("mode", audit.MODES)
def test_coupled_refinement_and_frozen_mesh_gradients(mode, enable_x64):
    rows = [audit.run_case("mixed_scale", mode, .5, tol, "float64", max_steps=256)
            for tol in audit.TOLERANCES]
    assert all(row.valid_output and row.completed for row in rows)
    assert rows[1].max_dimensionless_error < rows[0].max_dimensionless_error

    # Keep forcing fixed while varying the solver order: this tests the order
    # sensitivity of the integral equation, not a co-varying manufactured RHS.
    scale = jnp.asarray(audit.SCALES["mixed_scale"])
    initial = jnp.asarray([1., .5])

    def run(y0, feedback, alpha, mesh=None):
        solver = (AdaptivePredictorCorrector(
            order=alpha, rtol=1e-2, atol=1e-4, max_steps=128,
            error_estimator=mode,
        ) if mesh is None else NonUniformPredictorCorrector(order=alpha))
        return simulate(
            model=audit.dynamics, ts=jnp.asarray([0., 1.]) if mesh is None else mesh,
            solver=solver, initial_state=y0 * scale,
            params=(scale, jnp.asarray(.5), feedback),
        )

    result = run(initial, 1., .5)
    assert result.solver_info.diagnostics["success"]
    mesh = result.ts[:result.solver_info.diagnostics["n_nodes"]]
    weights = jnp.asarray([.3, .7])

    def objective(y0, feedback, alpha, mesh=None):
        return weights @ (run(y0, feedback, alpha, mesh).latent_state[-1] / scale)

    args = (initial, jnp.asarray(1.), jnp.asarray(.5))
    actual = jax.jit(jax.grad(objective, argnums=(0, 1, 2)))(*args)
    def frozen(y, p, a):
        return objective(y, p, a, mesh)

    expected = jax.grad(frozen, argnums=(0, 1, 2))(*args)
    for found, target in zip(actual, expected, strict=True):
        np.testing.assert_allclose(found, target, rtol=2e-8, atol=2e-10)
    epsilon = 1e-5
    for index in range(4):
        direction = jnp.eye(4)[index]

        def perturb(sign, direction=direction):
            return frozen(initial + sign * epsilon * direction[:2],
                          1. + sign * epsilon * direction[2],
                          .5 + sign * epsilon * direction[3])

        fd = (perturb(1) - perturb(-1)) / (2 * epsilon)
        gradient = jnp.concatenate((actual[0], jnp.stack(actual[1:])))
        np.testing.assert_allclose(gradient[index], fd, rtol=2e-5, atol=2e-8)


def test_measurement_detects_corrupt_padding_and_failed_runs(enable_x64):
    scale = jnp.asarray(audit.SCALES["balanced"])
    output = audit.runner("subdivision", 1e-2, 128)(
        scale * jnp.asarray([1., .5]), scale, jnp.asarray(.5), jnp.asarray(1.),
    )
    row = audit.summarize(output, "balanced", "subdivision", .5, 1e-2, "float64", 128)
    assert row.completed
    corrupt = (output[0], output[1].at[-1, 0].add(1.), output[2])
    row = audit.summarize(corrupt, "balanced", "subdivision", .5, 1e-2, "float64", 128)
    assert not row.valid_output and row.max_scaled_error is None
    failed = audit.run_case("balanced", "subdivision", .5, 1e-3,
                            "float64", max_steps=2)
    assert failed.valid_output and not failed.completed and failed.max_steps_reached
    assert failed.component_max_abs_error is None
    assert failed.endpoint_scaled_error is None


def test_checked_matrix_rejects_missing_duplicate_and_failed_rows(enable_x64):
    row = audit.run_case("balanced", "step_doubling", .5, 1e-2,
                         "float64", max_steps=128)
    assert row.completed
    rows = [replace(row, case=case, order=order, estimator=mode, rtol=tol)
            for case in audit.SCALES for order in audit.ORDERS
            for mode in audit.MODES for tol in audit.TOLERANCES]
    assert audit.rows_passed(rows, "float64", 128)
    assert not audit.rows_passed(rows[:-1], "float64", 128)
    assert not audit.rows_passed(rows[:-1] + [rows[0]], "float64", 128)
    assert not audit.rows_passed(rows[:-1] + [replace(rows[-1], completed=False)],
                                 "float64", 128)


def test_compiled_quarter_failure_retains_accepted_prefix():
    def model(t, state, params, *, rng_key=None, inputs=None):
        del params, rng_key, inputs
        # Only a quarter probe in the second trial reaches this point.
        return jnp.where(t == .1875, jnp.full_like(state, jnp.nan), 0.)

    @jax.jit
    def run(initial):
        result = simulate(
            model=model, ts=jnp.asarray([0., 1.]), initial_state=initial, params=None,
            solver=AdaptivePredictorCorrector(
                order=.5, first_step=.125, safety=1., max_steps=4,
                error_estimator="subdivision",
            ),
        )
        d = result.solver_info.diagnostics
        return result.ts, result.latent_state, {k: d[k] for k in audit.STATUS_FIELDS}

    times, states, d = jax.device_get(run(jnp.asarray([1., 2.])))
    assert d["nonfinite_detected"] and not d["success"]
    assert d["n_accepted"] == d["n_rejected"] == 1 and d["n_attempted"] == 2
    np.testing.assert_array_equal(times, [0., .125, .125, .125, .125, .125])
    np.testing.assert_array_equal(d["valid_mask"], [True, True, False, False,
                                                  False, False])
    np.testing.assert_array_equal(states, np.tile([1., 2.], (6, 1)))
