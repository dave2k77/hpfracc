"""Independent nonlinear/stiff oracles, gradients, and diagnostic boundaries."""

from dataclasses import replace
from math import gamma

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erfcx

from benchmarks.numerical import implicit_validation as audit
from hpfracc.solvers import ImplicitPredictorCorrector, simulate


@pytest.mark.parametrize("alpha", [.3, .5, .8])
def test_continuous_manufactured_oracles_and_panel_moments(alpha, enable_x64):
    for case in ("cubic", "coupled", "scaled"):
        scales = jnp.asarray(audit.SCALES[case])
        for t in (.1, .7, 1.):
            exact = jnp.asarray(audit.reference(case, np.asarray(t), 30., scales))
            rhs = audit.dynamics(case, jnp.asarray(t), exact, (30., alpha, scales))
            # Continuous Caputo integral of u'(s), independent of PECE weights.
            integral = np.asarray([
                quad(lambda s, power=power: power * s**(power - 1), 0, t,
                     weight="alg", wvar=(0., -alpha), epsabs=1e-12)[0]
                / gamma(1 - alpha) for power in (2, 3)
            ])[:len(scales)]
            np.testing.assert_allclose(rhs / scales, integral, rtol=3e-12, atol=1e-12)
    n = 65
    weights = audit.quadrature_weights(n, alpha)
    times = np.linspace(0, 1, n)
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights @ np.ones(n), times**alpha / gamma(alpha + 1),
                               rtol=3e-13, atol=1e-14)
    np.testing.assert_allclose(weights @ times, times**(alpha + 1) / gamma(alpha + 2),
                               rtol=3e-13, atol=1e-14)


def test_continuous_half_decay_oracle_at_strong_rate():
    strength, t = 30., 1.
    # s=t*sin(theta)^2 removes both endpoint singularities of the Caputo
    # integral, using y'=rate^2*y-rate/sqrt(pi*s). No ML power-series truncation.
    integral = quad(lambda theta: (
        2 * np.sqrt(t) * np.sin(theta) * strength**2
        * erfcx(strength * np.sqrt(t) * np.sin(theta))
        - 2 * strength / np.sqrt(np.pi)
    ), 0, np.pi / 2, epsabs=1e-11)[0] / np.sqrt(np.pi)
    expected = -strength * erfcx(strength * np.sqrt(t))
    np.testing.assert_allclose(integral, expected, rtol=1e-10, atol=1e-10)


def test_manufactured_oracle_detects_wrong_rhs(monkeypatch, enable_x64):
    original = audit.dynamics
    monkeypatch.setattr(audit, "dynamics", lambda *args, **kwargs:
                        2 * original(*args, **kwargs))
    with pytest.raises(AssertionError):
        test_continuous_manufactured_oracles_and_panel_moments(.5, enable_x64)


def test_converged_stiff_linear_step_can_be_negative(enable_x64):
    outputs = []
    for iterations in (1, 8):
        output = audit.runner("half_decay", 17, iterations, "float64")(
            jnp.ones(1), jnp.asarray(30.), jnp.asarray(.5),
        )
        assert output[2]["newton_converged"]
        outputs.append(np.asarray(output[1]))
    mu = 30 * (1 / 16)**.5 / gamma(2.5)
    discrete_first = (1 - .5 * mu) / (1 + mu)
    exact_first = erfcx(30 * np.sqrt(1 / 16))
    assert discrete_first < 0 < exact_first
    np.testing.assert_allclose(outputs[0][1, 0], discrete_first, atol=1e-13)
    np.testing.assert_allclose(outputs[1], outputs[0], rtol=1e-12, atol=1e-13)
    row = audit.run_case("half_decay", 17, 8, 30., .5, "float64")
    assert row.measurement_ok and row.newton_converged
    assert row.minimum_dimensionless_state < 0 and row.max_dimensionless_error > .3
    fine = audit.run_case("half_decay", 257, 8, 30., .5, "float64")
    assert fine.newton_converged
    assert fine.max_dimensionless_error < row.max_dimensionless_error


def test_nonlinear_first_root_and_full_trajectory_residual(enable_x64):
    output = audit.runner("cubic", 17, 8, "float64")(
        jnp.ones(1), jnp.asarray(30.), jnp.asarray(.3),
    )
    h, alpha, strength = 1 / 16, .3, 30.
    scale = h**alpha / gamma(alpha + 2)
    forcing = 2 * h**(2 - alpha) / gamma(3 - alpha)
    root = brentq(lambda y: y - 1 - scale * (
        forcing - strength * (y**3 - (1 + h*h)**3)
    ), 0., 3., xtol=1e-14)
    np.testing.assert_allclose(output[1][1, 0], root, rtol=1e-13, atol=1e-13)
    # A finite, under-iterated trajectory has large independently reconstructed
    # residuals; reporting it as an accurate measurement does not certify a solve.
    bad = audit.run_case("cubic", 17, 1, 1., .5, "float64")
    assert bad.measurement_ok and bad.finite_trajectory and not bad.newton_converged
    assert bad.independent_max_scaled_residual > 1e3
    np.testing.assert_allclose(bad.reported_max_scaled_residual,
                               bad.independent_max_scaled_residual,
                               rtol=1e-8, atol=.01)


@pytest.mark.parametrize("case", ["cubic", "coupled", "scaled"])
def test_nonlinear_refinement_with_converged_correctors(case, enable_x64):
    rows = [audit.run_case(case, n, 8, 1., .5, "float64") for n in (33, 65, 129)]
    assert all(r.measurement_ok and r.newton_converged for r in rows)
    errors = np.asarray([r.max_dimensionless_error for r in rows])
    rates = np.log2(errors[:-1] / errors[1:])
    # Selected smooth manufactured cases exhibit near-second-order max errors.
    assert np.all((rates > 1.7) & (rates < 2.3)), (errors, rates)
    assert errors[-1] < 2e-5


def test_coupled_scaling_and_active_nonlinearity(enable_x64):
    results = []
    for case in ("coupled", "scaled"):
        scales = jnp.asarray(audit.SCALES[case])
        initial = scales * jnp.asarray([1., .5])
        output = audit.runner(case, 65, 8, "float64")(
            initial, jnp.asarray(30.), jnp.asarray(.5),
        )
        assert output[2]["newton_converged"]
        results.append(output[1] / scales)
    np.testing.assert_allclose(results[0], results[1], rtol=2e-11, atol=2e-11)
    scales = jnp.ones(2)
    def fn(y):
        return audit.dynamics("coupled", .5, y, (30., .5, scales))

    jac = jax.jacfwd(fn)(jnp.asarray([1.25, .625]))
    assert jac[0, 1] > 0 and jac[1, 0] > 0
    assert jnp.max(jnp.abs(jax.jacfwd(jax.jacfwd(fn))(jnp.ones(2)))) > 1


@pytest.mark.parametrize("case", ["cubic", "scaled"])
@pytest.mark.parametrize("alpha", [.3, .5, .8])
def test_nonlinear_initial_parameter_and_order_gradients(case, alpha, enable_x64):
    scales = jnp.asarray(audit.SCALES[case])
    y0 = jnp.asarray([1.] if len(scales) == 1 else [1., .5])
    times = jnp.linspace(0., 1., 33)

    def model(t, state, params, *, rng_key=None, inputs=None):
        return audit.dynamics(case, t, state, params, rng_key=rng_key, inputs=inputs)

    def run(initial, strength, order):
        result = simulate(
            model=model, ts=times, initial_state=scales * initial,
            # Hold manufactured forcing fixed during the order perturbation.
            params=(strength, jnp.asarray(alpha), scales),
            solver=ImplicitPredictorCorrector(dt=1 / 32, order=order,
                                              newton_iterations=8),
        )
        return result.latent_state / scales, result.solver_info.diagnostics[
            "newton_converged"
        ]

    args = (y0, jnp.asarray(3.), jnp.asarray(alpha))
    eager, converged = run(*args)
    compiled, compiled_converged = jax.jit(run)(*args)
    assert converged and compiled_converged
    np.testing.assert_allclose(compiled, eager, rtol=1e-10, atol=1e-11)

    def objective(initial, strength, order):
        return jnp.mean(run(initial, strength, order)[0][-1])

    gradients = jax.jit(jax.grad(objective, argnums=(0, 1, 2)))(*args)
    for i in range(len(y0)):
        direction = jnp.eye(len(y0))[i] * 1e-5
        fd = (objective(y0 + direction, args[1], args[2])
              - objective(y0 - direction, args[1], args[2])) / 2e-5
        np.testing.assert_allclose(gradients[0][i], fd, rtol=3e-5, atol=2e-8)
    for index, epsilon in ((1, 1e-3), (2, 1e-5)):
        plus, minus = list(args), list(args)
        plus[index], minus[index] = args[index] + epsilon, args[index] - epsilon
        fd = (objective(*plus) - objective(*minus)) / (2 * epsilon)
        np.testing.assert_allclose(gradients[index], fd, rtol=3e-4, atol=3e-8)


def test_failed_trajectories_and_corrupted_diagnostic_contract(enable_x64):
    failed = audit.run_case("cubic", 17, 1, 30., .3, "float64")
    assert failed.measurement_ok and not failed.finite_trajectory
    assert failed.newton_nonfinite_count > 0 and not failed.newton_converged
    assert failed.max_dimensionless_error is None
    assert failed.independent_max_scaled_residual is None
    output = audit.runner("cubic", 17, 8, "float64")(
        jnp.ones(1), jnp.asarray(1.), jnp.asarray(.5),
    )
    bad_diagnostics = {**output[2], "newton_failure_count": jnp.asarray(16)}
    corrupt = audit.summarize((output[0], output[1], bad_diagnostics),
                              "cubic", "float64", .5, 1., 17, 8)
    assert not corrupt.measurement_ok
    # Independent residuals react to a corrupted state even if flags are untouched.
    changed = audit.summarize((output[0], output[1].at[-1, 0].add(.1), output[2]),
                              "cubic", "float64", .5, 1., 17, 8)
    assert changed.independent_max_scaled_residual > 1e6
    assert changed.residual_flag_disagreements > 0


def test_matrix_gate_requires_all_cases_and_retains_labeled_failures(enable_x64):
    failed = audit.run_case("cubic", 17, 1, 30., .3, "float64")
    rows = [replace(failed, case=case, order=a, strength=k, n_samples=n,
                    newton_iterations=i)
            for case in audit.SCALES for a in audit.orders(case)
            for k in audit.STRENGTHS for n in audit.LENGTHS for i in audit.ITERATIONS]
    assert len(rows) == 180 and audit.rows_passed(rows, "float64")
    assert not audit.rows_passed(rows[:-1], "float64")
    assert not audit.rows_passed(rows[:-1] + [rows[0]], "float64")
    rows[-1] = replace(rows[-1], measurement_ok=False)
    assert not audit.rows_passed(rows, "float64")
