"""Fixed SOE and stable-uniform cases for the aggregate numerical gate.

Execution is explicitly float32, with host float64 references. These selected
cases enforce tolerances; the larger measurement audits keep their own scope.
"""

from __future__ import annotations

import argparse
import sys
from math import gamma

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import erfcx

from benchmarks.numerical.core_hardening import HardeningRow, _row, write_csv
from benchmarks.numerical.implicit_validation import independent_residuals
from benchmarks.numerical.uniform_weight_validation import reference_row
from hpfracc.ops import caputo, riemann_liouville
from hpfracc.ops._soe import _fit_soe
from hpfracc.solvers import (
    AdaptiveForcingQuadrature,
    BoundedForcing,
    ImplicitPredictorCorrector,
    PredictorCorrector,
    simulate,
)
from hpfracc.solvers._weights import _stable_pece_weights


def _soe_metadata_ok(result, kernel):
    d = result.operator_info.diagnostics
    return (result.operator_info.history.value == "soe"
            and d.get("soe_fit") == "scaled_tail"
            and d.get("soe_application") == "recursive"
            and d.get("soe_target_kernel") == kernel
            and d.get("soe_fit_finite") is True)


def soe_analytic_rows() -> list[HardeningRow]:
    n, poles = 257, 12
    times = jnp.linspace(0., 1., n, dtype=jnp.float32)
    order = jnp.asarray(.5, dtype=jnp.float32)
    kwargs = dict(dt=1/256, order=order, history="soe", soe_fit="scaled_tail",
                  soe_poles=poles, return_info=True)
    cap = caputo(times**2, **kwargs)
    expected = 2 * np.asarray(times, dtype=np.float64)**1.5 / gamma(2.5)
    cap_error = float(np.max(np.abs(np.asarray(cap.values) - expected)))
    rl = riemann_liouville(jnp.ones(n, dtype=jnp.float32), **kwargs)
    expected_rl = np.asarray(times[n//2:], dtype=np.float64)**(-.5) / gamma(.5)
    rl_error = float(np.max(np.abs(np.asarray(rl.values[n//2:]) / expected_rl - 1)))
    return [
        _row("operator", "soe_caputo_scaled_tail_float32", "max_abs_error", cap_error,
             cap.values.shape == (n,) and cap.values.dtype == jnp.float32
             and np.all(np.isfinite(cap.values)) and _soe_metadata_ok(cap, "l1")
             and cap_error <= 5e-4,
             order=.5, nodes=n, poles=poles, fit="scaled_tail", precision="float32",
             reference="analytic_caputo_quadratic", tolerance=5e-4),
        _row("operator", "soe_rl_constant_scaled_tail_float32", "max_relative_error",
             rl_error, rl.values.shape == (n,) and rl.values.dtype == jnp.float32
             and np.all(np.isfinite(rl.values)) and np.all(np.asarray(rl.values) > 0)
             and _soe_metadata_ok(rl, "gl_prefix") and rl_error <= .02,
             order=.5, nodes=n, poles=poles, fit="scaled_tail", precision="float32",
             reference="analytic_rl_constant_t>=0.5", tolerance=.02),
    ]


def soe_recurrence_row() -> HardeningRow:
    n, poles, dt = 33, 8, 1/32
    t = jnp.linspace(0., 1., n, dtype=jnp.float32)
    values = jnp.stack((jnp.sin(3*t), .4 + t*t), axis=-1)
    order = jnp.asarray(.5, dtype=jnp.float32)
    actual = caputo(values, dt=dt, order=order, history="soe",
                    soe_fit="scaled_tail", soe_poles=poles)
    # Hold the fitted decays/coefficients fixed and independently sum their
    # convolution on the host. This tests application, not fit optimality.
    fit = _fit_soe(order, n-1, n_poles=poles, t_max=n*dt,
                   kernel="l1", method="scaled_tail")
    decays = np.asarray(fit.rhos, dtype=np.float64)
    coefficients = np.asarray(fit.coefficients, dtype=np.float64)
    weights = np.r_[1., (decays[None, :]**np.arange(n-2)[:, None]) @ coefficients]
    increments = np.diff(np.asarray(values, dtype=np.float64), axis=0)
    tail = np.stack([np.convolve(increments[:, i], weights)[:n-1]
                     for i in range(2)], axis=-1) / (gamma(1.5)*dt**.5)
    expected = np.concatenate((np.zeros((1, 2)), tail), axis=0)
    error = float(np.max(np.abs(np.asarray(actual) - expected)))
    return _row(
        "operator", "soe_scaled_tail_recurrence_float32", "max_abs_fitted_sum_error",
        error, actual.shape == (n, 2) and actual.dtype == jnp.float32 and error <= 5e-5,
        order=.5, nodes=n, poles=poles, fit="scaled_tail", precision="float32",
        reference="float64_direct_sum_of_actual_fitted_kernel", tolerance=5e-5,
    )


def soe_gradient_row() -> HardeningRow:
    n = 33
    t = jnp.linspace(0., 1., n, dtype=jnp.float32)

    def objective(order):
        return jnp.mean(caputo(t*t, dt=1/32, order=order,
                               history="soe", soe_fit="scaled_tail", soe_poles=8))

    order, epsilon = jnp.asarray(.5, dtype=jnp.float32), 1e-3
    compiled = jax.jit(objective)
    derivative = jax.jit(jax.grad(objective))(order)
    fd = (compiled(order + epsilon) - compiled(order - epsilon)) / (2*epsilon)
    error = float(jnp.abs(derivative - fd))
    return _row(
        "gradient", "soe_scaled_tail_order_gradient_float32", "abs_error", error,
        derivative.dtype == jnp.float32 and jnp.isfinite(derivative)
        and jnp.isfinite(fd) and jnp.abs(fd) > 1e-3 and error <= 2e-3,
        order=.5, nodes=n, poles=8, fit="scaled_tail", precision="float32",
        reference="centered_difference_of_compiled_objective", epsilon=epsilon,
        tolerance=2e-3, autodiff=float(derivative), finite_difference=float(fd),
    )


def stable_weight_row() -> HardeningRow:
    step, order = 256, jnp.asarray(.5, dtype=jnp.float32)
    pred, corr = _stable_pece_weights(jnp.asarray(step), jnp.arange(step+1),
                                      order, jnp.float32)
    expected_p, expected_c = reference_row(step, .5)
    error = float(max(np.max(np.abs(np.asarray(pred[:-1]) / expected_p - 1)),
                      np.max(np.abs(np.asarray(corr[:-1]) / expected_c - 1))))
    valid = (pred.shape == corr.shape == (step+1,)
             and pred.dtype == corr.dtype == jnp.float32
             and np.all(np.asarray(pred[:-1]) > 0) and np.all(np.asarray(corr[:-1]) > 0)
             and float(pred[-1]) == float(corr[-1]) == 0)
    return _row(
        "solver", "uniform_stable_weight_precision_float32", "max_relative_error",
        error, valid and error <= 5e-6, order=.5, steps=step,
        precision="float32", evaluation="stable",
        reference="float64_panel_quadrature", tolerance=5e-6,
    )


def stable_explicit_row() -> HardeningRow:
    n = 65
    times = jnp.linspace(0., 1., n, dtype=jnp.float32)
    solver = PredictorCorrector(dt=1/64, order=jnp.asarray(.5, dtype=jnp.float32),
                                weight_evaluation="stable")

    def constant(t, state, params, *, rng_key=None, inputs=None):
        return jnp.full_like(state, params)

    @jax.jit
    def run(initial):
        return simulate(model=constant, ts=times, solver=solver, initial_state=initial,
                        params=jnp.asarray(1., dtype=jnp.float32)).latent_state

    actual = run(jnp.zeros((2, 3), dtype=jnp.float32))
    expected = np.asarray(times, dtype=np.float64)[:, None, None]**.5 / gamma(1.5)
    error = float(np.max(np.abs(np.asarray(actual) - expected)))
    return _row(
        "solver", "uniform_stable_explicit_float32", "max_abs_error", error,
        actual.shape == (n, 2, 3) and actual.dtype == jnp.float32 and error <= 3e-6,
        order=.5, nodes=n, precision="float32", evaluation="stable", jit=True,
        reference="analytic_constant_forcing", tolerance=3e-6,
    )


def stable_implicit_row() -> HardeningRow:
    n = 257
    times = jnp.linspace(0., 1., n, dtype=jnp.float32)
    order = jnp.asarray(.5, dtype=jnp.float32)

    def decay(t, state, params, *, rng_key=None, inputs=None):
        return -params*state

    result = simulate(
        model=decay, ts=times, initial_state=jnp.ones(1, dtype=jnp.float32),
        params=jnp.asarray(1., dtype=jnp.float32),
        solver=ImplicitPredictorCorrector(dt=1/256, order=order, newton_iterations=8,
                                          newton_rtol=1e-5, newton_atol=1e-7,
                                          weight_evaluation="stable"),
    )
    values = np.asarray(result.latent_state, dtype=np.float64)
    residuals = independent_residuals("half_decay", np.asarray(times), values,
                                      1., .5, np.ones(1), 1e-5, 1e-7)
    maximum = float(np.max(residuals))
    expected = erfcx(np.sqrt(np.asarray(times, dtype=np.float64)))
    error = float(np.max(np.abs(values[:, 0] - expected)))
    d = result.solver_info.diagnostics
    return _row(
        "solver", "uniform_stable_implicit_float32", "independent_scaled_residual",
        maximum, result.latent_state.dtype == jnp.float32 and values.shape == (n, 1)
        and d["weight_evaluation"] == "stable" and d["newton_converged"]
        and d["newton_failure_count"] == d["newton_nonfinite_count"] == 0
        and np.all(np.isfinite(values)) and maximum <= 1 and error <= 1e-3,
        order=.5, strength=1., nodes=n, precision="float32", evaluation="stable",
        iterations=8, rtol=1e-5, atol=1e-7, threshold=1,
        reference="float64_panel_residual_and_erfcx", max_abs_error=error,
        error_tolerance=1e-3,
    )


def bounded_forcing_row() -> HardeningRow:
    def forcing(t, params, *, rng_key=None, inputs=None):
        return t*t

    result = simulate(
        model=BoundedForcing(forcing, 2.), ts=jnp.asarray([0., 1.], dtype=jnp.float32),
        initial_state=jnp.asarray(0., dtype=jnp.float32), params=None,
        solver=AdaptiveForcingQuadrature(order=jnp.asarray(.5, dtype=jnp.float32),
                                          atol=1e-3, first_step=1., max_steps=128),
    )
    d = result.solver_info.diagnostics
    n = int(d["n_nodes"])
    times = np.asarray(result.ts[:n], dtype=np.float64)
    values = np.asarray(result.latent_state[:n], dtype=np.float64)
    bound = np.asarray(d["interpolation_error_bound"][:n], dtype=np.float64)
    error = np.abs(values-2*times**2.5/gamma(3.5))
    maximum = float(np.max(error))
    return _row(
        "solver", "bounded_forcing_quadratic_float32", "max_abs_error", maximum,
        d["success"] and d["n_rejected"] > 0 and times[-1] == 1
        and result.latent_state.dtype == jnp.float32
        and np.all(np.isfinite(values)) and np.all(bound >= 0)
        and np.all(bound <= 1e-3*(1+2e-6)) and np.all(error <= bound+3e-6)
        and maximum <= 1e-3,
        order=.5, curvature=2., atol=1e-3, rtol=0., max_steps=128,
        precision="float32", reference="analytic_quadratic_forcing",
        scope="conditional_nodal_interpolation_bound", roundoff_slack=3e-6,
    )


def generate_rows() -> list[HardeningRow]:
    return [*soe_analytic_rows(), soe_recurrence_row(), soe_gradient_row(),
            stable_weight_row(), stable_explicit_row(), stable_implicit_row(),
            bounded_forcing_row()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    rows = generate_rows()
    write_csv(rows, sys.stdout)
    return int(args.check and (not rows or not all(row.passed for row in rows)))


if __name__ == "__main__":
    raise SystemExit(main())
