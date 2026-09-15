"""Independent report oracle checks and failure-sensitive envelope validation."""

from dataclasses import replace
from math import gamma

import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from benchmarks.numerical import bounded_forcing_validation as report


def test_polynomial_oracles_against_continuous_quadrature():
    for alpha in report.ORDERS:
        for time in (.1, .7, 1.):
            for case in ("affine", "quadratic", "cubic_blind_spot"):
                def q(u, time=time, case=case):
                    s = time*u
                    if case == "affine":
                        return 1+2*s
                    return s*s if case == "quadratic" else s*s-2*s**3/3
                expected = time**alpha/gamma(alpha)*quad(
                    q, 0, 1, weight="alg", wvar=(0, alpha-1), epsabs=1e-12,
                )[0]
                actual = report.reference(case, np.asarray([time]), alpha)[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def test_checked_case_and_incomplete_error_suppression():
    row = report.run_case("quadratic", .5, 1e-3, "float32", max_steps=128)
    assert row.passed and row.completed and row.max_reference_scaled_error < 1
    failed = report.run_case("quadratic", .5, 1e-4, "float32", max_steps=1)
    assert not failed.completed and not failed.passed and failed.max_steps_reached
    assert failed.max_abs_error is None and failed.max_bound_budget_ratio is None
    output = report.runner("quadratic", 1e-4, 1)(
        jnp.asarray(0., dtype=jnp.float32),
        jnp.asarray([0., 1.], dtype=jnp.float32),
        jnp.asarray(.5, dtype=jnp.float32),
    )
    d = {**output[2], "success": jnp.asarray(True)}
    corrupted = report.summarize((output[0], output[1], d), "quadratic",
                                 "float32", .5, 1e-4, 1)
    assert not corrupted.completed and not corrupted.passed


def test_corrupted_envelope_fails_independent_check():
    output = report.runner("quadratic", 1e-3, 128)(
        jnp.asarray(0., dtype=jnp.float32),
        jnp.asarray([0., 1.], dtype=jnp.float32),
        jnp.asarray(.5, dtype=jnp.float32),
    )
    assert output[2]["success"]
    assert report.summarize(output, "quadratic", "float32", .5, 1e-3, 128).passed
    d = {**output[2], "interpolation_error_bound": jnp.zeros_like(output[1])}
    row = report.summarize((output[0], output[1], d), "quadratic", "float32", .5,
                           1e-3, 128)
    assert not row.envelope_consistent and not row.passed


def test_exact_matrix_gate_and_failure_sensitivity():
    row = report.run_case("quadratic", .5, 1e-3, "float32", max_steps=128)
    rows = [replace(row, case=case, order=float(np.float32(a)), atol=tol)
            for case in report.CASES for a in report.ORDERS
            for tol in report.TOLERANCES]
    assert report.rows_passed(rows, "float32", 128)
    assert not report.rows_passed(rows[:-1], "float32", 128)
    assert not report.rows_passed(rows[:-1]+[rows[0]], "float32", 128)
    assert not report.rows_passed(rows[:-1]+[replace(rows[-1], passed=False)],
                                  "float32", 128)
