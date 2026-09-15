"""Reference consistency and complete, discriminating report gates."""

from dataclasses import replace
from math import gamma

import numpy as np

from benchmarks.numerical import implicit_validation as implicit
from benchmarks.numerical import uniform_weight_validation as audit


def test_row_quadrature_matches_independent_matrix():
    alpha, step = .3, 16
    pred, corr = audit.reference_row(step, alpha)
    weights = np.r_[corr, 1.] * (1 / step)**alpha / gamma(alpha + 2)
    np.testing.assert_allclose(
        weights, implicit.quadrature_weights(step + 1, alpha)[-1],
        rtol=2e-14, atol=2e-14,
    )
    np.testing.assert_allclose(np.sum(pred), step**alpha, rtol=2e-14)


def test_coefficient_gate_detects_errors_and_incomplete_matrices():
    row = audit.coefficient_case("float32", "stable", .5, 256)
    assert row.measurement_ok and row.max_relative_corrector_error < 5e-6
    rows = [replace(row, order=float(np.float32(a)), step=step, weight_evaluation=e)
            for a in audit.ORDERS for step in audit.STEPS for e in audit.EVALUATIONS]
    assert audit.coefficients_passed(rows, "float32")
    assert not audit.coefficients_passed(rows[:-1], "float32")
    assert not audit.coefficients_passed(rows[:-1] + [rows[0]], "float32")
    rows[-1] = replace(rows[-1], max_relative_corrector_error=.1)
    assert not audit.coefficients_passed(rows, "float32")


def test_solver_measurement_and_matrix_gate():
    row = audit.solver_case("half_decay", 17, 30., .5, "float32", "stable")
    assert row.result.measurement_ok and row.result.newton_converged
    assert row.result.minimum_dimensionless_state < 0
    rows = [audit.SolverRow(e, replace(
        row.result, case=case, order=float(np.float32(a)), strength=k, n_samples=n,
    ))
            for case in implicit.SCALES for a in implicit.orders(case)
            for k in implicit.STRENGTHS for n in implicit.LENGTHS
            for e in audit.EVALUATIONS]
    assert len(rows) == 120 and audit.solvers_passed(rows, "float32")
    assert not audit.solvers_passed(rows[:-1], "float32")
    assert not audit.solvers_passed(rows[:-1] + [rows[0]], "float32")
    rows[-1] = audit.SolverRow("stable", replace(rows[-1].result, measurement_ok=False))
    assert not audit.solvers_passed(rows, "float32")
