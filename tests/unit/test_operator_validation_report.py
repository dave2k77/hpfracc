from __future__ import annotations

import io

import pytest

pytest.importorskip("jax")

from benchmarks.numerical.operator_validation.report import generate_rows, write_csv


def test_operator_validation_report_generates_expected_rows() -> None:
    rows = generate_rows(order=0.5, power=2.0, n_steps_values=(11, 21))
    assert len(rows) == 8
    assert {row.case for row in rows} == {
        "constant",
        "power_law",
        "rl_power_law",
        "rl_constant",
    }


def test_operator_validation_report_validates_rl_against_analytic_reference() -> None:
    # The RL rows must be genuine discretisation errors, not the old tautological
    # rl - gl == 0. The constant case is the decisive RL-vs-Caputo distinction.
    rows = generate_rows(order=0.5, power=2.0, n_steps_values=(201,))
    rl_power = next(row for row in rows if row.case == "rl_power_law")
    rl_constant = next(row for row in rows if row.case == "rl_constant")

    assert rl_power.reference == "analytic_riemann_liouville_power_law"
    assert 0.0 < rl_power.max_abs_error < 1e-1
    assert 0.0 < rl_constant.max_abs_error < 1e-1


def test_operator_validation_report_power_law_refines() -> None:
    rows = generate_rows(order=0.5, power=2.0, n_steps_values=(11, 41))
    power_rows = [row for row in rows if row.case == "power_law"]
    assert power_rows[1].max_abs_error < power_rows[0].max_abs_error


def test_operator_validation_report_writes_csv() -> None:
    rows = generate_rows(order=0.5, power=2.0, n_steps_values=(11,))
    stream = io.StringIO()
    write_csv(rows, stream)
    output = stream.getvalue()
    assert output.startswith("operator,case,order,power,n_steps,dt,max_abs_error")
    assert "caputo,power_law" in output

