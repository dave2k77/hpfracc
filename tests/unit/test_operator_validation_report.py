from __future__ import annotations

import io

import pytest

pytest.importorskip("jax")

from benchmarks.numerical.operator_validation.report import generate_rows, write_csv


def test_operator_validation_report_generates_expected_rows() -> None:
    rows = generate_rows(order=0.5, power=2.0, n_steps_values=(11, 21))
    assert len(rows) == 6
    assert {row.case for row in rows} == {
        "constant",
        "power_law",
        "gl_baseline_consistency",
    }


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

