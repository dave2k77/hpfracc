from __future__ import annotations

import io

import pytest

pytest.importorskip("jax")

from benchmarks.numerical.solver_validation.report import generate_rows, write_csv


def test_solver_validation_report_generates_expected_rows() -> None:
    rows = generate_rows(order=0.7, rate=-0.8, n_steps_values=(11, 21))
    assert len(rows) == 2
    assert {row.case for row in rows} == {"linear_caputo_fde"}


def test_solver_validation_report_linear_case_refines() -> None:
    rows = generate_rows(order=0.7, rate=-0.8, n_steps_values=(21, 81))
    assert rows[1].max_abs_error < rows[0].max_abs_error


def test_solver_validation_report_writes_csv() -> None:
    rows = generate_rows(order=0.7, rate=-0.8, n_steps_values=(11,))
    stream = io.StringIO()
    write_csv(rows, stream)
    output = stream.getvalue()
    assert output.startswith("solver,case,order,rate,n_steps,dt,max_abs_error")
    assert "predictor_corrector,linear_caputo_fde" in output
