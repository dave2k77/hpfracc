from __future__ import annotations

import io

import pytest

pytest.importorskip("jax")

from benchmarks.numerical.baseline import generate_rows as baseline_rows
from benchmarks.numerical.baseline import write_csv as write_baseline_csv
from benchmarks.numerical.gradient_checks import generate_rows as gradient_rows
from benchmarks.numerical.gradient_checks import write_csv as write_gradient_csv
from benchmarks.numerical.stability import generate_rows as stability_rows
from benchmarks.numerical.stability import write_csv as write_stability_csv
from benchmarks.numerical.validation_summary import generate_rows as summary_rows
from benchmarks.numerical.validation_summary import write_csv as write_summary_csv


def test_gradient_checks_are_close_to_finite_difference() -> None:
    rows = gradient_rows(step=1e-3)
    assert {row.target for row in rows} == {"caputo_operator", "caputo_solver"}
    assert all(row.abs_error < 2e-3 for row in rows)


def test_gradient_checks_write_csv() -> None:
    rows = gradient_rows(step=1e-3)
    stream = io.StringIO()
    write_gradient_csv(rows, stream)
    output = stream.getvalue()
    assert output.startswith("target,parameter,autodiff,finite_difference")
    assert "caputo_solver,rate" in output


def test_stability_checks_pass() -> None:
    rows = stability_rows(order=0.7, n_steps=21)
    assert len(rows) == 2
    assert all(row.passed for row in rows)


def test_stability_checks_write_csv() -> None:
    rows = stability_rows(order=0.7, n_steps=21)
    stream = io.StringIO()
    write_stability_csv(rows, stream)
    output = stream.getvalue()
    assert output.startswith("target,case,order,n_steps,metric,value,passed")
    assert "linear_decay_non_amplifying" in output


def test_baseline_benchmark_generates_context_rows() -> None:
    rows = baseline_rows(n_steps_values=(8,), state_dims=(1,), repeats=1)
    assert {row.target for row in rows} == {
        "caputo",
        "grunwald_letnikov",
        "predictor_corrector",
    }
    assert all(row.seconds >= 0.0 for row in rows)
    assert all(row.backend for row in rows)
    assert all(row.platform for row in rows)


def test_baseline_benchmark_writes_csv() -> None:
    rows = baseline_rows(n_steps_values=(8,), state_dims=(1,), repeats=1)
    stream = io.StringIO()
    write_baseline_csv(rows, stream)
    output = stream.getvalue()
    assert output.startswith("target,case,n_steps,state_dim,order,repeats,seconds")
    assert "predictor_corrector,linear_caputo_fde" in output


def test_validation_summary_reports_all_areas_passing() -> None:
    rows = summary_rows(order=0.7, n_steps_values=(21, 41), gradient_tolerance=2e-3)
    assert {row.area for row in rows} == {
        "operator",
        "solver",
        "gradient",
        "stability",
        "convergence",
    }
    assert all(row.passed for row in rows)


def test_validation_summary_writes_csv() -> None:
    rows = summary_rows(order=0.7, n_steps_values=(21, 41), gradient_tolerance=2e-3)
    stream = io.StringIO()
    write_summary_csv(rows, stream)
    output = stream.getvalue()
    assert output.startswith("area,case,metric,value,passed,details")
    assert "linear_caputo_fde_refinement" in output
