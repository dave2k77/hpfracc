"""Coverage and failure sensitivity for the Phase A report extension."""

from __future__ import annotations

import csv
import importlib
import io
from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from benchmarks.numerical import validation_summary as summary

HISTORY_CASES = {
    "soe_caputo_scaled_tail_float32", "soe_rl_constant_scaled_tail_float32",
    "soe_scaled_tail_recurrence_float32", "soe_scaled_tail_order_gradient_float32",
    "uniform_stable_weight_precision_float32", "uniform_stable_explicit_float32",
    "uniform_stable_implicit_float32",
    "bounded_forcing_quadratic_float32",
}

EXPECTED_CASES = {
    "graded_history_weights_float32",
    "graded_caputo_history_float32",
    "nonuniform_fde_refinement_float32",
    "implicit_nonlinear_converged",
    "implicit_nonlinear_failure_detected",
    "implicit_jit_failure_detected",
    "adaptive_completed",
    "adaptive_budget_failure_detected",
    "adaptive_nonfinite_failure_detected",
    "adaptive_step_size_failure_detected",
    "adaptive_integer_precision_failure_detected",
    "adaptive_jit_status",
}
COLUMNS = ["area", "case", "metric", "value", "passed", "details"]
BASELINE_CASES = (
    "caputo_constant", "caputo_power_law_refinement",
    "linear_caputo_fde_refinement", "caputo_operator_order",
    "riemann_liouville_order", "solver_endpoint_order",
    "finite_difference_checks", "basic_stability_checks",
)


def _module():
    return importlib.import_module("benchmarks.numerical.core_hardening")


@pytest.fixture(scope="module")
def hardening_rows():
    before = jax.config.jax_enable_x64
    rows = _module().generate_rows()
    assert jax.config.jax_enable_x64 == before
    return rows


def test_hardening_report_covers_named_cases(hardening_rows):
    assert {row.case for row in hardening_rows} == EXPECTED_CASES
    assert len(hardening_rows) == len(EXPECTED_CASES)
    assert all(row.passed for row in hardening_rows)
    assert all(row.details and row.metric for row in hardening_rows)


def test_hardening_report_preserves_csv_schema(hardening_rows):
    stream = io.StringIO()
    _module().write_csv(hardening_rows, stream)
    reader = csv.DictReader(io.StringIO(stream.getvalue()))
    assert reader.fieldnames == COLUMNS
    parsed = list(reader)
    assert len(parsed) == len(EXPECTED_CASES)
    assert all(row["passed"] == "True" for row in parsed)
    assert all(jnp.isfinite(float(row["value"])) for row in parsed)


def test_fixed_cases_remain_float32_with_x64_enabled(enable_x64, monkeypatch):
    report = _module()
    original = report.simulate

    def check_dtype(**kwargs):
        result = original(**kwargs)
        assert result.latent_state.dtype == jnp.float32
        return result

    monkeypatch.setattr(report, "simulate", check_dtype)
    assert all(row.passed for row in report.generate_rows())
    assert jax.config.jax_enable_x64 is True


def test_summary_includes_new_and_existing_cases():
    rows = summary.generate_rows(n_steps_values=(21, 41))
    names = [row.case for row in rows]
    assert len(names) == len(set(names))
    assert set(names) == EXPECTED_CASES | set(BASELINE_CASES) | HISTORY_CASES
    assert names[:len(BASELINE_CASES)] == list(BASELINE_CASES)
    assert all(row.passed for row in rows)


def test_negative_history_weight_fails_precision_row(monkeypatch):
    report = _module()
    original = report._nonuniform_pece_weights

    def corrupted(*args, **kwargs):
        p, c, d = original(*args, **kwargs)
        return p, c.at[0].set(-1.0), d

    monkeypatch.setattr(report, "_nonuniform_pece_weights", corrupted)
    rows = report.graded_precision_rows()
    row = next(r for r in rows if r.case == "graded_history_weights_float32")
    assert not row.passed


def test_nonrefining_trajectory_fails_refinement_row(monkeypatch):
    report = _module()
    original = report.simulate

    def corrupted(**kwargs):
        result = original(**kwargs)
        return replace(result, latent_state=jnp.full_like(result.latent_state, 100.0))

    monkeypatch.setattr(report, "simulate", corrupted)
    assert not report.nonuniform_refinement_row().passed


@pytest.mark.parametrize("corruption", ("nan", "wrong_endpoint", "negative_residual"))
def test_implicit_rows_reject_invalid_solution_or_residual(monkeypatch, corruption):
    report = _module()
    original = report.simulate

    def corrupted(**kwargs):
        result = original(**kwargs)
        if corruption == "nan":
            return replace(
                result, latent_state=jnp.full_like(result.latent_state, jnp.nan)
            )
        if corruption == "wrong_endpoint":
            return replace(result, latent_state=result.latent_state.at[-1].add(1.0))
        diag = dict(result.solver_info.diagnostics)
        diag["newton_max_scaled_residual"] = -100.0
        diag["newton_scaled_residuals"] = [-100.0]
        return replace(
            result, solver_info=replace(result.solver_info, diagnostics=diag)
        )

    monkeypatch.setattr(report, "simulate", corrupted)
    assert all(not row.passed for row in report.implicit_rows())


@pytest.mark.parametrize("family", ("implicit", "adaptive"))
def test_failure_detection_rows_reject_false_success(monkeypatch, family):
    report = _module()
    original = report.simulate

    def corrupted(**kwargs):
        result = original(**kwargs)
        diag = dict(result.solver_info.diagnostics)
        if family == "implicit":
            diag["newton_converged"] = True
            diag["newton_failure_count"] = 0
        else:
            diag["success"] = True
            diag["max_steps_reached"] = False
        return replace(
            result, solver_info=replace(result.solver_info, diagnostics=diag)
        )

    monkeypatch.setattr(report, "simulate", corrupted)
    rows = report.implicit_rows() if family == "implicit" else report.adaptive_rows()
    case = (
        "implicit_nonlinear_failure_detected"
        if family == "implicit" else "adaptive_budget_failure_detected"
    )
    assert not next(row for row in rows if row.case == case).passed


@pytest.mark.parametrize(
    "checked,passed,expected", ((True, False, 1), (True, True, 0), (False, False, 0))
)
@pytest.mark.parametrize("target", ("aggregate", "hardening"))
def test_summary_check_exit_code_and_csv(
    monkeypatch, capsys, checked, passed, expected, target
):
    report = summary if target == "aggregate" else _module()
    row = summary.ValidationSummaryRow("solver", "test", "error", 1.0, passed, "test")
    monkeypatch.setattr(report, "generate_rows", lambda **kwargs: [row])
    assert report.main(["--check"] if checked else []) == expected
    parsed = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert parsed[0]["passed"] == str(passed)


@pytest.mark.parametrize("target", ("aggregate", "hardening"))
def test_checked_empty_summary_is_not_success(monkeypatch, target):
    report = summary if target == "aggregate" else _module()
    monkeypatch.setattr(report, "generate_rows", lambda **kwargs: [])
    assert report.main(["--check"]) == 1
