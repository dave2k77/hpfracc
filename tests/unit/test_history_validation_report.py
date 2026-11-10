"""Fixed history-extension checks must fail when numerical behavior is corrupted."""

import csv
import io
from dataclasses import replace

import jax
import numpy as np
import pytest

from benchmarks.numerical import history_validation as report

EXPECTED_CASES = {
    "soe_caputo_scaled_tail_float32", "soe_rl_constant_scaled_tail_float32",
    "soe_scaled_tail_recurrence_float32", "soe_scaled_tail_order_gradient_float32",
    "uniform_stable_weight_precision_float32", "uniform_stable_explicit_float32",
    "uniform_stable_implicit_float32",
    "bounded_forcing_quadratic_float32",
}


@pytest.fixture(scope="module")
def rows():
    before = jax.config.x64_enabled
    result = report.generate_rows()
    assert jax.config.x64_enabled == before
    return result


def test_named_cases_and_csv_contract(rows):
    assert len(rows) == len(EXPECTED_CASES)
    assert {r.case for r in rows} == EXPECTED_CASES
    assert all(r.passed and np.isfinite(r.value) for r in rows)
    stream = io.StringIO()
    report.write_csv(rows, stream)
    reader = csv.DictReader(io.StringIO(stream.getvalue()))
    assert reader.fieldnames == ["area", "case", "metric", "value", "passed", "details"]
    assert len(list(reader)) == len(rows)
    assert all("float32" in r.details for r in rows)


def test_fixed_precision_is_preserved_when_x64_enabled(enable_x64):
    assert all(r.passed for r in report.generate_rows())
    assert jax.config.x64_enabled


def test_corrupted_soe_values_fail_analytic_and_recurrence_checks(monkeypatch):
    original = report.caputo

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        return (replace(result, values=result.values + 1) if kwargs.get("return_info")
                else result + 1)

    monkeypatch.setattr(report, "caputo", corrupt)
    assert not report.soe_analytic_rows()[0].passed
    assert not report.soe_recurrence_row().passed


def test_rl_constant_reference_rejects_caputo_substitution(monkeypatch):
    monkeypatch.setattr(report, "riemann_liouville", report.caputo)
    assert not report.soe_analytic_rows()[1].passed


def test_order_gradient_check_rejects_stopped_derivative(monkeypatch):
    original = report.caputo
    monkeypatch.setattr(report, "caputo", lambda *a, **kw:
                        jax.lax.stop_gradient(original(*a, **kw)))
    assert not report.soe_gradient_row().passed


def test_stable_weight_check_rejects_corrupted_coefficients(monkeypatch):
    original = report._stable_pece_weights

    def corrupt(*args, **kwargs):
        predictor, corrector = original(*args, **kwargs)
        return predictor, .99 * corrector

    monkeypatch.setattr(report, "_stable_pece_weights", corrupt)
    assert not report.stable_weight_row().passed


def test_false_implicit_success_cannot_hide_wrong_state(monkeypatch):
    original = report.simulate

    def corrupt(**kwargs):
        result = original(**kwargs)
        return replace(result, latent_state=result.latent_state.at[-1, 0].add(.1))

    monkeypatch.setattr(report, "simulate", corrupt)
    assert not report.stable_implicit_row().passed


def test_checked_command_rejects_empty_or_failed_rows(monkeypatch, capsys):
    monkeypatch.setattr(report, "generate_rows", lambda: [])
    assert report.main(["--check"]) == 1
    row = report.HardeningRow("operator", "corrupt", "error", 1., False, "test")
    monkeypatch.setattr(report, "generate_rows", lambda: [row])
    assert report.main(["--check"]) == 1
    assert report.main([]) == 0
    capsys.readouterr()


def test_bound_check_rejects_zero_envelope(monkeypatch):
    original = report.simulate

    def corrupt(**kwargs):
        result = original(**kwargs)
        d = dict(result.solver_info.diagnostics)
        d["interpolation_error_bound"] = np.zeros_like(result.latent_state).tolist()
        return replace(result, solver_info=replace(result.solver_info, diagnostics=d))

    monkeypatch.setattr(report, "simulate", corrupt)
    assert not report.bounded_forcing_row().passed
