"""Independent-reference checks for the adaptive calibration audit."""

from __future__ import annotations

import csv
import importlib
import io
from dataclasses import replace
from math import gamma

import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

MODULE = "benchmarks.numerical.adaptive_calibration"


def _audit():
    return importlib.import_module(MODULE)


def test_reference_oracles_match_independent_high_precision_values():
    audit = _audit()
    times = np.array([0.0, 0.01, 0.3, 1.0])
    with mp.workdps(60):
        decay = np.array([
            float(mp.exp(mp.mpf(float(t))) * mp.erfc(mp.sqrt(float(t))))
            for t in times
        ])
    assert np.allclose(audit.reference("half_order_decay", times, 0.5), decay,
                       rtol=2e-15, atol=2e-15)
    assert np.allclose(audit.reference("constant_forcing", times, 0.5),
                       times**0.5 / gamma(1.5))
    assert np.allclose(audit.reference("quadratic_forcing", times, 0.5),
                       2 * times**2.5 / gamma(3.5))
    assert np.array_equal(audit.reference("manufactured_feedback", times, 0.8),
                          times**2)


def test_first_step_pairs_match_closed_form_oracles(enable_x64):
    rows = _audit().first_step_rows(dtype="float64")
    assert len(rows) == 3
    for row in rows:
        a, h = row.order, row.step
        amplitude = h ** (a + 2) / gamma(a + 2)
        assert row.pair_matches_oracle
        assert row.coarse_value == pytest.approx(amplitude, rel=2e-13)
        assert row.fine_value == pytest.approx(
            amplitude * (0.5 + 2 ** (-a - 1)), rel=2e-13,
        )
        assert row.coarse_abs_error == pytest.approx(
            a * h ** (a + 2) / gamma(a + 3), rel=2e-13,
        )
        # This tests the documented indicator formula, not a requirement that
        # future controllers must continue accepting this under-calibrated step.
        assert row.underestimation_factor > 1


def test_first_step_oracle_detects_corrupted_pairs(monkeypatch):
    audit = _audit()
    original = audit.simulate

    def corrupted(**kwargs):
        result = original(**kwargs)
        if isinstance(kwargs["solver"], audit.NonUniformPredictorCorrector):
            return replace(result, latent_state=jnp.zeros_like(result.latent_state))
        return result

    monkeypatch.setattr(audit, "simulate", corrupted)
    assert not any(row.pair_matches_oracle for row in audit.first_step_rows())


def test_audit_measures_refinement_without_claiming_global_tolerance(enable_x64):
    audit = _audit()
    rows = [audit.run_case("quadratic_forcing", rtol=tol, dtype="float64")
            for tol in audit.TOLERANCES]
    assert all(row.completed for row in rows)
    errors = [row.max_abs_error for row in rows]
    assert all(
        left > right for left, right in zip(errors[:-1], errors[1:], strict=True)
    )
    assert all(row.dtype == "float64" for row in rows)
    assert all(row.max_scaled_error >= 0 for row in rows)


def test_corrupted_trajectory_changes_measured_error(monkeypatch):
    audit = _audit()
    original = audit.simulate

    def corrupted(**kwargs):
        result = original(**kwargs)
        return replace(result, latent_state=jnp.zeros_like(result.latent_state))

    monkeypatch.setattr(audit, "simulate", corrupted)
    row = audit.run_case("quadratic_forcing", rtol=1e-3, dtype="float32")
    assert row.completed  # completion is not an accuracy assertion
    assert row.max_abs_error == pytest.approx(2 / gamma(3.5), rel=1e-6)
    assert row.max_scaled_error > 100


def test_incomplete_run_does_not_report_full_span_error():
    row = _audit().run_case("half_order_decay", rtol=1e-5,
                           dtype="float32", max_steps=2)
    assert not row.completed
    assert row.max_abs_error is None
    assert row.endpoint_abs_error is None
    assert row.max_scaled_error is None


def test_nonfinite_trajectory_cannot_pass_audit(monkeypatch):
    audit = _audit()
    original = audit.simulate

    def corrupted(**kwargs):
        result = original(**kwargs)
        return replace(result, latent_state=jnp.full_like(result.latent_state, jnp.nan))

    monkeypatch.setattr(audit, "simulate", corrupted)
    row = audit.run_case("constant_forcing", rtol=1e-3, dtype="float32")
    assert not row.completed
    assert row.max_abs_error is None


def test_cli_checks_measurement_completeness_not_error_ratio(
    monkeypatch, capsys, tmp_path,
):
    audit = _audit()
    row = audit.run_case("quadratic_forcing", rtol=1e-3, dtype="float32")
    row = replace(row, max_scaled_error=100.0)
    monkeypatch.setattr(audit, "collect_rows", lambda **kwargs: [row])
    assert audit.main(["--check", "--output-dir", str(tmp_path)]) == 0
    records = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(records) == 1
    assert float(records[0]["max_scaled_error"]) == 100.0
    assert (tmp_path / "adaptive_calibration_float32.csv").is_file()
    assert (tmp_path / "adaptive_calibration_float32.json").is_file()
    monkeypatch.setattr(audit, "collect_rows", lambda **kwargs: [
        replace(row, completed=False, max_abs_error=None)
    ])
    assert audit.main(["--check"]) == 1
    capsys.readouterr()
    monkeypatch.setattr(audit, "collect_rows", lambda **kwargs: [])
    assert audit.main(["--check"]) == 1
