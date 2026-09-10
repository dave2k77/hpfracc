"""Independent startup scaling, reference, and audit-sensitivity checks."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from math import erfc, exp, gamma, sqrt

import jax.numpy as jnp
import pytest


def _audit():
    return importlib.import_module("benchmarks.numerical.adaptive_startup")


def test_power_coefficients_recover_quadratic_identity():
    a = 0.5
    p, coarse, fine, exact = _audit().leading_coefficients("power_quadratic", a)
    assert p == 2.5
    assert coarse == pytest.approx(1 / gamma(a + 2))
    assert fine == pytest.approx(coarse * (0.5 + 2 ** (-a - 1)))
    assert exact == pytest.approx(2 / gamma(a + 3))


def test_autonomous_leading_term_tracks_jacobian_times_initial_drift():
    audit = _audit()
    linear = audit.leading_coefficients("linear_decay", 0.5)
    nonlinear = audit.leading_coefficients("nonlinear_decay", 0.5)
    assert linear[0] == nonlinear[0] == 1.0
    assert nonlinear[1:] == pytest.approx(tuple(2 * value for value in linear[1:]))


def test_series_oracles_match_closed_form_controls():
    audit = _audit()
    value, agreement = audit.reference("linear_decay", 0.5, 0.0001)
    assert value == pytest.approx(exp(0.0001) * erfc(sqrt(0.0001)), rel=2e-15)
    assert agreement < 1e-50
    value, agreement = audit.reference("nonlinear_decay", 1.0, 0.02)
    assert value == pytest.approx(1 / 1.02, rel=2e-15)
    assert agreement < 1e-50


@pytest.mark.parametrize("case", ["power_weak", "power_quadratic",
                                  "linear_decay", "nonlinear_decay"])
def test_measured_startup_scaling_matches_predicted_regime(case, enable_x64):
    result = _audit().run_case(case, 0.5)
    assert result.summary.passed
    assert len(result.measurements) == 4
    assert result.summary.observed_power == pytest.approx(
        result.summary.expected_power, abs=0.05,
    )


def test_corrupted_coarse_and_fine_outputs_cannot_pass(monkeypatch, enable_x64):
    audit = _audit()
    original = audit.simulate

    def corrupted(**kwargs):
        result = original(**kwargs)
        return replace(result, latent_state=jnp.zeros_like(result.latent_state))

    monkeypatch.setattr(audit, "simulate", corrupted)
    assert not audit.run_case("power_weak", 0.5).summary.passed


def test_nan_outputs_cannot_pass(monkeypatch, enable_x64):
    audit = _audit()
    original = audit.simulate

    def corrupted(**kwargs):
        result = original(**kwargs)
        return replace(result, latent_state=jnp.full_like(result.latent_state, jnp.nan))

    monkeypatch.setattr(audit, "simulate", corrupted)
    assert not audit.run_case("linear_decay", 0.5).summary.passed


def test_cubic_blind_spot_matches_independent_integrals(enable_x64):
    rows = _audit().blind_spot_rows()
    assert len(rows) == 3
    for row in rows:
        a = row["order"]
        assert row["pair_matches_oracle"]
        assert row["coarse"] == pytest.approx(1 / (3 * gamma(a + 2)))
        assert row["fine"] == pytest.approx(row["coarse"], abs=2e-15)
        error = a * (a - 1) / (3 * (a + 2) * (a + 3) * gamma(a + 2))
        assert row["signed_error"] == pytest.approx(error, rel=2e-12)
        # The mathematical aliasing identity is pinned, not future acceptance.


def test_checked_cli_and_raw_evidence(monkeypatch, capsys, tmp_path, enable_x64):
    audit = _audit()
    result = audit.run_case("power_weak", 0.5)
    blind = audit.blind_spot_rows()
    monkeypatch.setattr(audit, "collect_results", lambda: [result])
    monkeypatch.setattr(audit, "blind_spot_rows", lambda: blind)
    assert audit.main(["--check", "--output-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    context = json.loads((tmp_path / "adaptive_startup_float64.json").read_text())
    assert len(context["measurements"][0]["samples"]) == 4
    assert len(context["blind_spots"]) == 3
    monkeypatch.setattr(audit, "collect_results", lambda: [
        replace(result, summary=replace(result.summary, passed=False))
    ])
    assert audit.main(["--check"]) == 1
    capsys.readouterr()
    monkeypatch.setattr(audit, "collect_results", lambda: [])
    assert audit.main(["--check"]) == 1
