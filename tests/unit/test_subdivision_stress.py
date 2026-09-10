"""Independent oracles and negative controls for the stress audit."""

import importlib
from dataclasses import replace
from math import gamma

import numpy as np
import pytest
from scipy.integrate import quad


def module():
    return importlib.import_module("benchmarks.numerical.subdivision_stress")


@pytest.mark.parametrize("case", ["nonlinear", "strong_feedback", "long_decay"])
def test_oracle_satisfies_caputo_equation(case):
    audit = module()
    problem = audit.CASES[case]
    assert audit.reference(case, np.asarray(0.0)) == 1.0
    for t in (0.1, 0.7, problem.end):
        # D^1/2 (1+t^2) from a continuous Caputo integral, not PECE weights.
        if case != "long_decay":
            derivative = quad(lambda s, t=t: 2 * s / np.sqrt(t - s), 0, t)[0]
            derivative /= gamma(0.5)
            value = audit.reference(case, np.asarray(t))
            assert float(audit.rhs(case, t, value)) == pytest.approx(
                derivative, rel=2e-10, abs=2e-10,
            )
        else:
            from scipy.special import erfcx

            # s=t*sin(theta)^2 removes both endpoint singularities from
            # integral y'(s)/sqrt(t-s) ds, with y'=y-1/sqrt(pi*s).
            derivative = quad(
                lambda theta, t=t: (
                    2 * np.sqrt(t) * np.sin(theta)
                    * erfcx(np.sqrt(t) * np.sin(theta)) - 2 / np.sqrt(np.pi)
                ), 0, np.pi / 2,
            )[0] / np.sqrt(np.pi)
            value = audit.reference(case, np.asarray(t))
            assert float(audit.rhs(case, t, value)) == pytest.approx(
                derivative, rel=2e-10, abs=2e-10,
            )


@pytest.mark.parametrize("mode,pece,rhs,padding", [
    ("step_doubling", 3, 5, 2), ("subdivision", 7, 13, 4),
])
def test_work_includes_rejections_and_padded_history(mode, pece, rhs, padding):
    audit = module()
    work = audit.work_counts(mode, 8, 2, 32)
    assert work == (pece * 10, 1 + rhs * 10, pece * 10 * (32 + padding))


def test_completed_and_failed_measurements(enable_x64):
    audit = module()
    row = audit.run_case("nonlinear", "subdivision", 1e-3, "float64", max_steps=256)
    assert row.completed and row.max_abs_error < 0.01
    assert row.endpoint_abs_error is not None
    failed = audit.run_case("long_decay", "subdivision", 1e-4, "float64", max_steps=2)
    assert not failed.completed
    assert failed.max_abs_error is None and failed.endpoint_abs_error is None
    assert failed.max_scaled_error is None


def test_work_matching_rejects_distant_and_failed_rows(enable_x64):
    audit = module()
    row = audit.run_case("nonlinear", "step_doubling", 1e-3, "float64", max_steps=128)
    assert row.completed
    other = replace(row, estimator="subdivision", weight_slots=row.weight_slots * 2)
    assert audit.match_work([row, other]) == []
    near = replace(other, weight_slots=int(row.weight_slots * 1.05))
    assert len(audit.match_work([row, near])) == 1
    assert audit.match_work([row, replace(near, completed=False)]) == []
    assert audit.match_work([row, replace(near, case="long_decay")]) == []
    assert audit.match_work([row, replace(near, dtype="float32")]) == []


def test_corrupt_endpoint_is_not_a_complete_measurement(monkeypatch, enable_x64):
    audit = module()
    real_simulate = audit.simulate

    def corrupt(**kwargs):
        result = real_simulate(**kwargs)
        n = int(result.solver_info.diagnostics["n_nodes"])
        return replace(result, ts=result.ts.at[n - 1].set(0.9))

    monkeypatch.setattr(audit, "simulate", corrupt)
    row = audit.run_case("nonlinear", "step_doubling", 1e-3, "float64", max_steps=128)
    assert not row.completed and row.max_abs_error is None


def test_checked_cli_rejects_empty_rows(monkeypatch, capsys):
    audit = module()
    monkeypatch.setattr(audit, "collect_rows", lambda **kwargs: [])
    assert audit.main(["--check"]) == 1
    assert "weight_slots" in capsys.readouterr().out


def test_long_decay_oracle_detects_wrong_rhs(monkeypatch):
    audit = module()
    monkeypatch.setattr(audit, "rhs", lambda case, t, state: -2 * state)
    with pytest.raises(AssertionError):
        test_oracle_satisfies_caputo_equation("long_decay")


def test_checked_cli_rejects_partial_or_duplicate_sweep(monkeypatch, capsys):
    audit = module()
    row = audit.run_case("nonlinear", "step_doubling", 1e-3, "float32", max_steps=128)
    assert row.completed
    # Synthetic metadata isolates the CLI's matrix gate from accuracy tests.
    rows = [replace(row, case=case, estimator=mode, rtol=tol)
            for case in audit.CASES for mode in audit.MODES for tol in audit.TOLERANCES]
    monkeypatch.setattr(audit, "collect_rows", lambda **kwargs: rows)
    assert audit.main(["--check", "--max-steps", "128"]) == 0
    partial = rows[:-1]
    monkeypatch.setattr(audit, "collect_rows", lambda **kwargs: partial)
    assert audit.main(["--check", "--max-steps", "128"]) == 1
    duplicate = partial + [rows[0]]
    monkeypatch.setattr(audit, "collect_rows", lambda **kwargs: duplicate)
    assert audit.main(["--check", "--max-steps", "128"]) == 1
    capsys.readouterr()


@pytest.mark.parametrize("mode", ["step_doubling", "subdivision"])
def test_work_model_matches_executed_calls(monkeypatch, mode):
    import jax

    audit = module()
    core = importlib.import_module("hpfracc.solvers.predictor_corrector")
    original_weights, original_rhs = core._nonuniform_pece_weights, audit.rhs
    weights, calls = [], []

    def record_weights(step, times, *args):
        jax.debug.callback(lambda n: weights.append(int(n)), times.size)
        return original_weights(step, times, *args)

    def record_rhs(*args):
        jax.debug.callback(lambda: calls.append(1))
        return original_rhs(*args)

    monkeypatch.setattr(core, "_nonuniform_pece_weights", record_weights)
    monkeypatch.setattr(audit, "rhs", record_rhs)
    row = audit.run_case("long_decay", mode, 1e-4, "float32", max_steps=2)
    jax.effects_barrier()
    assert row.n_rejected > 0
    assert len(weights) == row.pece_evaluations
    assert sum(weights) == row.weight_slots
    assert len(calls) == row.rhs_evaluations
