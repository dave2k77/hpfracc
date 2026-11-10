"""Scientific accuracy is separate from Newton and measurement integrity."""

import hashlib
import json
from dataclasses import replace

import jax
import numpy as np
import pytest

from benchmarks.numerical import implicit_validation as audit
from benchmarks.numerical import scientific_acceptance as policy


@pytest.fixture
def accurate_row(enable_x64):
    row = audit.run_case("cubic", 129, 8, 1., .5, "float64")
    assert row.measurement_ok and row.finite_trajectory and row.newton_converged
    return row


def test_converged_inaccurate_trajectory_is_not_accepted(enable_x64):
    row = audit.run_case("half_decay", 17, 8, 30., .5, "float64")
    assert row.measurement_ok and row.finite_trajectory and row.newton_converged
    assert row.max_dimensionless_error is not None
    assert row.independent_max_scaled_residual is not None
    assert row.max_dimensionless_error > .3
    assert row.independent_max_scaled_residual <= 1

    result = policy.assess(row, error_limit=1e-3)
    assert result.decision == "inaccurate"
    assert not result.accepted


def test_nonfinite_trajectory_is_rejected_not_an_accuracy_measurement(enable_x64):
    row = audit.run_case("cubic", 17, 1, 30., .3, "float64")
    assert row.measurement_ok and not row.finite_trajectory
    assert row.max_dimensionless_error is None

    result = policy.assess(row, error_limit=2e-5)
    assert result.decision == "nonfinite_trajectory"
    assert not result.accepted


@pytest.mark.parametrize("error", [None, float("nan"), float("inf"), -1.])
def test_invalid_error_measurements_fail_closed(accurate_row, error):
    row = replace(accurate_row, max_dimensionless_error=error)
    result = policy.assess(row, error_limit=2e-5)
    assert result.decision == "invalid_measurement"
    assert not result.accepted


def test_invalid_measurement_cannot_be_an_expected_numerical_failure(enable_x64):
    row = audit.run_case("cubic", 17, 1, 30., .3, "float64")
    result = policy.assess(replace(row, measurement_ok=False), error_limit=2e-5)
    assert result.decision == "invalid_measurement"
    assert not result.accepted


def test_finite_under_iterated_trajectory_is_not_accepted(enable_x64):
    row = audit.run_case("cubic", 17, 1, 1., .5, "float64")
    assert row.measurement_ok and row.finite_trajectory and not row.newton_converged
    result = policy.assess(row, error_limit=1.)
    assert result.decision == "unconverged_corrector"
    assert not result.accepted


def test_reported_convergence_does_not_override_independent_residual(accurate_row):
    row = replace(accurate_row, independent_max_scaled_residual=2.,
                  residual_flag_disagreements=1)
    result = policy.assess(row, error_limit=2e-5)
    assert result.decision == "independent_residual_failure"
    assert not result.accepted


@pytest.mark.parametrize("limit", [0., -1., float("nan"), float("inf"), None, True])
def test_error_budget_must_be_finite_and_positive(accurate_row, limit):
    with pytest.raises(ValueError, match="finite and positive"):
        policy.assess(accurate_row, error_limit=limit)


@pytest.mark.parametrize("field", [
    "reported_max_scaled_residual", "independent_max_scaled_residual",
    "endpoint_dimensionless_error", "minimum_dimensionless_state",
])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_missing_or_nonfinite_summary_metrics_are_not_accepted(
    accurate_row, field, value,
):
    result = policy.assess(replace(accurate_row, **{field: value}), error_limit=2e-5)
    assert result.decision == "invalid_measurement"
    assert not result.accepted


@pytest.mark.parametrize("changes", [
    {"reported_max_scaled_residual": -1.},
    {"independent_max_scaled_residual": -1.},
    {"endpoint_dimensionless_error": -1.},
    {"newton_failure_count": -1},
    {"newton_nonfinite_count": 129},
    {"residual_flag_disagreements": -1},
    {"newton_failure_count": 1},
    {"newton_converged": False},
    {"reported_max_scaled_residual": 2.},
    {"max_dimensionless_error": 0., "endpoint_dimensionless_error": 1.},
    {"component_max_abs_error": None},
    {"component_max_abs_error": ()},
    {"component_max_abs_error": (float("nan"),)},
    {"component_max_abs_error": (-1.,)},
    {"component_max_abs_error": (1.,)},
])
def test_contradictory_summary_metrics_fail_closed(accurate_row, changes):
    result = policy.assess(replace(accurate_row, **changes), error_limit=2e-5)
    assert result.decision == "invalid_measurement"
    assert not result.accepted


def test_rejection_reasons_do_not_hide_continuous_inaccuracy(enable_x64):
    row = audit.run_case("half_decay", 17, 8, 30., .5, "float64")
    row = replace(row, independent_max_scaled_residual=2.,
                  residual_flag_disagreements=1)
    result = policy.assess(row, error_limit=1e-3)
    assert set(result.reasons) == {"independent_residual_failure", "inaccurate"}
    assert not result.accepted


def test_nonfinite_control_needs_coherent_failed_diagnostics(enable_x64):
    row = audit.run_case("cubic", 17, 1, 30., .3, "float64")
    row = replace(row, newton_converged=True, newton_failure_count=0,
                  newton_nonfinite_count=0)
    assert policy.assess(row, error_limit=2e-5).decision == "invalid_measurement"


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("evaluation", ["direct", "stable"])
def test_manifest_freezes_case_configuration_and_budgets(dtype, evaluation):
    specs = policy.case_manifest(dtype, weight_evaluation=evaluation)
    assert [s.case_id for s in specs] == [
        "weak_decay", "cubic", "coupled", "scaled",
        "coarse_stiff_decay", "nonfinite_cubic",
    ]
    assert [(s.case, s.n_samples, s.newton_iterations, s.strength) for s in specs] == [
        ("half_decay", 257, 8, 1.), ("cubic", 129, 8, 1.),
        ("coupled", 129, 8, 1.), ("scaled", 129, 8, 1.),
        ("half_decay", 17, 8, 30.), ("cubic", 17, 1, 30.),
    ]
    assert all(s.dtype == dtype and s.weight_evaluation == evaluation for s in specs)
    assert all(s.newton_rtol == audit.RTOL[dtype] for s in specs)
    assert all(s.newton_atol == .01 * audit.RTOL[dtype] for s in specs)
    assert [s.role for s in specs] == ["required_success"] * 4 + [
        "expected_rejection",
    ] * 2
    assert [s.expected_reason for s in specs] == ["accepted"] * 4 + [
        "inaccurate", "nonfinite_trajectory",
    ]
    assert specs[0].error_limit == specs[4].error_limit == 1e-3
    assert all(s.error_limit == (1e-3 if dtype == "float32" else 2e-5)
               for s in (*specs[1:4], specs[5]))


@pytest.mark.parametrize("dtype,evaluation", [
    ("float16", "direct"), ("float32", "auto"),
])
def test_manifest_rejects_unsupported_precision_or_weight_mode(dtype, evaluation):
    with pytest.raises(ValueError):
        policy.case_manifest(dtype, weight_evaluation=evaluation)


def test_selected_float64_matrix_separates_success_and_expected_rejections(enable_x64):
    rows = policy.collect_rows("float64", weight_evaluation="direct")
    assert len(rows) == 6
    assert all(row.accepted and row.policy_passed for row in rows[:4])
    assert all(not row.accepted and row.policy_passed for row in rows[4:])
    assert policy.rows_passed(rows, "float64", weight_evaluation="direct")


@pytest.fixture
def matrix_rows(enable_x64):
    return policy.collect_rows("float64", weight_evaluation="direct")


@pytest.mark.parametrize("field", ["accepted", "policy_passed"])
def test_gate_rejects_truthy_integer_flags_not_true_booleans(matrix_rows, field):
    matrix_rows[0] = replace(matrix_rows[0], **{field: 1})
    assert not policy.rows_passed(matrix_rows, "float64")


@pytest.mark.parametrize("field", [
    "measurement_ok", "finite_trajectory", "newton_converged",
])
def test_measurement_flags_must_be_actual_booleans(accurate_row, field):
    row = replace(accurate_row, **{field: 1})
    assert policy.assess(row, error_limit=2e-5).decision == "invalid_measurement"


@pytest.mark.parametrize("changes", [
    {"case_id": "weak_decay"}, {"dtype": "float32"},
    {"weight_evaluation": "stable"}, {"order": .3}, {"strength": 2.},
    {"n_samples": 65}, {"newton_iterations": 1},
    {"strength": True}, {"n_samples": 129.},
    {"newton_rtol": 1e-5}, {"newton_atol": 1e-7},
    {"error_limit": 1.}, {"role": "expected_rejection"},
    {"expected_reason": "inaccurate"},
])
def test_gate_rejects_altered_policy_metadata(matrix_rows, changes):
    matrix_rows[1] = replace(
        matrix_rows[1], spec=replace(matrix_rows[1].spec, **changes),
    )
    assert not policy.rows_passed(matrix_rows, "float64")


@pytest.mark.parametrize("changes", [{"strength": True}, {"newton_iterations": 8.}])
def test_gate_does_not_coerce_measurement_configuration_types(matrix_rows, changes):
    row = matrix_rows[0]
    matrix_rows[0] = replace(row, measurement=replace(row.measurement, **changes))
    assert not policy.rows_passed(matrix_rows, "float64")


def test_gate_rejects_missing_duplicate_extra_and_forged_result_rows(matrix_rows):
    assert not policy.rows_passed([], "float64")
    assert not policy.rows_passed(matrix_rows[:-1], "float64")
    assert not policy.rows_passed(matrix_rows[:-1] + [matrix_rows[0]], "float64")
    assert not policy.rows_passed(matrix_rows + [matrix_rows[0]], "float64")
    control = replace(matrix_rows[4], accepted=True, policy_passed=True,
                      assessment=policy.Assessment(()))
    assert not policy.rows_passed(
        matrix_rows[:4] + [control, matrix_rows[5]], "float64",
    )
    row = matrix_rows[0]
    wrong = replace(row, measurement=replace(row.measurement, strength=30.))
    assert not policy.rows_passed([wrong, *matrix_rows[1:]], "float64")


@pytest.mark.parametrize("previous", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_collector_checks_effective_precision_and_restores_caller(previous, dtype):
    with jax.enable_x64(previous):
        rows = policy.collect_rows(dtype)
        assert bool(jax.config.read("jax_enable_x64")) is previous
        assert all(row.effective_dtypes == (dtype,) * 5 for row in rows)


def test_collector_restores_precision_on_execution_failure(monkeypatch):
    def failed(spec):
        raise RuntimeError("deliberate execution failure")

    monkeypatch.setattr(policy, "_execute", failed)
    with jax.enable_x64(True):
        with pytest.raises(RuntimeError, match="deliberate"):
            policy.collect_rows("float32")
        assert jax.config.read("jax_enable_x64")


def test_cli_saves_raw_replayable_evidence_and_effective_runtime(tmp_path, capsys):
    previous = bool(jax.config.read("jax_enable_x64"))
    code = policy.main(["--dtype", "float64", "--weight-evaluation", "stable",
                        "--check", "--output-dir", str(tmp_path)])
    assert code == 0
    assert bool(jax.config.read("jax_enable_x64")) is previous
    document = next(tmp_path.glob("*.json"))
    report = json.loads(document.read_text())
    assert report["protocol_version"] == "implicit-scientific-v1"
    assert report["runtime"]["jax_enable_x64"] is True
    assert report["runtime"]["backend"] == jax.default_backend()
    assert report["summary"]["scientifically_accepted"] == 4
    assert report["summary"]["expected_rejections_verified"] == 2
    assert len(report["rows"]) == len(report["raw_artifacts"]) == 6
    for source in ("src/hpfracc/ops/_special.py", "src/hpfracc/solvers/_weights.py"):
        assert source in report["runtime"]["source_sha256"]
    for record in report["rows"]:
        spec = record["spec"]
        artifact = report["raw_artifacts"][spec["case_id"]]
        path = tmp_path / artifact["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        with np.load(path, allow_pickle=False) as raw:
            diagnostics = {key: raw[key] for key in audit.FIELDS}
            measured = audit.summarize(
                (raw["times"], raw["states"], diagnostics), spec["case"],
                spec["dtype"], spec["order"], spec["strength"],
                spec["n_samples"], spec["newton_iterations"],
            )
            decision = policy.assess(measured, error_limit=spec["error_limit"])
            assert list(decision.reasons) == record["assessment"]["reasons"]
            assert raw["times"].dtype == raw["states"].dtype == np.float64
            assert int(raw["weight_evaluation_code"]) == 1
    csv_path = document.with_suffix(".csv")
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == report["csv_sha256"]
    output = capsys.readouterr().out
    assert "expected_rejection" in output and "policy_passed" in output


def test_cli_check_rejects_empty_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(policy, "collect_rows", lambda *args, **kwargs: [])
    assert policy.main(["--check", "--output-dir", str(tmp_path)]) == 1
    report = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert report["summary"]["configurations"] == 0
    assert report["summary"]["policy_passed"] is False


def test_direct_float32_gate_accepts_repaired_weights_without_switching_mode(
    tmp_path, capsys,
):
    assert policy.main(["--dtype", "float32", "--check",
                        "--output-dir", str(tmp_path)]) == 0
    report = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert all(s["weight_evaluation"] == "direct" for s in report["manifest"])
    weak = report["rows"][0]
    assert weak["measurement"]["newton_converged"] is True
    assert weak["measurement"]["max_dimensionless_error"] <= 1e-3
    assert weak["measurement"]["independent_max_scaled_residual"] <= 1
    assert weak["assessment"]["reasons"] == []
    assert weak["accepted"] is True and weak["policy_passed"] is True
