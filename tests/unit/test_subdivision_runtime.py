"""Synchronization, repeatability and negative controls for runtime evidence."""

import importlib
from dataclasses import replace

import numpy as np
import pytest


def module():
    return importlib.import_module("benchmarks.numerical.subdivision_runtime")


def test_timer_waits_for_all_outputs(monkeypatch):
    audit = module()
    events = []
    tree = {"times": [0., 1.], "state": (42, {"success": True})}

    def clock():
        events.append("clock")
        return float(len(events))

    def run(value):
        events.append("execute")
        return value

    def block(value):
        assert value is tree
        events.append("sync")
        return value

    monkeypatch.setattr(audit.time, "perf_counter", clock)
    monkeypatch.setattr(audit.jax, "block_until_ready", block)
    output, seconds = audit.timed_call(run, (tree,))
    assert output is tree and seconds == 3
    assert events == ["clock", "execute", "sync", "clock"]


def test_compiled_measurements_are_repeatable(enable_x64):
    audit = module()
    rows = audit.collect_rows(dtype="float64", max_steps=128, repeats=3,
                              cases=("nonlinear",), tolerances=(1e-2,))
    assert len(rows) == 2
    for row in rows:
        assert row.measurement.completed
        assert row.stable_repeats
        assert len(row.samples_seconds) == 3
        assert all(np.isfinite(x) and x > 0 for x in row.samples_seconds)
        assert row.compile_seconds > 0 and row.warmup_seconds > 0
        assert row.measurement.max_abs_error is not None


def test_runtime_pairs_require_complete_compatible_nearby_rows():
    audit = module()
    rows = audit.collect_rows(dtype="float32", max_steps=64, repeats=1,
                              cases=("nonlinear",), tolerances=(1e-2,))
    a, b = rows
    a = replace(a, samples_seconds=(1.,))
    b = replace(b, samples_seconds=(1.1,))
    assert len(audit.match_runtime([a, b])) == 1
    assert audit.match_runtime([a, replace(b, samples_seconds=(2.,))]) == []
    failed = replace(b, measurement=replace(b.measurement, completed=False))
    assert audit.match_runtime([a, failed]) == []
    assert audit.match_runtime([a, replace(b, stable_repeats=False)]) == []
    assert audit.match_runtime([a, replace(b, samples_seconds=(float("nan"),))]) == []
    other = replace(b, measurement=replace(b.measurement, dtype="float64"))
    assert audit.match_runtime([a, other]) == []


def test_incomplete_solve_retains_timing_but_not_accuracy():
    audit = module()
    rows = audit.collect_rows(dtype="float32", max_steps=2, repeats=1,
                              cases=("long_decay",), tolerances=(1e-4,))
    assert len(rows) == 2
    for row in rows:
        assert not row.measurement.completed
        assert row.measurement.max_abs_error is None
        assert row.samples_seconds[0] > 0
    assert audit.match_runtime(rows) == []


@pytest.mark.parametrize("repeats", [0, -1])
def test_invalid_repeat_count(repeats):
    with pytest.raises(ValueError, match="repeats"):
        module().collect_rows(dtype="float32", max_steps=2, repeats=repeats)


def test_checked_cli_rejects_empty_sweep(monkeypatch, capsys):
    audit = module()
    monkeypatch.setattr(audit, "collect_rows", lambda **kwargs: [])
    assert audit.main(["--check"]) == 1
    assert "median_seconds" in capsys.readouterr().out


def test_compiled_inputs_remain_dynamic():
    audit = module()
    compiled, args, output, _, _ = audit.prepare(
        "nonlinear", "step_doubling", 1e-2, "float32", 64,
    )
    changed, _ = audit.timed_call(compiled, (args[0] + 0.1, args[1]))
    assert float(changed[1][0]) == pytest.approx(1.1)
    assert not audit._same_output(changed, audit._host_tree(output))


def test_matrix_gate_and_raw_sample_serialization(monkeypatch, capsys, tmp_path):
    import csv
    import hashlib
    import json

    audit = module()
    row = audit.collect_rows(dtype="float32", max_steps=64, repeats=1,
                             cases=("nonlinear",), tolerances=(1e-2,))[0]
    # Synthetic metadata tests matrix shape independently of the timing experiment.
    rows = [replace(row, measurement=replace(
        row.measurement, case=case, estimator=mode, rtol=tol,
    )) for case in audit.stress.CASES for mode in audit.stress.MODES
        for tol in audit.stress.TOLERANCES]
    def check(items):
        return audit.checked(items, dtype="float32", max_steps=64, repeats=1)

    assert check(rows)
    assert not check(rows[:-1])
    assert not check(rows[:-1] + [rows[0]])
    assert not check([replace(row, samples_seconds=())] + rows[1:])
    assert not check([replace(row, samples_seconds=(1., 2.))] + rows[1:])
    assert not check([replace(row, stable_repeats=False)] + rows[1:])
    failed = replace(row, measurement=replace(row.measurement, completed=False))
    assert not check([failed] + rows[1:])
    monkeypatch.setattr(audit, "collect_rows", lambda **kwargs: [row])
    assert audit.main(["--output-dir", str(tmp_path), "--check"]) == 1
    stem = tmp_path / "subdivision_runtime_float32"
    content = stem.with_suffix(".csv").read_bytes()
    data = json.loads(stem.with_suffix(".json").read_text())
    assert data["csv_sha256"] == hashlib.sha256(content).hexdigest()
    assert len(list(csv.DictReader(content.decode().splitlines()))) == 1
    assert data["rows"][0]["samples_seconds"] == list(row.samples_seconds)
    capsys.readouterr()


@pytest.mark.parametrize("args", [["--repeats", "0"], ["--max-steps", "0"]])
def test_cli_rejects_invalid_budgets(args):
    with pytest.raises(SystemExit) as exc:
        module().main(args)
    assert exc.value.code == 2


def test_cli_requires_explicit_float64_support():
    audit = module()
    previous = audit.jax.config.x64_enabled
    try:
        audit.jax.config.update("jax_enable_x64", False)
        with pytest.raises(SystemExit) as exc:
            audit.main(["--dtype", "float64"])
        assert exc.value.code == 2
    finally:
        audit.jax.config.update("jax_enable_x64", previous)


def test_repeat_equality_is_elementwise_not_bitwise():
    audit = module()
    assert audit._same_output(np.array([-0.]), (np.array([0.]),))
    assert not audit._same_output(np.array([1.]), (np.array([0.]),))


def test_dense_grid_is_fixed_and_preserves_coarse_tolerances():
    audit = module()
    assert audit.sweep_tolerances("coarse") == audit.stress.TOLERANCES
    dense = audit.sweep_tolerances("dense")
    assert dense == (1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1.5e-3, 1e-3,
                     7e-4, 5e-4, 3e-4, 2e-4, 1.5e-4, 1e-4)
    assert len(dense) == len(set(dense))
    assert set(audit.stress.TOLERANCES) <= set(dense)
    with pytest.raises(ValueError, match="sweep"):
        audit.sweep_tolerances("unknown")


def test_dense_matrix_gate_does_not_accept_coarse_subset():
    audit = module()
    row = audit.collect_rows(dtype="float32", max_steps=64, repeats=1,
                             cases=("nonlinear",), tolerances=(1e-2,))[0]
    dense = audit.sweep_tolerances("dense")
    rows = [replace(row, measurement=replace(
        row.measurement, case=case, estimator=mode, rtol=tol,
    )) for case in audit.stress.CASES for mode in audit.stress.MODES for tol in dense]
    assert audit.checked(rows, dtype="float32", max_steps=64, repeats=1,
                         tolerances=dense)
    assert not audit.checked(rows[:-1], dtype="float32", max_steps=64, repeats=1,
                             tolerances=dense)
    coarse = [r for r in rows if r.measurement.rtol in audit.stress.TOLERANCES]
    assert not audit.checked(coarse, dtype="float32", max_steps=64, repeats=1,
                             tolerances=dense)


def test_cli_wires_dense_grid_into_collection_and_provenance(
    monkeypatch, tmp_path, capsys,
):
    import json

    audit = module()
    captured = {}

    def collect(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(audit, "collect_rows", collect)
    assert audit.main(["--sweep", "dense", "--check",
                       "--output-dir", str(tmp_path)]) == 1
    assert captured["tolerances"] == audit.sweep_tolerances("dense")
    data = json.loads((tmp_path / "subdivision_runtime_float32.json").read_text())
    # The exported experiment configuration must identify the chosen sweep.
    assert data["sweep"] == "dense"
    assert data["tolerances"] == list(audit.sweep_tolerances("dense"))
    capsys.readouterr()
