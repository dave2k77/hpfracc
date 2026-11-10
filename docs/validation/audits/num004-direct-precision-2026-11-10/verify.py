"""Replay archived implicit scientific-acceptance measurements on the host."""

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmarks.numerical import implicit_validation as audit
from benchmarks.numerical import scientific_acceptance as policy

ROOT = Path.cwd()
BASE = ROOT / "docs/validation/audits/num004-direct-precision-2026-11-10"
executions = json.loads((BASE / "executions.json").read_text())
assert len(executions) == 8
results = []
counts = Counter()
for execution in executions:
    backend = execution["backend"]
    dtype = execution["dtype"]
    mode = execution["weight_evaluation"]
    stem = BASE / f"scientific_acceptance_{backend}_{dtype}_{mode}"
    report = json.loads(stem.with_suffix(".json").read_text())
    expected = policy.case_manifest(dtype, weight_evaluation=mode)
    manifest = [asdict(spec) for spec in expected]
    assert report["manifest"] == manifest
    protocol = {"version": policy.PROTOCOL_VERSION, "cases": manifest}
    digest = hashlib.sha256(json.dumps(
        protocol, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    assert digest == report["manifest_sha256"]
    assert report["runtime"]["backend"] == backend
    assert report["runtime"]["jax_enable_x64"] is (dtype == "float64")
    sources = report["runtime"]["source_sha256"]
    for name, digest in sources.items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT)
        assert name.startswith(("src/", "benchmarks/", "tests/")) or name in (
            "pyproject.toml", "uv.lock",
        )
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    assert hashlib.sha256(stem.with_suffix(".csv").read_bytes()).hexdigest() == (
        report["csv_sha256"]
    )
    with stem.with_suffix(".csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(expected) == len(report["rows"])
    assert set(report["raw_artifacts"]) == {spec.case_id for spec in expected}
    reconstructed = []
    for spec, record in zip(expected, report["rows"], strict=True):
        assert record["spec"] == asdict(spec)
        artifact = report["raw_artifacts"][spec.case_id]
        path = (BASE / artifact["file"]).resolve()
        assert path.parent == BASE
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        with np.load(path, allow_pickle=False) as raw:
            assert raw["times"].dtype == raw["states"].dtype == np.dtype(dtype)
            assert raw["initial"].dtype == raw["order"].dtype == np.dtype(dtype)
            assert raw["strength"].dtype == np.dtype(dtype)
            assert float(raw["order"]) == spec.order
            assert float(raw["strength"]) == spec.strength
            assert int(raw["weight_evaluation_code"]) == int(mode == "stable")
            diagnostics = {key: raw[key] for key in audit.FIELDS}
            measured = audit.summarize(
                (raw["times"], raw["states"], diagnostics), spec.case, dtype,
                spec.order, spec.strength, spec.n_samples, spec.newton_iterations,
            )
            result = policy._evaluate(spec, measured, mode, (dtype,) * 5)
        assert json.loads(json.dumps(asdict(result))) == record
        reconstructed.append(result)
        counts["configurations"] += 1
        counts["raw_artifacts"] += 1
        if spec.role == "required_success":
            counts["required_success_cases"] += 1
            counts["scientifically_accepted"] += result.accepted
            counts["required_success_failures"] += not result.accepted
        else:
            assert not result.accepted
            counts["expected_rejections_verified"] += result.policy_passed
    passed = policy.rows_passed(reconstructed, dtype, weight_evaluation=mode)
    assert passed == report["summary"]["policy_passed"]
    assert execution["exit_code"] == int(not passed)
    counts["reports"] += 1
    counts["reports_passing"] += passed
    counts["reports_rejected"] += not passed
    results.append({
        "backend": backend, "dtype": dtype, "weight_evaluation": mode,
        "policy_passed": passed, "exit_code": execution["exit_code"],
        "source_inputs_verified": len(sources),
        "weak_decay_independent_scaled_residual": (
            reconstructed[0].measurement.independent_max_scaled_residual
        ),
    })
assert counts["configurations"] == sum(6 for _ in executions)
print(json.dumps({
    "scope": ("Artifact/source/host-reconstruction integrity; "
              "not universal scientific acceptance"),
    "summary": dict(counts), "runs": results,
}, indent=2))