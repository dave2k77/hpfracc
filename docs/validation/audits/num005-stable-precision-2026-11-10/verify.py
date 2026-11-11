"""Replay the NUM-005 frozen-policy reports without rerunning the solver."""

import hashlib
import itertools
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmarks.numerical import implicit_validation as audit
from benchmarks.numerical import scientific_acceptance as policy

ROOT = Path.cwd()
BASE = ROOT / "docs/validation/audits/num005-stable-precision-2026-11-10"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


executions = json.loads((BASE / "executions.json").read_text())
expected_keys = set(itertools.product(
    ("cpu", "gpu"), ("float32", "float64"), ("direct", "stable"),
))
keys = [(item["backend"], item["dtype"], item["weight_evaluation"])
        for item in executions]
assert len(keys) == len(expected_keys) and set(keys) == expected_keys
expected_sources = policy._source_hashes()
results = []
counts = Counter()
for execution in executions:
    backend, dtype, mode = (
        execution["backend"], execution["dtype"], execution["weight_evaluation"]
    )
    stem = BASE / f"scientific_acceptance_{backend}_{dtype}_{mode}"
    report = json.loads(stem.with_suffix(".json").read_text())
    expected = policy.case_manifest(dtype, weight_evaluation=mode)
    manifest = [asdict(spec) for spec in expected]
    assert canonical(report["manifest"]) == canonical(manifest)
    protocol = {"version": policy.PROTOCOL_VERSION, "cases": manifest}
    assert hashlib.sha256(canonical(protocol).encode()).hexdigest() == (
        report["manifest_sha256"]
    )
    assert report["protocol_version"] == policy.PROTOCOL_VERSION
    assert report["runtime"]["backend"] == backend
    assert report["runtime"]["jax_enable_x64"] is (dtype == "float64")
    assert report["runtime"]["source_sha256"] == expected_sources
    assert digest(stem.with_suffix(".csv")) == report["csv_sha256"]
    assert len(report["rows"]) == len(expected)
    assert set(report["raw_artifacts"]) == {spec.case_id for spec in expected}
    reconstructed = []
    for spec, record in zip(expected, report["rows"], strict=True):
        assert canonical(record["spec"]) == canonical(asdict(spec))
        artifact = report["raw_artifacts"][spec.case_id]
        path = (BASE / artifact["file"]).resolve()
        assert path.parent == BASE and digest(path) == artifact["sha256"]
        with np.load(path, allow_pickle=False) as raw:
            for key in ("times", "states", "initial", "order", "strength"):
                assert raw[key].dtype == np.dtype(dtype)
            assert float(raw["order"]) == spec.order
            assert float(raw["strength"]) == spec.strength
            assert int(raw["weight_evaluation_code"]) == int(mode == "stable")
            diagnostics = {key: raw[key] for key in audit.FIELDS}
            measured = audit.summarize(
                (raw["times"], raw["states"], diagnostics), spec.case, dtype,
                spec.order, spec.strength, spec.n_samples, spec.newton_iterations,
            )
            result = policy._evaluate(spec, measured, mode, (dtype,) * 5)
        assert canonical(asdict(result)) == canonical(record)
        reconstructed.append(result)
        counts["configurations"] += 1
        counts["raw_artifacts"] += 1
        if spec.role == "required_success":
            assert result.accepted
            counts["scientifically_accepted"] += 1
        else:
            assert not result.accepted and result.policy_passed
            counts["expected_rejections_verified"] += 1
    assert policy._csv_text(reconstructed).encode() == (
        stem.with_suffix(".csv").read_bytes()
    )
    passed = policy.rows_passed(reconstructed, dtype, weight_evaluation=mode)
    assert report["summary"] == {
        "configurations": len(expected), "expected_configurations": len(expected),
        "scientifically_accepted": 4, "expected_rejections_verified": 2,
        "policy_passed": True,
    }
    assert passed and execution["exit_code"] == 0
    counts["reports_passing"] += 1
    results.append({
        "backend": backend, "dtype": dtype, "weight_evaluation": mode,
        "policy_passed": passed, "exit_code": execution["exit_code"],
        "source_inputs_verified": len(expected_sources),
        "weak_decay_independent_scaled_residual": (
            reconstructed[0].measurement.independent_max_scaled_residual
        ),
    })
assert counts == {
    "reports_passing": 8, "configurations": 48, "raw_artifacts": 48,
    "scientifically_accepted": 32, "expected_rejections_verified": 16,
}
document = (ROOT / "docs/validation/uniform-weights.md").read_text()
heading = "## NUM-005 — history accumulation"
assert heading in document, "Missing current NUM-005 evidence section"
current = document.split(heading, 1)[1].split(
    "## Tests and compatibility boundary", 1,
)[0]
weak = current.split("### Frozen weak-decay policy case", 1)[1].split(
    "### Separate stiff algebraic regression", 1,
)[0]
for result in results:
    if result["dtype"] == "float32":
        label = "CPU" if result["backend"] == "cpu" else "CUDA"
        mode = result["weight_evaluation"]
        value = result["weak_decay_independent_scaled_residual"]
        assert f"| {label} | {mode} | {value} |" in weak
compatibility = json.loads((BASE / "compatibility/verification.json").read_text())
stiff = current.split("### Separate stiff algebraic regression", 1)[1]
for backend, label in (("cpu", "CPU"), ("cuda", "CUDA")):
    for mode in ("direct", "stable"):
        value = compatibility[backend]["properties"][
            f"{mode}_max_independent_scaled_residual"
        ]
        assert f"| {label} | {mode} | {value} |" in stiff
execution = json.loads((BASE / "full-suite-executions.json").read_text())
full = json.loads((BASE / "full-suite-verification.json").read_text())
snapshot = json.loads((BASE / "suite-source-inputs.json").read_text())
source_paths = [
    *(ROOT / "src/hpfracc").rglob("*.py"),
    *(ROOT / "benchmarks").rglob("*.py"),
    *(ROOT / "tests").rglob("*.py"),
    ROOT / "pyproject.toml", ROOT / "uv.lock",
]
assert snapshot["source_sha256"] == {
    str(path.relative_to(ROOT)): digest(path) for path in source_paths
}
assert digest(BASE / snapshot["archive"]) == snapshot["archive_sha256"]
assert {run["backend"] for run in execution["runs"]} == {"cpu", "cuda"}
assert len(execution["runs"]) == len(full["runs"]) == 2
full_records = {run["backend"]: run for run in full["runs"]}
identities = []
for run in execution["runs"]:
    backend = run["backend"]
    assert run["status"] == "completed" and run["exit_code"] == 0
    record = full_records[backend]
    path = BASE / f"full-{backend}.xml"
    assert digest(path) == record["xml_sha256"]
    suite = ET.parse(path).getroot().find("testsuite")
    assert suite is not None
    cases = suite.findall("testcase")
    nodes = {(case.attrib["classname"], case.attrib["name"]) for case in cases}
    assert len(nodes) == len(cases) == int(suite.attrib["tests"]) == record["tests"]
    assert len(cases) == execution["expected_default_tests"]
    assert all(int(suite.attrib[key]) == 0 for key in ("failures", "errors", "skipped"))
    properties = {item.attrib["name"]: item.attrib["value"]
                  for item in suite.findall("./properties/property")}
    for mode in ("direct", "stable"):
        key = f"{mode}_max_independent_scaled_residual"
        value = properties[key]
        assert record["full_suite_stiff_properties"][key] == value
        assert record["isolated_stiff_properties"][key] == (
            compatibility[backend]["properties"][key]
        )
        assert math.isfinite(float(value)) and 0 <= float(value) <= 1
        if backend == "cuda":
            assert value in stiff
    identities.append(nodes)
assert identities[0] == identities[1]
print(json.dumps({
    "scope": "Exact coverage/source/artifact/host replay; not universal accuracy",
    "summary": dict(counts), "runs": results,
    "current_documentation_values_verified": True,
    "full_suite_tests_verified": {key: run["tests"]
                                 for key, run in full_records.items()},
    "full_suite_source_inputs_verified": len(source_paths),
}, indent=2))
