"""Capture and verify the source-bound API-001 validation increment."""

import argparse
import ast
import gzip
import hashlib
import json
import subprocess
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path.cwd()
BASE = ROOT / "docs/validation/audits/api001-metadata-export-2026-11-11"
OLD = ROOT / "docs/validation/audits/num005-stable-precision-2026-11-10"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = [
        *(ROOT / "src/hpfracc").rglob("*.py"),
        *(ROOT / "benchmarks").rglob("*.py"),
        *(ROOT / "tests").rglob("*.py"),
        ROOT / "pyproject.toml", ROOT / "uv.lock",
    ]
    assert all(not path.is_symlink() for path in paths)
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def without_export(source):
    module = ast.parse(source)
    info = next(node for node in module.body
                if isinstance(node, ast.ClassDef) and node.name == "SolverInfo")
    info.body = [node for node in info.body
                 if not isinstance(node, ast.FunctionDef) or node.name != "to_dict"]
    return ast.dump(module, include_attributes=False)


def regression(source):
    module = ast.parse(source)
    node = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                and node.name == "test_strong_scalar_order_metadata_exports_plain_json")
    return ast.dump(node, include_attributes=False)


parser = argparse.ArgumentParser()
parser.add_argument("--capture", action="store_true")
args = parser.parse_args()
collected = [line for line in (BASE / "collection.txt").read_text().splitlines()
             if line.startswith("tests/") and "::" in line]
assert len(collected) == len(set(collected))
current = source_hashes()
previous = json.loads((OLD / "suite-source-inputs.json").read_text())["source_sha256"]
changed = "src/hpfracc/solvers/base.py"
comment_only = "tests/unit/test_direct_weight_precision.py"
new_test = "tests/unit/test_solver_info_export.py"
assert set(current) - set(previous) == {new_test}
assert set(previous) <= set(current)
assert all(current[name] == value for name, value in previous.items()
           if name not in {changed, comment_only})
with tarfile.open(OLD / "suite-sources.tar.gz", "r:gz") as old_tar:
    old_stream = old_tar.extractfile(comment_only)
    assert old_stream is not None
    old_test = old_stream.read()
assert hashlib.sha256(old_test).hexdigest() == previous[comment_only]
assert ast.dump(ast.parse(old_test)) == ast.dump(
    ast.parse((ROOT / comment_only).read_text())
)
before = gzip.decompress((BASE / "solver-base-before.py.txt.gz").read_bytes())
assert hashlib.sha256(before).hexdigest() == previous[changed]
assert without_export(before) == without_export(
    (ROOT / changed).read_text()
)
assert regression((BASE / "red-test-source.py.txt").read_text()) == regression(
    (ROOT / new_test).read_text()
)
if args.capture:
    assert not (BASE / "source-snapshot.json").exists()
    archive = BASE / "source-snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in current:
            tar.add(ROOT / name, arcname=name, recursive=False)
    snapshot = {
        "scope": "API-001 source capture; older numerical reports keep their identity",
        "baseline_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": current,
        "archive_sha256": digest(archive),
        "collected_tests": len(collected),
        "red_xml_raw_sha256": hashlib.sha256(gzip.decompress(
            (BASE / "red-cpu.xml.gz").read_bytes()
        )).hexdigest(),
        "red_xml_gzip_sha256": digest(BASE / "red-cpu.xml.gz"),
    }
    (BASE / "source-snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps({"source_inputs_captured": len(current),
                      "collected_tests": len(collected)}))
else:
    snapshot = json.loads((BASE / "source-snapshot.json").read_text())
    assert snapshot["source_sha256"] == current
    assert snapshot["archive_sha256"] == digest(BASE / "source-snapshot.tar.gz")
    with tarfile.open(BASE / "source-snapshot.tar.gz", "r:gz") as tar:
        assert set(tar.getnames()) == set(current)
        for member in tar:
            stream = tar.extractfile(member)
            assert member.isfile() and stream is not None
            assert hashlib.sha256(stream.read()).hexdigest() == current[member.name]
    red_compressed = (BASE / "red-cpu.xml.gz").read_bytes()
    red_raw = gzip.decompress(red_compressed)
    assert hashlib.sha256(red_compressed).hexdigest() == snapshot["red_xml_gzip_sha256"]
    assert hashlib.sha256(red_raw).hexdigest() == snapshot["red_xml_raw_sha256"]
    red = ET.fromstring(red_raw)
    red_suite = red.find("testsuite")
    assert red_suite is not None
    assert int(red_suite.attrib["tests"]) == int(red_suite.attrib["failures"]) == 2
    ci = json.loads((BASE / "num005-ci-readback.json").read_text())
    assert ci["headSha"] == "7a11d2d62546470119618a75d15640e69e940a8d"
    assert ci["status"] == "completed" and ci["conclusion"] == "success"
    expected = set()
    for node in collected:
        parts = node.split("::")
        module = parts[0].removesuffix(".py").replace("/", ".")
        expected.add((".".join([module, *parts[1:-1]]), parts[-1]))
    results = {}
    for backend in ("cpu", "cuda"):
        path = BASE / f"full-{backend}.xml"
        suite = ET.parse(path).getroot().find("testsuite")
        assert suite is not None
        cases = suite.findall("testcase")
        identities = {(case.attrib["classname"], case.attrib["name"])
                      for case in cases}
        assert identities == expected
        assert len(cases) == int(suite.attrib["tests"]) == len(collected)
        assert all(int(suite.attrib[k]) == 0 for k in ("failures", "errors", "skipped"))
        results[backend] = {"tests": len(cases), "xml_sha256": digest(path)}
    execution = json.loads((BASE / "executions.json").read_text())
    completed = execution["completed_runs"]
    assert len(completed) == 2
    assert {run["backend"] for run in completed} == {"cpu", "cuda"}
    for run in completed:
        assert run["status"] == "exited"
        assert type(run["exit_code"]) is int and run["exit_code"] == 0
        assert run["tests_passed"] == results[run["backend"]]["tests"]
    print(json.dumps({
        "scope": "Metadata repair/full tests, not new scientific acceptance",
        "source_inputs_verified": len(current),
        "only_existing_runtime_change": "SolverInfo.to_dict",
        "selector_test_comment_only_change_verified": True,
        "original_regression_ast_unchanged": True,
        "full_suites": results,
        "prior_num005_ci_verified": ci["url"],
    }, indent=2))
