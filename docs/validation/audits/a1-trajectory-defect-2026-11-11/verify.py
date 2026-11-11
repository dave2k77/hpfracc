"""Replay preserved A1 evidence; this verifies records, not numerical certification."""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.dont_write_bytecode = True
from completion_integrity import fields, validate_completion  # noqa: E402

ROOT = Path.cwd()
BASE = ROOT / "docs/validation/audits/a1-trajectory-defect-2026-11-11"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def archived(name, *, raw=None):
    with tarfile.open(
        BASE / name if raw is None else None, "r:gz",
        fileobj=io.BytesIO(raw) if raw is not None else None,
    ) as archive:
        result = {}
        for member in archive:
            path = Path(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in result
            stream = archive.extractfile(member)
            assert stream is not None
            result[member.name] = stream.read()
        return result


def xml_counts(raw):
    tree = ET.fromstring(raw)
    suites = list(tree.iter("testsuite"))
    assert len(suites) == 1
    cases = list(tree.iter("testcase"))
    counts = {"tests": len(cases),
            "failures": sum(c.find("failure") is not None for c in cases),
            "errors": sum(c.find("error") is not None for c in cases),
            "skipped": sum(c.find("skipped") is not None for c in cases)}
    assert all(int(suites[0].attrib[key]) == value for key, value in counts.items())
    return counts


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--current-source", action="store_true",
                    help="also verify actual local source inputs and local exit files")
args = parser.parse_args()
snapshot = json.loads((BASE / "source-snapshot.json").read_text())
source = archived("source-snapshot.tar.gz")
assert set(source) == set(snapshot["paths"])
assert len(source) == snapshot["count"]
assert all(digest(raw) == snapshot["paths"][name] for name, raw in source.items())
if args.current_source:
    names = subprocess.check_output([
        "git", "ls-files", "--", "src", "tests", "benchmarks",
        "pyproject.toml", "uv.lock",
    ], text=True).splitlines()
    names.append(str((BASE / "frozen-fixtures.json").relative_to(ROOT)))
    assert set(names) == set(source), "Current declared source set changed"
    assert all(not (ROOT / name).is_symlink()
               and digest((ROOT / name).read_bytes()) == snapshot["paths"][name]
               for name in source), "Current input bytes changed"
    subprocess.run(["git", "diff", "--exit-code", snapshot["baseline_commit"],
                    "--", "src", "pyproject.toml", "uv.lock"], check=True,
                   capture_output=True)

development = archived("development-evidence.tar.gz")
index = json.loads((BASE / "development-index.json").read_text())
assert set(development) == set(index)
for name, raw in development.items():
    assert digest(raw) == index[name]["sha256"], name
    if name.endswith(".xml"):
        counts = xml_counts(raw)
        assert all(counts[key] == index[name][key]
                   for key in ("tests", "failures", "errors")), name
features = ((1, 1, 1), (2, 6, 7), (3, 28, 35), (4, 1, 36), (5, 4, 40))
for feature, failed, green in features:
    red = xml_counts(development[f"feature{feature}-red.xml"])
    passed = xml_counts(development[f"feature{feature}-green.xml"])
    assert red == dict(tests=failed, failures=failed, errors=0, skipped=0)
    assert passed == dict(tests=green, failures=0, errors=0, skipped=0)
    assert f"feature{feature}-red-source.py.txt" in development
assert xml_counts(development["compatibility-cpu.xml"])["failures"] == 2

gamma = archived("gamma-repair-evidence.tar.gz")
gamma_index = json.loads((BASE / "gamma-repair-index.json").read_text())
assert set(gamma) == set(gamma_index)
assert all(digest(raw) == gamma_index[name]["sha256"] for name, raw in gamma.items())
assert xml_counts(gamma["feature6-red.xml"]) == dict(
    tests=9, failures=9, errors=0, skipped=0,
)
assert xml_counts(gamma["feature6-green.xml"]) == dict(
    tests=64, failures=0, errors=0, skipped=0,
)
before = archived("source-snapshot-before-gamma-repair.tar.gz")
before_record = json.loads(
    (BASE / "source-snapshot-before-gamma-repair.json").read_text()
)
new_arithmetic_test = "tests/unit/test_volterra_arithmetic.py"
assert set(before) == set(before_record["paths"])
assert set(source) == set(before) | {new_arithmetic_test}
assert all(digest(raw) == before_record["paths"][name] for name, raw in before.items())
assert {name for name in before if source[name] != before[name]} == {
    "benchmarks/numerical/_volterra_defect.py", "tests/unit/test_volterra_defect.py",
}

panel = archived("panel-repair-evidence.tar.gz")
panel_index = json.loads((BASE / "panel-repair-index.json").read_text())
assert set(panel) == set(panel_index)
assert all(digest(raw) == panel_index[name]["sha256"] for name, raw in panel.items())
assert xml_counts(panel["feature7-red.xml"]) == dict(
    tests=5, failures=5, errors=0, skipped=0,
)
assert xml_counts(panel["feature7-green.xml"]) == dict(
    tests=69, failures=0, errors=0, skipped=0,
)
assert panel["panel-design-initial-exit.txt"].strip() == b"1"
for name in ("panel-targeted-cpu.xml", "panel-targeted-cuda.xml"):
    assert xml_counts(panel[name]) == dict(tests=81, failures=0, errors=0, skipped=0)

gamma_source = archived("source-snapshot-before-panel-repair.tar.gz")
gamma_record = json.loads(
    (BASE / "source-snapshot-before-panel-repair.json").read_text()
)
assert set(gamma_source) == set(gamma_record["paths"])
assert set(source) == set(gamma_source) | {new_arithmetic_test}
assert all(digest(raw) == gamma_record["paths"][name]
           for name, raw in gamma_source.items())
assert {name for name in gamma_source if source[name] != gamma_source[name]} == {
    "benchmarks/numerical/_volterra_defect.py", "tests/unit/test_volterra_defect.py",
}
rejected = archived("gamma-code-review-failed.tar.gz")
rejected_index = json.loads(rejected["evidence-index.json"])
assert set(rejected) == {row["path"] for row in rejected_index["files"]} | {
    "evidence-index.json",
}
assert all(digest(rejected[row["path"]]) == row["sha256"]
           and len(rejected[row["path"]]) == row["bytes"]
           for row in rejected_index["files"])
rejected_verdict = json.loads((BASE / "gamma-review-failed.json").read_text())
assert rejected_verdict["passed"] is False
assert len(rejected_verdict["logic_errors"]) == 2
old_runs = json.loads((BASE / "executions-before-panel.json").read_text())["full_runs"]
assert len(old_runs) == 2 and {row["backend"] for row in old_runs} == {"cpu", "cuda"}
for run in old_runs:
    assert type(run["exit_code"]) is int and run["exit_code"] == 0
    old_raw = (BASE / f"full-before-panel-{run['backend']}.xml").read_bytes()
    assert xml_counts(old_raw) == dict(tests=964, failures=0, errors=0, skipped=0)

pre_arithmetic = archived("pre-arithmetic-checkpoint.tar.gz")
layout = json.loads((BASE / "pre-arithmetic-layout.json").read_text())
historical_checksums = json.loads(pre_arithmetic["checksums.json"])
assert len(historical_checksums) == layout["old_protected_files_verified"] == 54
for name, expected_digest in historical_checksums.items():
    location = layout["historical_manifest_layout"][name]["location"]
    if "::" in location:
        archive_name, member_name = location.split("::", 1)
        assert archive_name == "pre-arithmetic-checkpoint.tar.gz"
        raw = pre_arithmetic[member_name]
    else:
        path = Path(location)
        assert not path.is_absolute() and ".." not in path.parts
        raw = (BASE / path).read_bytes()
    assert digest(raw) == expected_digest
pre_source = archived(None, raw=pre_arithmetic["source-snapshot.tar.gz"])
pre_record = json.loads(pre_arithmetic["source-snapshot.json"])
assert set(pre_source) == set(pre_record["paths"])
assert set(source) == set(pre_source) | {new_arithmetic_test}
assert all(digest(raw) == pre_record["paths"][name]
           for name, raw in pre_source.items())
assert {name for name in pre_source if source[name] != pre_source[name]} == {
    "benchmarks/numerical/_volterra_defect.py",
}
for backend in ("cpu", "cuda"):
    assert xml_counts(pre_arithmetic[f"full-{backend}.xml"]) == dict(
        tests=981, failures=0, errors=0, skipped=0,
    )
final_rejected = archived("final-code-review-rejected.tar.gz")
final_rejected_index = json.loads(final_rejected["hash-index.json"])
assert set(final_rejected) == set(final_rejected_index["files"]) | {"hash-index.json"}
assert all(digest(final_rejected[name]) == row["sha256"]
           and len(final_rejected[name]) == row["bytes"]
           for name, row in final_rejected_index["files"].items())
final_rejected_verdict = json.loads(final_rejected["review.json"])
assert final_rejected_verdict["passed"] is False
assert len(final_rejected_verdict["logic_errors"]) == 3

contract_review = archived("arithmetic-contract-review.tar.gz")
contract_record = json.loads((BASE / "arithmetic-contract-review.json").read_text())
assert contract_record["passed"] is True
assert contract_record["source_artifact_approved"] is False
assert digest(contract_review["reviewed-arithmetic-contract.md"]) == (
    contract_record["contract_sha256"]
)
for row in json.loads(contract_review["evidence-manifest.json"]):
    name = Path(row["path"]).name
    assert digest(contract_review[name]) == row["sha256"]
    assert len(contract_review[name]) == row["bytes"]
test_design = archived("arithmetic-test-design.tar.gz")
design_index = json.loads((BASE / "arithmetic-test-design-index.json").read_text())
assert set(test_design) == set(design_index)
assert all(digest(raw) == design_index[name]["sha256"]
           and len(raw) == design_index[name]["bytes"]
           for name, raw in test_design.items())
assert xml_counts(test_design["v2/baseline.xml"]) == dict(
    tests=36, failures=13, errors=0, skipped=0,
)
red_arithmetic = archived("arithmetic-regression-red.tar.gz")
red_index = json.loads((BASE / "arithmetic-regression-red-index.json").read_text())
assert set(red_arithmetic) == set(red_index)
assert all(digest(raw) == red_index[name]["sha256"]
           for name, raw in red_arithmetic.items())
assert xml_counts(red_arithmetic["red.xml"]) == dict(
    tests=36, failures=13, errors=0, skipped=0,
)
assert json.loads(red_arithmetic["red-result.json"])["actual_exit_code"] == 1

green_arithmetic = archived("arithmetic-repair-green.tar.gz")
green_index = json.loads((BASE / "arithmetic-repair-green-index.json").read_text())
assert set(green_arithmetic) == set(green_index)
assert all(digest(raw) == green_index[name]["sha256"]
           and len(raw) == green_index[name]["bytes"]
           for name, raw in green_arithmetic.items())
green_record = json.loads(green_arithmetic["green-result.json"])
assert green_record["actual_exit_code"] == 0
assert green_record["post_run_source_unchanged"] is True
assert green_record["source_inputs"] == snapshot["paths"]
assert xml_counts(green_arithmetic["green.xml"]) == dict(
    tests=117, failures=0, errors=0, skipped=0,
)
for backend in ("cpu", "cuda"):
    assert xml_counts((BASE / f"targeted-{backend}.xml").read_bytes()) == dict(
        tests=117, failures=0, errors=0, skipped=0,
    )

code_review = archived("arithmetic-code-review.tar.gz")
code_review_index = json.loads(code_review["hash-index.json"])
assert set(code_review) == set(code_review_index["files"]) | {"hash-index.json"}
assert all(digest(code_review[name]) == row["sha256"]
           and len(code_review[name]) == row["bytes"]
           for name, row in code_review_index["files"].items())
assert code_review["verdict.json"] == (
    BASE / "arithmetic-code-review.json"
).read_bytes()
code_verdict = json.loads(code_review["verdict.json"])
assert code_verdict["passed"] is True
assert not code_verdict["security_concerns"] and not code_verdict["logic_errors"]
assert code_verdict["reviewed_tree"] == snapshot["source_tree"]
assert code_verdict["a1_acceptance_passed"] is False
assert code_verdict["conditioning_probe_classification"]["blocking"] is False
conditioning = json.loads(code_review["conditioning-probe.json"])
assert conditioning["is_underflow"] is False
assert conditioning["point"]["defect"][0][0] == 0
assert conditioning["relative_point_error"] == 1
assert conditioning["reference_feedback"] == "1.0"

math = archived("independent-math-audit.tar.gz")
math_index = json.loads(math["artifact-index.json"])
assert set(math) == set(math_index) | {"artifact-index.json"}
assert all(digest(math[name]) == row["sha256"] and len(math[name]) == row["bytes"]
           for name, row in math_index.items())
assert math["frozen-fixtures.json"] == (BASE / "frozen-fixtures.json").read_bytes()
assert digest(math["frozen-fixtures.json"]) == (
    "63311bcecc3c31458df93b0c1079d0cec1d8cbc64ee1155abc07b028e9f9aaef"
)
limitation = json.loads(math["arithmetic-counterexample.json"])
assert limitation["computed_interval_contains_real_right_endpoint"] is False
assert limitation["arithmetic_verified"] is False
assert limitation["total_numerical_certified"] is False
parent = json.loads((BASE / "parent-math-replay.json").read_text())
assert len(parent["executions"]) == 3
assert set(parent["byte_matches"]) == {
    "probe-results.json", "cell-ranges.json", "precision-crosscheck.json",
    "arithmetic-counterexample.json", "frozen-fixtures.json",
}
assert all(value is True for value in parent["byte_matches"].values())
assert all(type(run["exit_code"]) is int and run["exit_code"] == 0
           for run in parent["executions"])

review = archived("initial-code-review.tar.gz")
review_index = json.loads(review["initial-review-evidence-index.json"])
assert set(review) == {row["path"] for row in review_index["files"]} | {
    "initial-review-evidence-index.json",
}
assert all(digest(review[row["path"]]) == row["sha256"]
           and len(review[row["path"]]) == row["bytes"]
           for row in review_index["files"])
assert (review["initial-review-verdict.json"]
        == (BASE / "initial-review.json").read_bytes())
initial = json.loads(review["initial-review-verdict.json"])
assert initial["passed"] is False and initial["logic_errors"]
assert initial["reviewed_tree"] == "6e7a156f025e08b31a314c5bda59183d6a96bf32"
parent_gamma = json.loads((BASE / "parent-gamma-replay.json").read_text())
assert [row["tree"] for row in parent_gamma["records"]] == [
    "6e7a156f025e08b31a314c5bda59183d6a96bf32",
    "8dcbfafd1a5373088e75437551692a85fd7f7b77",
]
assert parent_gamma["a1_acceptance_passed"] is False

# Preserve the failed documentation/evidence audit as a failure, independently
# of the corrected completion records. Never relabel its historical verdict.
preservation = json.loads(
    (BASE / "document-integrity-audit-preservation.json").read_text()
)
fields(preservation, {
    "passed": False, "a1_acceptance_passed": False,
    "original_review_files_verified": 41,
    "review_sha256": "f0515e743da507188e070f35502973a78581f5a6674ca900f470abdfe7c469c4",
    "review_index_sha256": (
        "bffce452fd509b5b4221322d0103e15582dfc9614805d49d100ef5efbd315888"
    ),
}, "failed audit preservation")
failed_name = "document-integrity-audit-failed.tar.gz"
assert digest((BASE / failed_name).read_bytes()) == preservation["archive_sha256"]
failed = archived(failed_name)
assert len(failed) == len(preservation["files"]) == 54
assert set(failed) == set(preservation["files"])
for name, raw in failed.items():
    fields(preservation["files"][name],
           {"bytes": len(raw), "sha256": digest(raw)}, name)
assert digest(failed["review/review.json"]) == preservation["review_sha256"]
assert digest(failed["review/hash-index.json"]) == preservation["review_index_sha256"]
failed_verdict = json.loads(failed["review/review.json"])
assert failed_verdict["passed"] is False
assert {row["id"] for row in failed_verdict["logic_errors"]} == {"DOC-01", "EVID-01"}
failed_index = json.loads(failed["review/hash-index.json"])
fields(failed_index, {"count": 41}, "original failed review index")
assert len(failed_index["files"]) == 41
assert {name[7:] for name in failed if name.startswith("review/")} == (
    set(failed_index["files"]) | {"hash-index.json"}
)
for name, row in failed_index["files"].items():
    fields(row, {"bytes": len(failed["review/" + name]),
                 "sha256": digest(failed["review/" + name])}, name)
frozen = json.loads(failed["frozen/manifest.json"])
assert len(frozen) == 10
assert {name[7:] for name in failed if name.startswith("frozen/")} == (
    set(frozen) | {"manifest.json"}
)
for name, row in frozen.items():
    fields(row, {"bytes": len(failed["frozen/" + name]),
                 "sha256": digest(failed["frozen/" + name])}, name)

records = validate_completion(ROOT, BASE, current_source=args.current_source)
results = {backend: {"tests": row["tests"], "sha256": row["xml_sha256"]}
           for backend, row in records.items()}

checksums = json.loads((BASE / "checksums.json").read_text())
unprotected = {"checksums.json", "verification.json"}
protected = {str(path.relative_to(BASE)) for path in BASE.rglob("*")
             if path.is_file() and path.name not in unprotected}
assert set(checksums) == protected and checksums
assert all(digest((BASE / name).read_bytes()) == value
           for name, value in checksums.items())
print(json.dumps({"scope": "record/source replay, not A1 acceptance or certification",
                  "source_count": len(source),
                  "current_source_verified": args.current_source,
                  "local_exit_files_verified": args.current_source,
                  "replay_mode": (
                      "current-source-and-local-exits" if args.current_source
                      else "preserved-artifacts"
                  ),
                  "process_completion_observations_verified": True,
                  "full_suites": results, "preserved_arithmetic_under_enclosure": True,
                  "scoped_source_review_passed": code_verdict["passed"],
                  "cancellation_false_zero_limitation_preserved": True,
                  "a1_acceptance_passed": False}, indent=2))
