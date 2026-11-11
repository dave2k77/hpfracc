"""Controlled integrity tests, NOT candidate full-suite results.

Invoke explicitly; these tests are outside the default numerical test collection.
The real CLIs run on copies. No actual candidate readback/finalization is written.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RELATIVE = HERE.relative_to(ROOT)
PRODUCER = HERE / "capture_completion.py"
BASELINE = os.environ.get("EVID01_BASELINE_SCRIPTS")
SCOPE = "Controlled synthetic integrity test; NOT candidate full-suite results"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def test_capture_cli_is_part_of_preserved_artifact():
    """Reproduction must not depend on the original machine's scratch directory."""
    assert (HERE / "capture_completion.py").is_file()
    assert PRODUCER == HERE / "capture_completion.py"


def put(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def load(path):
    return json.loads(path.read_text())


def rehash(base):
    put(
        base / "checksums.json",
        {
            str(path.relative_to(base)): sha(path.read_bytes())
            for path in sorted(base.rglob("*"))
            if path.is_file()
            and path.name not in {"checksums.json", "verification.json"}
        },
    )


@pytest.fixture
def sandbox(tmp_path):
    root = tmp_path / "controlled-sandbox"
    base = root / RELATIVE
    shutil.copytree(HERE, base, ignore=shutil.ignore_patterns("__pycache__"))
    scripts = Path(BASELINE) if BASELINE else None
    shutil.copyfile(
        scripts / "full_evidence.py" if scripts else PRODUCER, root / "full_evidence.py"
    )
    if scripts:
        shutil.copyfile(scripts / "verify.py", base / "verify.py")
    with tarfile.open(base / "source-snapshot.tar.gz", "r:gz") as archive:
        for member in archive:
            assert member.isfile() and ".." not in Path(member.name).parts
            path = root / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            assert stream is not None
            path.write_bytes(stream.read())
    plan = load(base / "execution-plan.json")
    completion = []
    for index, run in enumerate(plan["full_runs"]):
        backend = run["backend"]
        run.update(
            session_id=f"controlled-{backend}",
            pid=700001 + index,
            exit_record=str(root / f"controlled-{backend}.exit"),
        )
        Path(run["exit_record"]).write_bytes(b"0\n")
        preserved = f"completion-exit-{backend}.txt"
        (base / preserved).write_bytes(b"0\n")
        completion.append(
            {
                "backend": backend,
                "session_id": run["session_id"],
                "pid": run["pid"],
                "status": "exited",
                "exit_code": 0,
                "completion_reason": "exited",
                "termination_source": "",
            }
        )
    plan["scope"] = SCOPE
    put(base / "execution-plan.json", plan)
    put(
        base / "process-completion.json",
        {
            "scope": SCOPE,
            "source_tree": plan["source_tree"],
            "full_runs": completion,
            "a1_acceptance_passed": False,
        },
    )
    process_sha = sha((base / "process-completion.json").read_bytes())
    executions = []
    for run in plan["full_runs"]:
        backend = run["backend"]
        put(
            base / f"{backend}-readback.json",
            {
                "scope": SCOPE,
                "backend": backend,
                "session_id": run["session_id"],
                "pid": run["pid"],
                "exit_record": run["exit_record"],
                "source_tree": plan["source_tree"],
                "exit_code": 0,
                "tests": plan["expected_selected_tests"],
                "counts": dict(
                    tests=plan["expected_selected_tests"],
                    failures=0,
                    errors=0,
                    skipped=0,
                ),
                "source_inputs_verified": plan["source_count"],
                "source_mismatches": [],
                "exact_identities_verified": True,
                "xml_sha256": sha((base / f"full-{backend}.xml").read_bytes()),
                "preserved_exit_record": f"completion-exit-{backend}.txt",
                "exit_record_sha256": sha(b"0\n"),
                "process_completion_sha256": process_sha,
            },
        )
        executions.append(
            {
                "backend": backend,
                "session_id": run["session_id"],
                "pid": run["pid"],
                "exit_record": run["exit_record"],
                "status": "exited",
                "exited": True,
                "exit_code": 0,
                "tests_passed": plan["expected_selected_tests"],
            }
        )
    put(
        base / "executions.json",
        {
            "scope": SCOPE,
            "source_tree": plan["source_tree"],
            "full_runs": executions,
            "a1_acceptance_passed": False,
        },
    )
    return root, base


def cli(sandbox, tool, *extra):
    root, base = sandbox
    if tool == "verify-current":
        if not (root / ".git").exists():
            clone = root.parent / "controlled-git-reference"
            commands = [
                ["git", "clone", "--shared", "--no-checkout", str(ROOT), str(clone)],
                [
                    "git",
                    "-C",
                    str(root),
                    "add",
                    "--",
                    *load(base / "source-snapshot.json")["paths"],
                ],
            ]
            first = subprocess.run(commands[0], capture_output=True, check=True)
            shutil.copytree(clone / ".git", root / ".git")
            second = subprocess.run(commands[1], capture_output=True, check=True)
            put(
                root / "git-setup.json",
                {
                    "scope": SCOPE,
                    "commands": commands,
                    "actual_exit_codes": [0, 0],
                    "stderr": [first.stderr.decode(), second.stderr.decode()],
                },
            )
        extra = ("--current-source", *extra)
    rehash(base)  # Tampering must be caught semantically, not by the checksum wrapper.
    command = [sys.executable]
    if tool == "finalize":
        command += [str(root / "full_evidence.py"), "finalize"]
    elif tool in {"cpu", "cuda"}:
        command += [str(root / "full_evidence.py"), tool]
    else:
        command += [str(base / "verify.py"), *extra]
    result = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        timeout=30,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    number = len(list(root.glob("cli-*-result.json")))
    stem = root / f"cli-{number}"
    stem.with_suffix(".stdout").write_bytes(result.stdout)
    stem.with_suffix(".stderr").write_bytes(result.stderr)
    put(
        root / f"cli-{number}-result.json",
        {
            "scope": SCOPE,
            "command": command,
            "actual_exit_code": result.returncode,
            "stdout_sha256": sha(result.stdout),
            "stderr_sha256": sha(result.stderr),
        },
    )
    return result


@pytest.mark.parametrize("tool", ["finalize", "verify"])
def test_positive_real_cli(sandbox, tool):
    result = cli(sandbox, tool)
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize("tool", ["finalize", "verify"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("xml_sha256", "0" * 64),
        ("source_inputs_verified", 0),
    ],
)
def test_rejects_unbound_readback(sandbox, tool, field, value):
    base = sandbox[1]
    path = base / "cpu-readback.json"
    record = load(path)
    record[field] = value
    put(path, record)
    result = cli(sandbox, tool)
    assert result.returncode != 0, (
        f"{tool} accepted {field}={value!r}; controlled test, not suite evidence"
    )


READBACK_MUTATIONS = [
    ("backend", "cuda"),
    ("pid", 700099),
    ("pid", 700001.0),
    ("session_id", "controlled-other"),
    ("exit_record", "other.exit"),
    ("source_tree", "0" * 40),
    ("tests", 0),
    ("tests", 1017.0),
    ("tests", True),
    ("counts", dict(tests=1016, failures=0, errors=0, skipped=0)),
    ("counts", dict(tests=1017, failures=False, errors=0, skipped=0)),
    ("source_inputs_verified", 126.0),
    ("source_inputs_verified", True),
    ("source_mismatches", {}),
    ("source_mismatches", ["changed-input"]),
    ("exact_identities_verified", 1),
    ("exit_code", False),
    ("exit_code", 0.0),
    ("preserved_exit_record", "other.txt"),
    ("exit_record_sha256", "0" * 64),
    ("process_completion_sha256", "0" * 64),
    ("xml_sha256", ""),
]


@pytest.mark.parametrize("tool", ["finalize", "verify"])
@pytest.mark.parametrize("field,value", READBACK_MUTATIONS)
def test_rejects_readback_identity_counts_types(sandbox, tool, field, value):
    path = sandbox[1] / "cpu-readback.json"
    record = load(path)
    record[field] = value
    put(path, record)
    result = cli(sandbox, tool)
    assert result.returncode != 0, f"{tool} accepted {field}={value!r}"


@pytest.mark.parametrize("tool", ["finalize", "verify"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "cuda"),
        ("session_id", "controlled-other"),
        ("pid", 700099),
        ("pid", 700001.0),
        ("status", "running"),
        ("exit_code", 1),
        ("exit_code", False),
        ("exit_code", 0.0),
        ("completion_reason", "running"),
        ("termination_source", "killed"),
    ],
)
def test_rejects_process_observation_drift(sandbox, tool, field, value):
    base = sandbox[1]
    path = base / "process-completion.json"
    record = load(path)
    record["full_runs"][0][field] = value
    put(path, record)
    # Refresh the wrapper digest so observation semantics are the only barrier.
    for backend in ("cpu", "cuda"):
        path = base / f"{backend}-readback.json"
        record = load(path)
        record["process_completion_sha256"] = sha(
            (base / "process-completion.json").read_bytes()
        )
        put(path, record)
    result = cli(sandbox, tool)
    assert result.returncode != 0, f"{tool} accepted process {field}={value!r}"


@pytest.mark.parametrize("tool", ["finalize", "verify"])
@pytest.mark.parametrize(
    "target,field,value",
    [
        ("execution-plan.json", "source_tree", "0" * 40),
        ("execution-plan.json", "source_count", 0),
        ("execution-plan.json", "source_count", 126.0),
        ("execution-plan.json", "expected_selected_tests", 1016),
        ("execution-plan.json", "expected_selected_tests", 1017.0),
        ("source-snapshot.json", "count", 126.0),
        ("source-snapshot.json", "count", 0),
        ("process-completion.json", "source_tree", "0" * 40),
    ],
)
def test_rejects_plan_snapshot_completion_drift(sandbox, tool, target, field, value):
    path = sandbox[1] / target
    record = load(path)
    record[field] = value
    put(path, record)
    result = cli(sandbox, tool)
    assert result.returncode != 0, f"{tool} accepted {target}:{field}={value!r}"


@pytest.mark.parametrize("tool", ["finalize", "verify"])
@pytest.mark.parametrize(
    "name",
    [
        "process-completion.json",
        "completion-exit-cpu.txt",
        "cpu-readback.json",
    ],
)
def test_rejects_missing_evidence(sandbox, tool, name):
    (sandbox[1] / name).unlink()
    result = cli(sandbox, tool)
    assert result.returncode != 0, f"{tool} accepted missing {name}"


@pytest.mark.parametrize("tool", ["finalize", "verify"])
@pytest.mark.parametrize("mutation", ["identity", "header", "failure", "xml-bytes"])
def test_rejects_xml_drift(sandbox, tool, mutation):
    base = sandbox[1]
    path = base / "full-cpu.xml"
    raw = path.read_bytes()
    if mutation == "identity":
        raw = raw.replace(b'<testcase classname="', b'<testcase classname="changed.', 1)
    elif mutation == "header":
        raw = raw.replace(b'tests="1017"', b'tests="1016"', 1)
    elif mutation == "failure":
        raw = raw.replace(
            b" />", b"><failure>controlled failure</failure></testcase>", 1
        )
    else:
        raw += b"\n"
    assert raw != path.read_bytes()
    path.write_bytes(raw)
    if mutation != "xml-bytes":
        record = load(base / "cpu-readback.json")
        record["xml_sha256"] = sha(raw)
        put(base / "cpu-readback.json", record)
    result = cli(sandbox, tool)
    assert result.returncode != 0, f"{tool} accepted XML {mutation} drift"


@pytest.mark.parametrize("tool", ["finalize", "verify"])
def test_rejects_nonzero_preserved_exit_even_with_refreshed_hash(sandbox, tool):
    base = sandbox[1]
    (base / "completion-exit-cpu.txt").write_bytes(b"1\n")
    record = load(base / "cpu-readback.json")
    record["exit_record_sha256"] = sha(b"1\n")
    put(base / "cpu-readback.json", record)
    result = cli(sandbox, tool)
    assert result.returncode != 0


@pytest.mark.parametrize("tool", ["finalize", "verify-current"])
@pytest.mark.parametrize("mutation", ["nonzero-exit", "missing-exit", "source-bytes"])
def test_current_source_requires_actual_local_files(sandbox, tool, mutation):
    root, base = sandbox
    run = load(base / "execution-plan.json")["full_runs"][0]
    if mutation == "source-bytes":
        path = root / "benchmarks/numerical/README.md"
        path.write_bytes(path.read_bytes() + b"\ncontrolled mutation\n")
    elif mutation == "missing-exit":
        Path(run["exit_record"]).unlink()
    else:
        Path(run["exit_record"]).write_bytes(b"1\n")
    result = cli(sandbox, tool)
    assert result.returncode != 0, f"{tool} accepted {mutation}"


def test_portable_replay_explicitly_does_not_claim_local_checks(sandbox):
    root, base = sandbox
    for run in load(base / "execution-plan.json")["full_runs"]:
        Path(run["exit_record"]).unlink()
    (root / "benchmarks/numerical/README.md").unlink()
    result = cli(sandbox, "verify")
    assert result.returncode == 0, result.stderr.decode()
    report = json.loads(result.stdout)
    assert report["replay_mode"] == "preserved-artifacts"
    assert report["local_exit_files_verified"] is False
    assert report["current_source_verified"] is False
    assert report["process_completion_observations_verified"] is True


def test_current_source_positive_and_producer_roundtrip(sandbox):
    for backend in ("cpu", "cuda"):
        result = cli(sandbox, backend)
        assert result.returncode == 0, result.stderr.decode()
        record = json.loads(result.stdout)
        plan = load(sandbox[1] / "execution-plan.json")
        run = next(row for row in plan["full_runs"] if row["backend"] == backend)
        for key in ("session_id", "pid", "exit_record"):
            assert type(record[key]) is type(run[key]) and record[key] == run[key]
        assert (sandbox[1] / record["preserved_exit_record"]).read_bytes() == b"0\n"
    result = cli(sandbox, "finalize")
    assert result.returncode == 0, result.stderr.decode()
    result = cli(sandbox, "verify-current")
    assert result.returncode == 0, result.stderr.decode()
    report = json.loads(result.stdout)
    assert report["replay_mode"] == "current-source-and-local-exits"
    assert report["current_source_verified"] is True
    assert report["local_exit_files_verified"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("pid", 700099),
        ("session_id", "controlled-other"),
        ("exit_code", False),
        ("exit_record", "other.exit"),
        ("tests_passed", 1017.0),
    ],
)
def test_rejects_published_execution_drift(sandbox, field, value):
    path = sandbox[1] / "executions.json"
    record = load(path)
    record["full_runs"][0][field] = value
    put(path, record)
    result = cli(sandbox, "verify")
    assert result.returncode != 0


@pytest.mark.parametrize("mutation", ["passed", "count", "archive-bytes"])
def test_replays_failed_document_audit_preservation(sandbox, mutation):
    base = sandbox[1]
    path = base / "document-integrity-audit-preservation.json"
    record = load(path)
    if mutation == "passed":
        record["passed"] = True
    elif mutation == "count":
        record["original_review_files_verified"] = 0
    else:
        archive = base / "document-integrity-audit-failed.tar.gz"
        archive.write_bytes(archive.read_bytes() + b"controlled mutation")
    put(path, record)
    result = cli(sandbox, "verify")
    assert result.returncode != 0


@pytest.mark.parametrize("tool", ["finalize", "verify"])
@pytest.mark.parametrize("mutation", ["input-hash", "missing-input", "duplicate-id"])
def test_rejects_source_input_and_collection_drift(sandbox, tool, mutation):
    base = sandbox[1]
    path = base / "source-snapshot.json"
    record = load(path)
    if mutation == "input-hash":
        record["paths"]["benchmarks/numerical/README.md"] = "0" * 64
        put(path, record)
    elif mutation == "missing-input":
        del record["paths"]["benchmarks/numerical/README.md"]
        record["count"] -= 1
        put(path, record)
    else:
        path = base / "collection.txt"
        nodes = path.read_text().splitlines()
        first = next(
            node for node in nodes if node.startswith("tests/") and "::" in node
        )
        path.write_text(path.read_text() + first + "\n")
    result = cli(sandbox, tool)
    assert result.returncode != 0
