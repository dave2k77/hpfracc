"""Standard-library completion binding; record integrity, not numerical acceptance.

Preserved-artifact replay checks archived source and exit bytes plus the explicit
process-completion observations. Only current-source mode checks local inputs and
local exit paths. Observations must be supplied by the caller, never inferred from
JUnit summaries or manufactured by this module.
"""

import hashlib
import json
import re
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


class IntegrityError(ValueError):
    """Evidence is missing, inconsistent, or not canonically typed."""


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def same(actual, expected, label):
    require(type(actual) is type(expected), f"{label}: noncanonical type")
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), f"{label}: keys differ")
        for key, value in expected.items():
            same(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f"{label}: length differs")
        for index, value in enumerate(expected):
            same(actual[index], value, f"{label}[{index}]")
    else:
        require(actual == expected, f"{label}: value differs")


def fields(actual, expected, label):
    require(type(actual) is dict, f"{label}: expected object")
    for key, value in expected.items():
        require(key in actual, f"{label}: missing {key}")
        same(actual[key], value, f"{label}.{key}")


def relative(name):
    require(type(name) is str and bool(name), "Empty/non-string relative path")
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe path: {name}")
    return path


def runs_by_backend(rows, label):
    require(type(rows) is list and len(rows) == 2, f"{label}: require two runs")
    result = {}
    for row in rows:
        require(type(row) is dict, f"{label}: expected run object")
        backend = row.get("backend")
        require(
            type(backend) is str and backend in {"cpu", "cuda"},
            f"{label}: invalid backend",
        )
        require(backend not in result, f"{label}: duplicate backend")
        result[backend] = row
    return result


def source_inputs(root, base, plan, current_source):
    snapshot = load_json(base / "source-snapshot.json")
    paths = snapshot["paths"]
    require(type(paths) is dict and bool(paths), "Empty/non-object source inputs")
    fields(snapshot, {"count": len(paths)}, "snapshot")
    fields(
        plan,
        {"source_count": len(paths), "source_tree": snapshot["source_tree"]},
        "plan",
    )
    require(
        type(snapshot["source_tree"]) is str
        and re.fullmatch(r"[0-9a-f]{40}", snapshot["source_tree"]),
        "Invalid source tree",
    )
    review = load_json(base / "arithmetic-code-review.json")
    fields(
        review,
        {
            "passed": True,
            "logic_errors": [],
            "security_concerns": [],
            "reviewed_tree": snapshot["source_tree"],
        },
        "source review",
    )
    source = {}
    with tarfile.open(base / "source-snapshot.tar.gz", "r:gz") as archive:
        for member in archive:
            relative(member.name)
            require(
                member.isfile() and member.name not in source,
                "Non-file/duplicate source archive entry",
            )
            stream = archive.extractfile(member)
            require(stream is not None, "Missing source archive stream")
            source[member.name] = stream.read()
    require(source.keys() == paths.keys(), "Source archive input set differs")
    for name, expected in paths.items():
        relative(name)
        same(digest(source[name]), expected, f"archived source {name}")
        if current_source:
            path = root / name
            require(
                not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
                f"Unsafe current source: {name}",
            )
            same(digest(path.read_bytes()), expected, f"current source {name}")
    return snapshot


def xml_record(raw, expected, count):
    tree = ET.fromstring(raw)
    suites = list(tree.iter("testsuite"))
    require(len(suites) == 1, "Require one JUnit suite")
    cases = list(tree.iter("testcase"))
    actual = Counter((case.attrib["classname"], case.attrib["name"]) for case in cases)
    require(actual == expected, "Exact JUnit identities differ from collection")
    counts = {
        "tests": len(cases),
        "failures": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
    }
    same(counts, dict(tests=count, failures=0, errors=0, skipped=0), "JUnit counts")
    for key, value in counts.items():
        same(suites[0].attrib[key], str(value), f"JUnit header {key}")
    return counts


def expected_readbacks(
    root, base, *, current_source, backends=None, preserve_exits=False
):
    """Compute fields from bytes/observations; optionally preserve real local exits."""
    plan = load_json(base / "execution-plan.json")
    snapshot = source_inputs(root, base, plan, current_source)
    runs = runs_by_backend(plan["full_runs"], "plan")
    require(
        len({row["session_id"] for row in runs.values()}) == 2,
        "Duplicate process session",
    )
    require(len({row["pid"] for row in runs.values()}) == 2, "Duplicate process PID")
    require(
        len({row["exit_record"] for row in runs.values()}) == 2, "Duplicate exit record"
    )
    for run in runs.values():
        require(type(run["pid"]) is int and run["pid"] > 0, "Noncanonical plan PID")
        for key in ("session_id", "exit_record"):
            require(type(run[key]) is str and bool(run[key]), f"Invalid plan {key}")
    nodes = [
        line
        for line in (base / "collection.txt").read_text().splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    require(bool(nodes) and len(nodes) == len(set(nodes)), "Empty/duplicate collection")
    same(plan["expected_selected_tests"], len(nodes), "plan selected count")
    expected = Counter()
    for node in nodes:
        parts = node.split("::")
        module = parts[0].removesuffix(".py").replace("/", ".")
        expected[(".".join([module, *parts[1:-1]]), parts[-1])] += 1
    process_path = base / "process-completion.json"
    process_raw = process_path.read_bytes()
    process_record = load_json(process_path)
    fields(
        process_record,
        {"source_tree": snapshot["source_tree"], "a1_acceptance_passed": False},
        "process completion",
    )
    observations = runs_by_backend(process_record["full_runs"], "completion")
    result = {}
    for backend in backends if backends is not None else ("cpu", "cuda"):
        run = runs[backend]
        identity = {key: run[key] for key in ("backend", "session_id", "pid")}
        fields(
            observations[backend],
            {
                **identity,
                "status": "exited",
                "exit_code": 0,
                "completion_reason": "exited",
                "termination_source": "",
            },
            f"process completion {backend}",
        )
        preserved = f"completion-exit-{backend}.txt"
        if preserve_exits:
            require(current_source, "Exit capture requires current-source mode")
            raw_exit = Path(run["exit_record"]).read_bytes()
        else:
            raw_exit = (base / preserved).read_bytes()
        require(
            raw_exit.strip() == b"0", f"{backend}: actual exit contents are not zero"
        )
        if current_source:
            same(
                Path(run["exit_record"]).read_bytes(),
                raw_exit,
                f"{backend}: local/preserved exit bytes",
            )
        raw = (base / f"full-{backend}.xml").read_bytes()
        counts = xml_record(raw, expected, len(nodes))
        result[backend] = {
            **identity,
            "exit_record": run["exit_record"],
            "source_tree": snapshot["source_tree"],
            "exit_code": 0,
            "tests": len(nodes),
            "counts": counts,
            "source_inputs_verified": len(snapshot["paths"]),
            "source_mismatches": [],
            "exact_identities_verified": True,
            "xml_sha256": digest(raw),
            "preserved_exit_record": preserved,
            "exit_record_sha256": digest(raw_exit),
            "process_completion_sha256": digest(process_raw),
        }
        if preserve_exits:
            (base / preserved).write_bytes(raw_exit)
    return result


def validate_completion(root, base, *, current_source, require_executions=True):
    """Bind both readbacks (and published runs) to the plan and real evidence bytes."""
    expected = expected_readbacks(root, base, current_source=current_source)
    for backend, record in expected.items():
        fields(
            load_json(base / f"{backend}-readback.json"), record, f"readback {backend}"
        )
    if require_executions:
        executions = load_json(base / "executions.json")
        fields(
            executions,
            {
                "source_tree": expected["cpu"]["source_tree"],
                "a1_acceptance_passed": False,
            },
            "executions",
        )
        runs = runs_by_backend(executions["full_runs"], "executions")
        for backend, record in expected.items():
            fields(
                runs[backend],
                {
                    **{
                        key: record[key]
                        for key in (
                            "backend",
                            "session_id",
                            "pid",
                            "exit_record",
                            "exit_code",
                        )
                    },
                    "status": "exited",
                    "exited": True,
                    "tests_passed": record["tests"],
                },
                f"execution {backend}",
            )
    return expected
