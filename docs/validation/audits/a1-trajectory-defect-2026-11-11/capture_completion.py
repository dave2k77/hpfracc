"""Read actual current-suite exits/identities; finalize only completed evidence."""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# This cache CLI deliberately shares the portable audit's standard-library validator.
base = Path("docs/validation/audits/a1-trajectory-defect-2026-11-11")
sys.dont_write_bytecode = True
sys.path.insert(0, str(base.resolve()))
from completion_integrity import (  # noqa: E402
    expected_readbacks,
    validate_completion,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("action", choices=("cpu", "cuda", "finalize"))
args = parser.parse_args()
root = Path.cwd()

if args.action in ("cpu", "cuda"):
    record = expected_readbacks(
        root,
        base,
        current_source=True,
        backends=(args.action,),
        preserve_exits=True,
    )[args.action]
    record.update(
        captured_at_utc=datetime.now(UTC).isoformat(),
        scope=(
            "Source-bound actual default-suite evidence, "
            "not A1 acceptance or cancellation safety"
        ),
    )
    (base / f"{args.action}-readback.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))
    sys.exit(0)

# Fail before publishing anything unless both readbacks bind to actual local
# inputs, XML, exits AND independent process-completion observations for the plan.
records = validate_completion(root, base, current_source=True, require_executions=False)
runs = []
for record in records.values():
    runs.append(
        {
            **{
                key: record[key]
                for key in ("backend", "session_id", "pid", "exit_record", "exit_code")
            },
            "status": "exited",
            "exited": True,
            "tests_passed": record["tests"],
        }
    )
source_tree = records["cpu"]["source_tree"]
(base / "executions.json").write_text(
    json.dumps(
        {
            "source_tree": source_tree,
            "full_runs": runs,
            "scope": (
                "Actual source-bound suites; integrity is not scientific acceptance"
            ),
            "a1_acceptance_passed": False,
        },
        indent=2,
    )
    + "\n"
)
validate_completion(root, base, current_source=True)
checksums = {
    str(path.relative_to(base)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(base.rglob("*"))
    if path.is_file() and path.name not in {"checksums.json", "verification.json"}
}
(base / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
print(
    json.dumps(
        {
            "source_tree": source_tree,
            "full_runs": runs,
            "protected_artifacts": len(checksums),
            "replay_mode": "current-source-and-local-exits",
            "a1_acceptance_passed": False,
        },
        indent=2,
    )
)
