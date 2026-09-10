"""Guard current development pointers without snapshotting historical prose."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import hpfracc as hp

ROOT = Path(__file__).resolve().parents[2]
CURRENT_DOCS = (
    "docs/developer/roadmap.md",
    "v0.1_engineering_contract.md",
    "README.md",
    "docs/index.md",
    "docs/developer/index.md",
    "docs/validation/status.md",
)


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _field(path: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}: (.+)$", _read(path), re.MULTILINE)
    assert match, f"Missing {name!r} in {path}"
    return match.group(1)


def test_active_phase_agrees_across_current_guidance() -> None:
    phases = {_field(path, "Active phase") for path in CURRENT_DOCS}
    assert len(phases) == 1
    assert re.fullmatch(r"Phase [A-D]", phases.pop())


def test_latest_release_is_distinct_from_unreleased_changes() -> None:
    changelog = _read("CHANGELOG.md")
    release = re.search(
        r"^## (\S+)\n\nStatus: [^\n]*released (\d{4}-\d{2}-\d{2})\.",
        changelog,
        re.MULTILINE,
    )
    assert release, "Expected a dated release entry, not an Unreleased header"
    expected = f"`{release.group(1)}` ({release.group(2)})."
    for path in (*CURRENT_DOCS, "docs/developer/release-checklist.md"):
        assert _field(path, "Latest tagged release") == expected
    roadmap = _read(CURRENT_DOCS[0])
    assert '<a id="phase-a-unreleased"></a>' in roadmap
    assert '<a id="phase-a-gates"></a>' in roadmap


def test_engineering_contract_simulate_signature_matches_runtime() -> None:
    blocks = re.findall(r"```python\n(.*?)```", _read(CURRENT_DOCS[1]), re.DOTALL)
    signatures = [block for block in blocks if "def simulate(" in block]
    assert len(signatures) == 1
    function = ast.parse(signatures[0]).body[0]
    assert isinstance(function, ast.FunctionDef)
    actual = inspect.signature(hp.solvers.simulate).parameters
    args = function.args
    assert not (args.posonlyargs or args.args or args.vararg or args.kwarg)
    assert [arg.arg for arg in args.kwonlyargs] == list(actual)
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        parameter = actual[arg.arg]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        if default is None:
            assert parameter.default is inspect.Parameter.empty
        else:
            assert ast.literal_eval(default) == parameter.default


def test_release_template_requires_checked_reports_without_signoff() -> None:
    checklist = _read("docs/developer/release-checklist.md")
    for module in ("validation_summary", "core_hardening"):
        assert f"python -m benchmarks.numerical.{module} --check" in checklist
    assert "python -m pytest -m performance" in checklist
    assert "not an executed release record" in checklist
    assert not re.search(r"^- \[[xX]\]", checklist, re.MULTILINE)


def test_historical_plans_point_to_current_roadmap() -> None:
    for name in ("phase-2-caputo-fde-solver", "phase-6-research-ready-alpha"):
        intro = _read(f"docs/developer/{name}.md").split("##", maxsplit=1)[0]
        assert "Historical" in intro
        assert "roadmap.md" in intro


def test_entrypoints_link_to_current_development_and_validation() -> None:
    for path in ("README.md", "docs/index.md", "docs/developer/index.md"):
        text = _read(path)
        assert "roadmap.md" in text
        assert "validation/status.md" in text
