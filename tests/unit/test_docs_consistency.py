from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path

import hpfracc

ROOT = Path(__file__).resolve().parents[2]
DOC_PATHS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolve_dotted_name(name: str) -> object:
    parts = name.split(".")
    if parts[0] == "hp":
        obj: object = hpfracc
        parts = parts[1:]
    else:
        obj = importlib.import_module(parts[0])
        parts = parts[1:]

    dotted_so_far = "hpfracc"
    for part in parts:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            dotted_so_far = f"{dotted_so_far}.{part}"
            obj = importlib.import_module(dotted_so_far)
            continue
        dotted_so_far = f"{dotted_so_far}.{part}"
    return obj


def test_documented_benchmark_modules_import_cleanly() -> None:
    documented_modules: set[str] = set()
    pattern = re.compile(r"python\s+-m\s+(benchmarks(?:\.[A-Za-z_]\w*)+)")

    for path in DOC_PATHS:
        documented_modules.update(pattern.findall(_read(path)))

    assert documented_modules, "Expected docs to include benchmark module commands."

    missing_or_broken: dict[str, str] = {}
    for module_name in sorted(documented_modules):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - assertion reports details.
            missing_or_broken[module_name] = repr(exc)

    assert missing_or_broken == {}


def test_api_contract_public_dotted_names_exist() -> None:
    contract = _read(ROOT / "docs/api/contract.md")
    documented_names = set(
        re.findall(r"`(hp(?:\.[A-Za-z_]\w*)+|hpfracc(?:\.[A-Za-z_]\w*)+)`", contract)
    )

    assert documented_names, "Expected API contract to document public API names."

    missing: dict[str, str] = {}
    for name in sorted(documented_names):
        try:
            _resolve_dotted_name(name)
        except Exception as exc:  # pragma: no cover - assertion reports details.
            missing[name] = repr(exc)

    assert missing == {}


def test_release_version_metadata_is_consistent() -> None:
    pyproject = tomllib.loads(_read(ROOT / "pyproject.toml"))
    project_version = pyproject["project"]["version"]
    version_module = _read(ROOT / "src/hpfracc/_version.py")
    changelog = _read(ROOT / "CHANGELOG.md")
    citation = _read(ROOT / "CITATION.cff")

    assert hpfracc.__version__ == project_version
    assert f'__version__ = "{project_version}"' in version_module
    assert f"## {project_version}" in changelog
    assert f"version: {project_version}" in citation


def test_release_checklist_mentions_docs_consistency_test() -> None:
    checklist = _read(ROOT / "docs/developer/release-checklist.md")

    assert "uv run python -m pytest tests/unit/test_docs_consistency.py" in checklist


def test_domain_model_guardrails_are_documented() -> None:
    guardrails_path = ROOT / "docs/developer/domain-model-guardrails.md"
    developer_index = _read(ROOT / "docs/developer/index.md")
    mkdocs = _read(ROOT / "mkdocs.yml")

    assert guardrails_path.exists()
    guardrails = _read(guardrails_path).lower()

    required_phrases = [
        "neural mass",
        "neural field",
        "phantom-brain",
        "experimental until independently validated",
        "no biological, clinical, diagnostic, or subject-specific claims",
        "model-relative tests",
        "sensitivity checks",
    ]
    missing_phrases = [phrase for phrase in required_phrases if phrase not in guardrails]

    assert missing_phrases == []
    assert "domain-model-guardrails.md" in developer_index
    assert "developer/domain-model-guardrails.md" in mkdocs
