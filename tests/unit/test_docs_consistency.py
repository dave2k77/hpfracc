from __future__ import annotations

import importlib
import json
import re
import tomllib
from itertools import product
from pathlib import Path

import pytest

import hpfracc

ROOT = Path(__file__).resolve().parents[2]
DOC_PATHS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
SCIENTIFIC_ACCEPTANCE_BUNDLE = (
    ROOT / "docs/validation/audits/scientific-acceptance-2026-11-10"
)
NUM004_PRECISION_BUNDLE = (
    ROOT / "docs/validation/audits/num004-direct-precision-2026-11-10"
)
SCIENTIFIC_ACCEPTANCE_VARIANTS = list(
    product(("cpu", "gpu"), ("float32", "float64"), ("direct", "stable"))
)


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

    command = "uv run --frozen python -m pytest tests/unit/test_docs_consistency.py"
    assert command in checklist


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
    missing_phrases = [
        phrase for phrase in required_phrases if phrase not in guardrails
    ]

    assert missing_phrases == []
    assert "domain-model-guardrails.md" in developer_index
    assert "developer/domain-model-guardrails.md" in mkdocs


def _scientific_acceptance_outcome(
    backend: str,
    dtype: str,
    weights: str,
    *,
    bundle: Path = SCIENTIFIC_ACCEPTANCE_BUNDLE,
) -> tuple[int, float]:
    report = json.loads(
        _read(
            bundle / f"scientific_acceptance_{backend}_{dtype}_{weights}.json"
        )
    )
    assert report["runtime"]["backend"] == backend
    weak_decay = [
        row for row in report["rows"] if row["spec"]["case_id"] == "weak_decay"
    ]
    assert len(weak_decay) == 1, "Expected exactly one weak-decay measurement"
    row = weak_decay[0]
    assert (row["spec"]["dtype"], row["spec"]["weight_evaluation"]) == (
        dtype,
        weights,
    )
    policy_passed = all(row["policy_passed"] for row in report["rows"])
    assert report["summary"]["policy_passed"] is policy_passed
    return int(not policy_passed), row["measurement"]["independent_max_scaled_residual"]


def _assert_scientific_acceptance_readme_outcomes(
    bundle: Path, backend: str, dtype: str, weights: str
) -> None:
    readme = _read(bundle / "README.md")
    outcomes = readme.split("## Actual outcomes\n", maxsplit=1)[1].split(
        "\n## ", maxsplit=1
    )[0]
    table = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in outcomes.splitlines()
        if line.startswith("|")
    ][2:]
    assert all(len(row) == 5 for row in table)
    # Compare the list, not a dict/set that could silently discard duplicates.
    assert sorted(tuple(row[:3]) for row in table) == sorted(
        SCIENTIFIC_ACCEPTANCE_VARIANTS
    )
    published = next(row for row in table if row[:3] == [backend, dtype, weights])
    exit_code, residual = _scientific_acceptance_outcome(
        backend, dtype, weights, bundle=bundle
    )
    assert published[3:] == [str(exit_code), format(residual, ".6g")]

    verification = json.loads(_read(bundle / "verification.json"))
    runs = verification["runs"]
    assert sorted(
        (run["backend"], run["dtype"], run["weight_evaluation"]) for run in runs
    ) == sorted(SCIENTIFIC_ACCEPTANCE_VARIANTS)
    verified = next(
        run
        for run in runs
        if (run["backend"], run["dtype"], run["weight_evaluation"])
        == (backend, dtype, weights)
    )
    assert verified["exit_code"] == exit_code
    assert verified["weak_decay_independent_scaled_residual"] == residual


@pytest.mark.parametrize("backend,dtype,weights", SCIENTIFIC_ACCEPTANCE_VARIANTS)
def test_scientific_acceptance_readme_outcomes_match_bundled_rows(
    backend: str, dtype: str, weights: str
) -> None:
    _assert_scientific_acceptance_readme_outcomes(
        SCIENTIFIC_ACCEPTANCE_BUNDLE, backend, dtype, weights
    )


@pytest.mark.parametrize("backend,dtype,weights", SCIENTIFIC_ACCEPTANCE_VARIANTS)
def test_num004_precision_readme_outcomes_match_bundled_rows(
    backend: str, dtype: str, weights: str
) -> None:
    _assert_scientific_acceptance_readme_outcomes(
        NUM004_PRECISION_BUNDLE, backend, dtype, weights
    )


@pytest.mark.parametrize("weights", ("direct", "stable"))
def test_num004_current_residual_citations_match_bundled_rows(weights: str) -> None:
    tracker = _read(ROOT / "docs/developer/bug-resolution-tracker.md")
    sections = re.findall(
        r"^### NUM-004 [^\n]*\n(.*?)(?=^#{1,3} |\Z)",
        tracker,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert len(sections) == 1, "Expected exactly one NUM-004 tracker section"
    # Target only the current paired claims, never historical/archive numbers.
    current = " ".join(sections[0].split())
    claim = (
        r"scaled residuals are (\S+) on CPU and (\S+) on CUDA,"
        if weights == "direct"
        else r"stable residuals are (\S+) on CPU and (\S+) on CUDA\."
    )
    expected = tuple(
        repr(
            _scientific_acceptance_outcome(
                backend, "float32", weights, bundle=NUM004_PRECISION_BUNDLE
            )[1]
        )
        for backend in ("cpu", "gpu")
    )
    assert re.findall(claim, current) == [expected]
    # Unchanged stable values still need attribution to the current execution.
    assert f"audits/{NUM004_PRECISION_BUNDLE.name}/README.md" in sections[0]
