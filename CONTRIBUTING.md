# Contributing

HPFRACC is currently in pre-alpha development. Contributions should preserve the
v0.1 priority order:

1. Numerical correctness.
2. Numerical stability.
3. Differentiability.
4. Reproducibility and provenance.
5. Performance on CPU and single accelerator.

Public scientific functionality should include mathematical assumptions,
validation evidence, known failure modes, and a stability-tier designation.

## Reproducible development checks

From the repository root:

```bash
uv sync --locked --extra dev
uv run --frozen ruff check .
uv run --frozen python -m pytest
uv run --frozen mkdocs build --strict
uv run --frozen python -m benchmarks.numerical.validation_summary --check
```

When intentionally changing dependencies, edit `pyproject.toml`, run `uv lock`,
review the lockfile diff, then synchronize again with `--locked`. Ordinary checks
must not silently re-resolve project dependencies. Ruff is in the dev extra and lockfile.

CI pins uv 0.11.21 and uses these same locked/frozen commands. Its CPU test matrix
covers Python 3.11, 3.12 and 3.13; lint and strict docs are gating jobs. The separate
performance job remains non-gating because its timing result is machine-dependent.

Some numerical tests intentionally import after `pytest.importorskip` or precision
setup. Their narrow `E402` annotations preserve that ordering; other Ruff rules
remain enforced. Keep private compatibility re-exports explicit rather than
removing them as unused imports.

Use the [capability/evidence matrix](docs/validation/capabilities.md) to distinguish
implemented support from validated combinations. Preserve historical audit and
golden files; new evidence belongs in a new, revision-identified bundle.
