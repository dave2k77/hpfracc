# v0.1 Alpha Release Checklist

Status: Reusable candidate template; not an executed release record.

Latest tagged release: `0.1.0a1` (2026-06-26).

Goal: define the minimum checks required before tagging the next reviewed
pre-alpha candidate. No next version or release date is selected here. The
current Phase A hardening is unreleased; passing local checks does not close
Phase A or authorize a tag. See the [roadmap](roadmap.md#current-development-state)
for remaining milestone gates. HPFRACC remains research software only; this
checklist does not imply clinical, diagnostic, or subject-specific validity.

Run commands from the repository root with the project `uv` environment.

Leave this template unchecked. Copy it into the candidate's evidence bundle and
record outcomes, environment, and exact source revision there. Do not relabel or
overwrite the historical `docs/validation/reports/v0.1.0a0/` artifacts as evidence
for a newer release.

## 1. Repository State

- [ ] Confirm the working tree contains only intended release changes.

```bash
git status --short
```

- [ ] Review staged and unstaged changes before tagging.

```bash
git diff --check
git diff --stat
git diff --cached --stat
```

- [ ] Confirm the active branch is the intended release branch.

```bash
git branch --show-current
```

## 2. Environment Reproducibility

- [ ] Synchronize the development environment from the lockfile.

```bash
uv sync --locked --extra dev
```

- [ ] Record the Python, package, and backend versions used for validation.
- [ ] Record JAX/jaxlib versions, device, dtype/x64 settings, OS, and uv version.
  A commit hash alone does not identify a dirty development tree: record dirty
  state and source changes for development runs, and use reviewed clean source
  for final candidate evidence.

```bash
uv run python - <<'PY'
import json
import hpfracc as hp

print(json.dumps(hp.config.current_provenance().to_dict(), indent=2))
PY
```

CI now pins uv 0.11.21 and uses the lockfile for its Python 3.11–3.13 CPU matrix,
lint and docs. Ruff is declared in the dev extra. Record the candidate's actual
tool/environment versions and executed CI results; configuration alone is not
release sign-off. The [capability matrix](../validation/capabilities.md) identifies
remaining scientific and hardware evidence gaps.

## 3. Tests and Lint

- [ ] Run the documentation guards and default suite, including property and
  golden-regression tests. Performance tests are deselected by default.

```bash
uv run --frozen python -m pytest tests/unit/test_docs_consistency.py
uv run --frozen python -m pytest tests/unit/test_development_status_docs.py
uv run --frozen python -m pytest
```

- [ ] Record the separate machine-dependent performance result. CI treats this
  timing job as non-gating; it is not numerical correctness or accelerator proof.

```bash
uv run --frozen python -m pytest -m performance
```

- [ ] Run repository lint with the candidate's locked development Ruff and
  record its version. Do not describe changed-file lint as a clean full-repo run.

```bash
uv run --frozen ruff check .
```

- [ ] If failures occur, fix the underlying code or contract before continuing.
  Do not mark the release candidate ready with expected failures.

## 4. Documentation Build

- [ ] Build documentation in strict mode.

```bash
uv run --frozen mkdocs build --strict
```

- [ ] Confirm any warnings are understood and are not caused by HPFRACC docs.
  The current MkDocs Material warning about future MkDocs 2.0 compatibility is
  upstream and not release-blocking by itself.

## 5. Numerical Validation Reports

- [ ] Run the aggregate and focused hardening reports as executable gates.

```bash
uv run --frozen python -m benchmarks.numerical.validation_summary --check
uv run --frozen python -m benchmarks.numerical.core_hardening --check
uv run --frozen python -m benchmarks.numerical.history_validation --check
```

The `--check` flag makes failed or empty reports exit nonzero; report-only mode
does not enforce this gate. A passing diagnostic-failure case means the expected
failure was detected, not that its simulation succeeded. The aggregate is not
the entire Phase A evidence matrix.

- [ ] Run operator validation.

```bash
uv run python -m benchmarks.numerical.operator_validation.report
```

- [ ] Run solver validation.

```bash
uv run python -m benchmarks.numerical.solver_validation.report
```

- [ ] Review output for failures, warnings, or claims that exceed the documented
  v0.1 validation scope.
- [ ] Review separate convergence, gradient, history-comparison, vector-order,
  and solver test evidence referenced by [validation status](../validation/status.md).
  Reports without a `--check` option require inspection of their rows, not just
  successful process exit.
- [ ] Save CSV output and companion provenance JSON in a new candidate-specific
  bundle. Preserve existing case IDs/columns and identify fixed-case
  configurations; see [methodology](../validation/methodology.md).

## 6. Baseline Benchmark

- [ ] Run the baseline benchmark.

```bash
uv run python -m benchmarks.numerical.baseline
```

- [ ] Record enough context for any reported timings: Python version, JAX
  version, backend, device, operating system, and commit hash.
- [ ] Avoid broad performance claims unless the benchmark context is included.

## 7. Example Smoke Tests

- [ ] Run every documented fractional ODE example.

```bash
uv run python examples/fractional_ode/caputo_operator.py
uv run python examples/fractional_ode/caputo_solver.py
uv run python examples/fractional_ode/train_neural_fode.py
uv run python examples/fractional_ode/probabilistic_calibration.py
```

- [ ] Confirm each example completes without errors.
- [ ] Confirm the examples README describes the command purpose and expected
  high-level output.

## 8. Version, Changelog, and Citation Metadata

- [ ] Confirm the package version is consistent.

```bash
uv run python - <<'PY'
import hpfracc
print(hpfracc.__version__)
PY
```

- [ ] Check `pyproject.toml` version metadata.
- [ ] Check `src/hpfracc/_version.py`.
- [ ] Check the project entry in `uv.lock` after an intentional version update.
- [ ] Check `CHANGELOG.md` for a current alpha-release entry.
- [ ] Check `CITATION.cff` if present.
- [ ] Add a release date only when the tag date is known.

### Distribution verification

- [ ] Build wheel and source distributions from the reviewed candidate.

```bash
uv build
```

- [ ] Verify candidate filenames/version and package contents. Install the wheel
  in an isolated environment outside the source checkout and smoke-test import,
  version, and a numerical example; do not rely only on an editable installation.
- [ ] Record packaging results with candidate evidence. Building is not
  publishing; tagging, uploading, and publishing require an explicit release
  decision.

## 9. Research-Use Disclaimer and API Stability

- [ ] Confirm the README and docs state that HPFRACC is research software, not
  clinical or diagnostic software.
- [ ] Confirm `docs/api/contract.md` includes the public API stability tiers.
- [ ] Confirm experimental namespaces are labeled as experimental.
- [ ] Confirm future brain-model documentation avoids biological, clinical,
  diagnostic, or subject-specific claims.

## 10. Final Candidate Gate

Run the candidate gate from a clean checkout or a reviewed release
candidate branch:

```bash
uv sync --locked --extra dev
uv run --frozen ruff check .
uv run --frozen python -m pytest
uv run --frozen mkdocs build --strict
uv run --frozen python -m benchmarks.numerical.validation_summary --check
uv run --frozen python -m benchmarks.numerical.core_hardening --check
uv run --frozen python -m benchmarks.numerical.history_validation --check
uv run --frozen python -m benchmarks.numerical.operator_validation.report
uv run --frozen python -m benchmarks.numerical.solver_validation.report
uv run --frozen python -m benchmarks.numerical.baseline
git diff --check
git status --short
```

The v0.1 alpha candidate is ready for tagging only when:

- [ ] all gating commands above pass and report rows have been reviewed;
- [ ] separate performance results and candidate lint results are recorded;
- [ ] example smoke tests pass;
- [ ] distribution build and isolated wheel checks pass;
- [ ] release notes/changelog match the actual code and validation scope;
- [ ] provenance or validation-context gaps are either fixed or explicitly
  documented;
- [ ] the working tree is clean except for intentional generated release
  artifacts.

A candidate may ship a documented subset of Phase A without declaring the whole
phase complete. Any Phase A closure additionally requires the roadmap exit
criteria and explicitly reviewed deferrals; this checklist does not waive them.
