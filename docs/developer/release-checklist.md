# v0.1 Alpha Release Checklist

Status: Draft for Phase 6 release-readiness work.

Goal: define the minimum checks required before tagging a research-ready
`0.1.0a0` alpha candidate. HPFRACC remains research software only; this
checklist does not imply clinical, diagnostic, or subject-specific validity.

Run commands from the repository root with the project `uv` environment.

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
uv sync --extra dev
```

- [ ] Record the Python, package, and backend versions used for validation.

```bash
uv run python - <<'PY'
import platform
import jax
import hpfracc

print(f"python={platform.python_version()}")
print(f"hpfracc={hpfracc.__version__}")
print(f"jax={jax.__version__}")
print(f"jax_backend={jax.default_backend()}")
print(f"jax_devices={[str(device) for device in jax.devices()]}")
PY
```

## 3. Unit Tests

- [ ] Run the full unit test suite.

```bash
uv run python -m pytest
```

- [ ] If failures occur, fix the underlying code or contract before continuing.
  Do not mark the release candidate ready with expected failures.

## 4. Documentation Build

- [ ] Build documentation in strict mode.

```bash
uv run mkdocs build --strict
```

- [ ] Confirm any warnings are understood and are not caused by HPFRACC docs.
  The current MkDocs Material warning about future MkDocs 2.0 compatibility is
  upstream and not release-blocking by itself.

## 5. Numerical Validation Reports

- [ ] Run the validation summary.

```bash
uv run python -m benchmarks.numerical.validation_summary
```

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
- [ ] Check `CHANGELOG.md` for a current alpha-release entry.
- [ ] Check `CITATION.cff` if present.
- [ ] Add a release date only when the tag date is known.

## 9. Research-Use Disclaimer and API Stability

- [ ] Confirm the README and docs state that HPFRACC is research software, not
  clinical or diagnostic software.
- [ ] Confirm `docs/api/contract.md` includes the public API stability tiers.
- [ ] Confirm experimental namespaces are labeled as experimental.
- [ ] Confirm future brain-model documentation avoids biological, clinical,
  diagnostic, or subject-specific claims.

## 10. Final Candidate Gate

Run the complete Phase 6 gate from a clean checkout or a reviewed release
candidate branch:

```bash
uv run python -m pytest
uv run mkdocs build --strict
uv run python -m benchmarks.numerical.validation_summary
uv run python -m benchmarks.numerical.operator_validation.report
uv run python -m benchmarks.numerical.solver_validation.report
uv run python -m benchmarks.numerical.baseline
git diff --check
git status --short
```

The v0.1 alpha candidate is ready for tagging only when:

- [ ] all commands above pass;
- [ ] example smoke tests pass;
- [ ] release notes/changelog match the actual code and validation scope;
- [ ] provenance or validation-context gaps are either fixed or explicitly
  documented;
- [ ] the working tree is clean except for intentional generated release
  artifacts.
