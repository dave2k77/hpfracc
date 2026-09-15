# Consolidation checkpoint: 2026-09-15

Status: Locally verified consolidation on top of numerical checkpoint
`e19f0c117a01178184efc49b2b9c6bb9ed8e7753`. Phase A remains active.

## Numerical checkpoint

The accumulated native-convolution remediation, recursive SOE implementation,
implicit validation, stable uniform weights, tests and evidence were committed as:

```text
e19f0c1 feat(numerics): harden recursive history and uniform solver weights
```

Historical audit bundles remain snapshots with their original configurations and
source digests. The new consolidation is not a relabeling of those measurements
as clean-candidate release evidence.

## Consolidation delivered

- Ruff 0.16.7 is declared in the dev extra and locked in `uv.lock`. Review
  of the lock diff showed the added development tool without numerical dependency
  upgrades.
- CI pins uv 0.11.21, synchronizes with `--locked --extra dev`, and uses frozen
  commands. The configured CPU matrix is Python 3.11/3.12/3.13, with gating lint,
  strict docs and aggregate validation. Performance remains a separate non-gating
  job. Runtime context is printed in the test jobs.
- Repository-wide lint passes. Import guards retain narrow, intentional `E402`
  annotations; private compatibility re-exports remain available. Formatting and
  import fixes do not change the numerical algorithms.
- Package and metadata imports no longer load JAX. A fresh-interpreter test blocks
  JAX imports and exercises metadata construction; explicit runtime provenance
  requests JAX lazily and records the x64-enabled setting.
- The aggregate expands from 20 to 27 rows, preserving the original schema and
  case names. Seven fixed SOE/stable-weight cases enforce numerical tolerances,
  with corruption tests for values, gradients, coefficient selection and implicit
  residuals. Larger measurement audits retain their distinct check semantics.
- The [capability/evidence matrix](../validation/capabilities.md) records supported
  combinations, tested gradients/precision, and explicit gaps. Developer and
  candidate-checklist commands now follow the locked workflow.
- The README simulation/loss/calibration example now uses a consistent `rate`
  parameter dictionary and passed an executable smoke check.

## Local verification

Runtime context recorded at `2026-09-15T07:02:08.419649+00:00`: Python 3.13.13,
JAX 0.10.1, Windows 11 (`Windows-11-10.0.26200-SP0`), CPU `cpu:0`, uv 0.11.21,
Ruff 0.16.7, and default x64 disabled. Scientific tests enable x64 within their
declared scopes.

| Check | Result |
| --- | --- |
| `uv sync --locked --extra dev` | Passed; 43 packages checked. |
| `uv run --frozen python -m pytest` | **662 passed**, 1 performance test deselected, in 767.79 s; confirmed exit 0. |
| `uv run --frozen python -m benchmarks.numerical.validation_summary --check` | **27/27 rows passed**. |
| `uv run --frozen python -m benchmarks.numerical.history_validation --check` | **7/7 rows passed**. |
| `uv run --frozen ruff check .` | Passed for the repository. |
| `uv run --frozen mkdocs build --strict` | Passed. |
| Documentation/development-status/provenance guards | 14 passed. |
| README workflow smoke | Zero self-trajectory loss; calibration selected the expected rate near -0.8. |
| `git diff --check` | Passed. |

The CI configuration has been updated, but remote jobs have not been executed
for these changes. Local verification was Python 3.13 on Windows CPU, not the
entire configured Linux/Python matrix. No release, push or Phase A closure was
performed. The consolidation changes remain local and uncommitted.

## Remaining priorities

1. Define and validate an adaptive error-control contract beyond the existing
   step-doubling/subdivision heuristics.
2. Extend nonlinear/mixed-scale precision evidence and fill the capability matrix's
   transformation gaps, including dedicated FFT order-gradient finite differences.
3. Develop tolerance-driven SOE fit selection and broader reverse-mode/accelerator
   memory evidence.
4. Execute remote CI and capture a reviewed clean candidate's complete report,
   packaging and provenance dossier before release decisions.
