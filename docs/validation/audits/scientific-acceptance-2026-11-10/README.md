# Selected implicit scientific-acceptance development evidence

Status: Scoped local development evidence; not Phase A or release sign-off.

The [frozen protocol](../../scientific-acceptance.md) separates trajectory
acceptance from measurement integrity. Each execution measures four
required-success cases and two explicit negative controls. This bundle keeps
all eight CPU/CUDA × float32/float64 × direct/stable runs, including the two
nonzero gate exits. Existing historical audits are not overwritten.

## Actual outcomes

| Backend | Dtype | Weights | CLI exit | Weak-decay independent scaled residual |
| --- | --- | --- | --- | --- |
| cpu | float32 | direct | 1 | 12.4539 |
| cpu | float32 | stable | 0 | 0.052358 |
| cpu | float64 | direct | 0 | 0.00471574 |
| cpu | float64 | stable | 0 | 1.2035e-05 |
| gpu | float32 | direct | 1 | 8.49763 |
| gpu | float32 | stable | 0 | 0.0164322 |
| gpu | float64 | direct | 0 | 0.00242797 |
| gpu | float64 | stable | 0 | 5.26731e-06 |

Threshold: independent maximum scaled residual <= 1. The required weak-decay
case's continuous error is below its 1e-3 budget in both direct float32 runs;
Newton convergence and small trajectory error nevertheless do not establish
independent algebraic convergence. These are retained numerical limitations,
not report-execution errors. Stable evaluation is an explicit configuration,
not an automatic replacement for the direct default (tracker NUM-004).

Across 48 measured configurations, 30 required-success trajectories are accepted,
two required-success trajectories fail, and 16 explicit negative controls are
correctly rejected. Six complete reports pass their scoped policy; two fail it.
A verified rejection is never counted as an accepted trajectory.

The coarse stiff float32 direct control has both independent-residual and
continuous-accuracy rejection reasons. The float64 control demonstrates clean
algebraic convergence without continuous accuracy; do not substitute the
float32 direct case as evidence of that cleaner separation.

## Artifacts and replay checks

- `executions.json` records the eight exact commands and actual process exits.
- `scientific_acceptance_*.csv` contains measured values, explicit roles,
  independent error/residual decisions and separate acceptance/policy flags.
- Companion JSON contains the full authoritative manifest and digest, source
  hashes, backend/device/version/effective-x64 context, CSV digest and raw
  artifact digests. Sources were checked unchanged across each execution.
- `*_case_*.npz` preserves all 48 raw time/state/input arrays and per-step Newton
  diagnostics. Read with `numpy.load(..., allow_pickle=False)`. Nonfinite arrays
  are preserved rather than replaced with fabricated finite values.
- `verification.json` records host reconstruction of every raw trajectory's
  continuous-reference errors and independent residuals, comparison with every
  reported row/decision, 61 matching source-input hashes per run, manifest/CSV/
  NPZ digests and actual command exits. This is artifact/measurement integrity,
  not universal scientific acceptance or cryptographic execution attestation.
- Focused final JUnit: 81 tests on CPU; 103 tests on CUDA including the existing
  implicit-validation and uniform-weight-report regressions. Both processes
  exited zero. Historical red JUnit files preserve selected TDD failures; they
  do not contain full historical test-source snapshots. These are `*.xml.gz`
  because raw pytest failure text contains trailing whitespace: compression
  preserves the original evidence bytes without a Git whitespace exception.
  `historical-red-artifacts.json` records raw and compressed digests.
- `full-cpu.xml` and `full-suite-verification.json` record the replacement full
  CPU run: 818 passed, one performance case deselected, process exit zero.
  The ten later documentation-consistency checks are not counted in that run;
  21 documentation/status guards pass separately. No full CUDA suite or new
  remote CI run is claimed for this increment.
- `type-guard-regression.json` records the later four-failure/81-pass CPU
  configuration-type correction and its raw/compressed red digests. Python
  value equality alone could treat `True` as `1` or `129.0` as `129`; the gate
  now requires canonical types too. `pre-type-guard-evidence.tar.gz` preserves
  the superseded reports and exact benchmark/test-source snapshots, with its
  digest in the companion JSON. Current reports were rerun after that correction;
  they are not relabelled earlier measurements. The earlier full CPU process
  was stopped as superseded; it is not a completed final verification result.
- `weight-probe-cpu.json` compares selected direct/stable dynamic-JIT weights
  to the existing 70-digit interval-integral oracle. This supports NUM-004's
  working-precision diagnosis; it is not a global solver error bound.
- `review-initial.json` retains the failed review of stale CUDA citations.
  The corrected outcome table is now checked directly against the final JSON,
  with current full-precision tracker citations checked too; historical values
  remain in their original archive rather than being overwritten.
- `docs-consistency-regression.json` records four red failures among ten new
  checks, followed by 21 passing documentation/status guards, with the new
  checker source hash and preserved red/green JUnit digests.

New module/unit code ran from the development checkout, not an isolated release
wheel. General adaptive calibration, further nonlinear/stiff/order/horizon
coverage and default direct-weight precision limits remain separate work.

## Reproduce

For each dtype and evaluation mode, from the repository root:

```bash
JAX_PLATFORMS=cpu uvx --from 'uv==0.11.21' uv run --frozen --extra dev python -m benchmarks.numerical.scientific_acceptance --dtype float32 --weight-evaluation stable --check --output-dir path/to/new-evidence
UV_PROJECT_ENVIRONMENT="$HOME/.local/share/hpfracc/venv-cuda13" JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uvx --from 'uv==0.11.21' uv run --frozen --extra dev --extra cuda13 python -m benchmarks.numerical.scientific_acceptance --dtype float32 --weight-evaluation stable --check --output-dir path/to/new-evidence
```

Use a new output directory when preserving evidence. Substitute `float64` and/or
`direct` to reproduce the other manifest variants. Direct float32 `--check`
currently exits one on both backends; this failure is intentional gate behavior
for an actual unacceptable required-success measurement, not a successful solve.
The command scopes precision per dtype; no global x64 override is needed.
