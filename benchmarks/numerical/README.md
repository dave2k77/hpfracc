# Numerical Benchmarks

v0.1 benchmarks will record compile time, steady-state runtime, memory use,
sequence-length scaling, state-dimension scaling, and CPU versus single
accelerator behavior where available.

## Operator Validation Report

Phase 1 includes a deterministic validation report for the initial fractional
operators:

```bash
uv run python -m benchmarks.numerical.operator_validation.report
```

The command writes CSV to standard output with columns for operator, case,
fractional order, grid size, timestep, maximum absolute error, and reference.

This report is a correctness artifact, not a performance benchmark.

## Phase 2 Solver Validation Report

Run the deterministic Caputo FDE solver validation report with:

```bash
uv run python -m benchmarks.numerical.solver_validation.report
```

The command writes CSV rows for the scalar linear Caputo FDE against a
truncated Mittag-Leffler reference.

## Aggregate Validation Summary

Run the aggregate validation summary with:

```bash
uv run python -m benchmarks.numerical.validation_summary --check
```

The command preserves the baseline operator convergence, solver refinement,
finite-difference gradient, and stability rows, and appends fixed Phase A
core-hardening cases. CSV columns remain `area,case,metric,value,passed,details`.
With `--check`, a failed or empty report exits with status 1 after writing CSV;
without it, the command retains report-only behavior (status 0 even for failing
rows). Unexpected execution errors still fail normally.

Run the focused hardening report separately:

```bash
uv run python -m benchmarks.numerical.core_hardening --check
```

It covers individual graded history weights and an isolated Caputo history
increment against 70-digit references, float32 nonuniform-solver refinement,
implicit nonlinear convergence/failure reporting, and adaptive completion,
budget/non-finite/precision failures and JIT status. Each row records its case
configuration and pass criterion. Deliberately failing solver cases pass the
validation row only when the expected failure is correctly reported.

The hardening cases are fixed float32 workloads and do not change JAX's global
precision setting. Aggregate `--order`, `--n-steps`, and `--gradient-tolerance`
affect the existing baseline checks, not these fixed cases. Reports remain
case-specific evidence, not a complete Phase A release dossier.

Run finite-difference gradient checks directly with:

```bash
uv run python -m benchmarks.numerical.gradient_checks
```

Run stability checks directly with:

```bash
uv run python -m benchmarks.numerical.stability
```

## Adaptive Calibration Audit

```bash
uv run --frozen python -m benchmarks.numerical.adaptive_calibration --check
uv run --frozen python -m benchmarks.numerical.adaptive_calibration --first-step --check
JAX_ENABLE_X64=1 uv run --frozen python -m benchmarks.numerical.adaptive_calibration --dtype float64 --check
```

The sweep measures errors against independent continuous solutions, while
`--first-step` checks quadratic-forcing panel identities and records an adaptive
acceptance counterexample. `--check` requires complete finite sweep measurements
or agreement with the panel oracles, **not** error/tolerance at most one.
Use `--output-dir` to save CSV plus source/environment provenance JSON.
See [the calibration audit](../../docs/validation/adaptive-calibration.md) for
the derivation, measured limitations, and remaining correction gate. This report
does not alter the aggregate CSV schema or existing cases.

Run the retained-coarse startup-scaling follow-up with:

```bash
JAX_ENABLE_X64=1 uv run --frozen python -m benchmarks.numerical.adaptive_startup --check --output-dir docs/validation/audits/adaptive-startup
```

This checks selected asymptotic coefficients and powers in float64 and records
a cubic pair-indicator blind spot. JSON preserves raw step-size measurements and
source/environment provenance; a passing report does not validate the controller
as an accuracy bound. See [startup analysis](../../docs/validation/adaptive-startup.md).

## Experimental Subdivision Comparison

```bash
uv run --frozen python -m benchmarks.numerical.subdivision_comparison --check --output-dir docs/validation/audits/subdivision-prototype
JAX_ENABLE_X64=1 uv run --frozen python -m benchmarks.numerical.subdivision_comparison --dtype float64 --check --output-dir docs/validation/audits/subdivision-prototype
```

This compares the default and opt-in subdivision modes against continuous
references. The checked command requires completed finite measurements, not
global error below tolerance. See [prototype scope and costs](../../docs/validation/adaptive-subdivision.md).
Historical audit bundles remain unchanged; comparison provenance identifies the
new implementation separately.

Run the nonlinear/long-span follow-up with explicit work-proxy accounting:

```bash
JAX_PLATFORMS=cpu uv run --frozen python -m benchmarks.numerical.subdivision_stress --max-steps 2048
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --frozen python -m benchmarks.numerical.subdivision_stress --dtype float64 --max-steps 2048
```

Use `--output-dir` to save CSV and JSON containing all approximately work-matched
pairs. The proxy counts padded PECE history-weight slots, not runtime. Add
`--check` to require complete trajectories and a pair per case; the archived
stress sweeps return 1 because several subdivision runs exhaust the budget.
Failures are part of the evidence. See the [stress follow-up](../../docs/validation/adaptive-subdivision.md#nonlinear-and-long-span-stress-follow-up).

## Synchronized Subdivision Runtime

Run these sequentially, without concurrent numerical workloads:

```bash
JAX_PLATFORMS=cpu uv run --frozen python -m benchmarks.numerical.subdivision_runtime --repeats 7 --output-dir docs/validation/audits/subdivision-runtime
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --frozen python -m benchmarks.numerical.subdivision_runtime --dtype float64 --repeats 7 --output-dir docs/validation/audits/subdivision-runtime
```

The report separates lowering/compilation, warm-up, and synchronized compiled
execution. CSV records medians/minima/maxima; JSON retains every sample and all
eligible approximately runtime-matched pairs. `--check` additionally requires
every solve to complete, so the archived budget failures do not pass it.
See the [runtime audit](../../docs/validation/adaptive-runtime.md) for actual
measurements and the distinction between work-proxy and runtime matching.
Use a new output directory when recording a new experiment.

The fixed `--sweep dense` option expands the tolerance grid while preserving the
default `--sweep coarse` behavior. It does not change the five-repeat default;
the locally predeclared dense experiment explicitly uses `--repeats 7`,
`--max-steps 2048`, and `--seed 1729` for both precisions. See the
[dense runtime protocol and report](../../docs/validation/adaptive-runtime-dense.md)
for serial commands and separate matrix, completion, and pair-coverage checks.

## Baseline Benchmarks

Run the lightweight operator scaling benchmark with:

```bash
uv run python -m benchmarks.numerical.operator_scaling
```

The command writes CSV to standard output with operator name, time-grid size,
state dimension, fractional order, average seconds, and output shape.

This benchmark is intended to establish a small local baseline for the current
full-history implementation. It is not a general performance claim.

Example with custom sizes:

```bash
uv run python -m benchmarks.numerical.operator_scaling \
  --n-steps 32 64 128 \
  --state-dims 1 8 \
  --repeats 5
```

Run the broader CPU-oriented baseline benchmark with:

```bash
uv run python -m benchmarks.numerical.baseline
```

This records operator and solver timing rows with backend and platform context.
