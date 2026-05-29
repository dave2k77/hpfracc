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

## Phase 3 Validation Summary

Run the aggregate validation summary with:

```bash
uv run python -m benchmarks.numerical.validation_summary
```

The command writes repeatable CSV summary rows for operator convergence, solver
refinement, finite-difference gradient checks, and stability checks.

Run finite-difference gradient checks directly with:

```bash
uv run python -m benchmarks.numerical.gradient_checks
```

Run stability checks directly with:

```bash
uv run python -m benchmarks.numerical.stability
```

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
