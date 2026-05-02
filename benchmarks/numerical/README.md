# Numerical Benchmarks

v0.1 benchmarks will record compile time, steady-state runtime, memory use,
sequence-length scaling, state-dimension scaling, and CPU versus single
accelerator behavior where available.

## Operator Validation Report

Phase 1 includes a deterministic validation report for the initial fractional
operators:

```bash
python -m benchmarks.numerical.operator_validation.report
```

The command writes CSV to standard output with columns for operator, case,
fractional order, grid size, timestep, maximum absolute error, and reference.

This report is a correctness artifact, not a performance benchmark.

## Operator Scaling Smoke Benchmark

Run the lightweight operator scaling benchmark with:

```bash
python -m benchmarks.numerical.operator_scaling
```

The command writes CSV to standard output with operator name, time-grid size,
state dimension, fractional order, average seconds, and output shape.

This benchmark is intended to establish a small local baseline for the current
full-history implementation. It is not a general performance claim.

Example with custom sizes:

```bash
python -m benchmarks.numerical.operator_scaling \
  --n-steps 32 64 128 \
  --state-dims 1 8 \
  --repeats 5
```
