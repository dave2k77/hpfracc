# Validation Methodology

Validation is part of the scientific contract. v0.1 validation covers analytic
operator identities, solver refinement, finite-difference gradient checks, JIT
consistency, and baseline performance measurements.

## Phase 1 Operator Validation

The first operator validation layer covers:

- Caputo derivative of constants.
- Caputo derivative of linear functions.
- Caputo power-law references for `t**beta`.
- Refinement behavior for smooth polynomial functions.
- Riemann-Liouville and Grunwald-Letnikov consistency under the v0.1 GL
  baseline discretisation.
- JIT compatibility.
- Gradients with respect to input samples.

Gradients with respect to fractional order `alpha` remain provisional until a
dedicated validation suite is added.

## Validation Report Command

Run the Phase 1 operator validation report with:

```bash
python -m benchmarks.numerical.operator_validation.report
```

The command prints CSV rows for:

- Caputo derivative of constants.
- Caputo derivative of `t**power` against the analytic power-law reference.
- Riemann-Liouville consistency with the v0.1 Grunwald-Letnikov baseline.

Use `--order`, `--power`, and `--n-steps` to vary the validation grid:

```bash
python -m benchmarks.numerical.operator_validation.report \
  --order 0.4 \
  --power 2.0 \
  --n-steps 21 41 81 161
```

## Scaling Smoke Benchmark

Run the lightweight scaling benchmark with:

```bash
python -m benchmarks.numerical.operator_scaling
```

The benchmark records small CPU-oriented timing rows for Caputo and
Grunwald-Letnikov operators over selected time-grid and state dimensions. These
rows are local baselines for regression checking, not broad performance claims.
