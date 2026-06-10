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
- Riemann-Liouville correctness against the analytic RL power-law reference,
  including the constant case `t**(-alpha) / Gamma(1 - alpha)` that distinguishes
  RL from Caputo.
- JIT compatibility.
- Gradients with respect to input samples.

Gradients with respect to fractional order `alpha` remain provisional until a
dedicated validation suite is added.

## Validation Report Commands

Run the Phase 1 operator validation report with:

```bash
uv run python -m benchmarks.numerical.operator_validation.report
```

The command prints CSV rows for:

- Caputo derivative of constants.
- Caputo derivative of `t**power` against the analytic power-law reference.
- Riemann-Liouville derivative of `t**power` against the analytic RL reference.
- Riemann-Liouville derivative of a constant against `t**(-alpha) / Gamma(1 -
  alpha)`.

Use `--order`, `--power`, and `--n-steps` to vary the validation grid:

```bash
uv run python -m benchmarks.numerical.operator_validation.report \
  --order 0.4 \
  --power 2.0 \
  --n-steps 21 41 81 161
```

Run the Phase 2 solver validation report with:

```bash
uv run python -m benchmarks.numerical.solver_validation.report
```

The command validates the scalar linear Caputo FDE against a truncated
Mittag-Leffler reference and records timestep-refinement errors.

## Phase 3 Validation Harness

Run the aggregate Phase 3 validation summary with:

```bash
uv run python -m benchmarks.numerical.validation_summary
```

The summary covers:

- Operator convergence checks.
- Solver refinement checks.
- Finite-difference gradient checks.
- Basic stability checks.

Run the component commands directly when detailed rows are needed:

```bash
uv run python -m benchmarks.numerical.gradient_checks
uv run python -m benchmarks.numerical.stability
```

## Scaling Smoke Benchmark

Run the lightweight scaling benchmark with:

```bash
uv run python -m benchmarks.numerical.operator_scaling
```

The benchmark records small CPU-oriented timing rows for Caputo and
Grunwald-Letnikov operators over selected time-grid and state dimensions. These
rows are local baselines for regression checking, not broad performance claims.

Run the broader baseline benchmark with:

```bash
uv run python -m benchmarks.numerical.baseline
```

The baseline benchmark records both operator and solver timing rows with JAX
backend and platform context.

## Provenance Context

Validation reports write plain CSV to stdout so their output can be piped directly
into downstream tools. Runtime provenance is therefore kept outside the CSV rows
instead of being emitted as comment headers that would break strict CSV parsers.

See [Validation Status](status.md) for the current v0.1 alpha claim boundary: what
is validated, what remains provisional, and how to interpret passing rows.

Capture provenance next to a report with:

```bash
uv run python - <<'PY'
import json
import hpfracc as hp

print(json.dumps(hp.config.current_provenance().to_dict(), indent=2))
PY
```

For committed release artifacts, save the provenance JSON alongside the CSV or
benchmark output and record which command produced each file.
