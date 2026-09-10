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

Gradients with respect to fractional order `alpha` have case-specific
finite-difference validation. See the validation status page for the supported
paths; this is not a blanket guarantee for every history strategy or regime.

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

## Aggregate Validation Harness

Run the aggregate baseline and Phase A validation summary with:

```bash
uv run python -m benchmarks.numerical.validation_summary --check
```

The summary covers:

- Operator convergence checks.
- Solver refinement checks.
- Finite-difference gradient checks.
- Basic stability checks.
- Float32 graded-mesh history precision and nonuniform-solver refinement.
- Implicit nonlinear residual convergence and expected-failure reporting.
- Adaptive completion, budget exhaustion, non-finite and step-size/precision
  failures, including JIT-accessible status.

The existing CSV schema and baseline case names are preserved. `--check` writes
the report and exits 1 if any row fails or the report is empty; omitting the flag
retains report-only behavior. A solver-failure detection row has `passed=True`
when the intentionally failing solve reports the expected failure, not when that
trajectory should be accepted for research use.

Run only the new fixed cases with:

```bash
uv run python -m benchmarks.numerical.core_hardening --check
```

These cases record their orders, grids, precision, references, tolerances, and
expected statuses in `details`. They intentionally do not inherit aggregate
`--order`, `--n-steps`, or `--gradient-tolerance` overrides:

- Graded weights/Caputo history: 160 nodes, grading exponent 4, order 0.4,
  float32 execution; 70-digit mpmath references on the actual stored nodes;
  relative-error tolerance `5e-6` and positive solver weights.
- Nonuniform decay: order 0.4, rate -1, 40/80/160 nodes, grading exponent 4;
  a host-float64, 100-term Mittag-Leffler series on mild arguments; monotone
  refinement and finest max error at most `5e-4`.
- Implicit cubic dynamics: order 0.7, rate -2, one step of 0.5; eight Newton
  iterations must meet the residual criterion and one iteration must report
  failure, including under JIT. Residual tolerances are `1e-6`/`1e-8`. The report
  independently reconstructs the first-step residual from the finite returned
  trajectory using host-float64 arithmetic, rather than trusting status fields.
  Both residuals must lie on the expected side of the threshold of one, and the
  reported residual must be nonnegative. Agreement is checked with `math.isclose`
  using `rel_tol=1e-5` and `abs_tol=1.0` in scaled-residual units to accommodate
  float32 gamma/arithmetic roundoff; both values and these comparison tolerances
  are recorded. This comparison allowance does not relax the independent
  convergence threshold.
- Adaptive zero-drift controls and a deliberately non-finite drift verify
  expected completion/failure flags, counts, retained-prefix masks, and states.
  Separate controls cover an unrepresentable midpoint, lossy integer time bounds,
  and eager/JIT agreement.

The hardening workload stays float32 even when the caller has enabled x64; the
report does not change that global setting. This extends the aggregate evidence
but does not replace the complete unit/property/regression/performance suites,
validate every Phase A feature, or establish general tolerance calibration.

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
