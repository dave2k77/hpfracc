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
- Selected scaled-tail SOE analytic, fitted-sum and order-gradient checks, plus
  stable uniform coefficients and explicit/implicit solver checks.
- A conditional full-history prescribed-forcing bound against an analytic solution.

The existing CSV schema and baseline case names are preserved. `--check` writes
the report and exits 1 if any row fails or the report is empty; omitting the flag
retains report-only behavior. A solver-failure detection row has `passed=True`
when the intentionally failing solve reports the expected failure, not when that
trajectory should be accepted for research use.

There are now 28 fixed rows: eight baseline, twelve original core-hardening and
eight history/bounded-forcing extension rows. See the [capability matrix](capabilities.md) for
which combinations still rely on their dedicated tests/audits.

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

### Fixed history-extension cases

```bash
uv run --frozen python -m benchmarks.numerical.history_validation --check
```

These eight rows also stay float32 when x64 is enabled and do not change the
caller's precision setting. They retain the shared CSV schema and fixed-case
configuration semantics:

| Case | Fixed configuration / check |
| --- | --- |
| Scaled-tail Caputo polynomial | Order 0.5, 257 nodes, 12 poles, `t^2`; analytic max error at most `5e-4`. |
| Scaled-tail RL constant | Same order/nodes/poles; positive response and analytic relative error at most `0.02` for `t>=0.5`. |
| SOE recursive fitted sum | 33 nodes, 8 poles, two signals; independent host float64 convolution of the actual fitted coefficients, max difference at most `5e-5`. This tests application, not fit optimality. |
| SOE order gradient | Compiled Caputo mean on `t^2`, 33 nodes, 8 poles, order 0.5; centered difference step `1e-3`, absolute gradient error at most `2e-3`, and nonzero reference sensitivity. |
| Stable uniform coefficients | Step 256, order 0.5; positive predictor/corrector history weights, correct future masks and max relative error at most `5e-6` against independent panel quadrature. |
| Stable explicit solver | Compiled constant forcing, 65 nodes, order 0.5, trailing shape `(2,3)`; analytic max error at most `3e-6`. |
| Stable implicit solver | Half-order decay at strength 1, 257 nodes, eight Newton updates; reported convergence and independent scaled residual at most one using `rtol=1e-5, atol=1e-7`, plus continuous max error at most `1e-3`. |
| Bounded prescribed quadratic forcing | Half order, `q=t^2`, curvature 2, `atol=1e-3`, `rtol=0`, first step 1, 128 attempts; completed with rejections, accepted envelopes at most tolerance, analytic max error at most tolerance and enclosed up to `3e-6` roundoff slack. |

Metadata/dtype checks and corruption tests ensure these cases are sensitive to
wrong values, lost order gradients, a direct-weight substitution, and false
implicit success and zeroed forcing envelopes. These are executable numerical gates, unlike measurement-only
audits whose `--check` can accept large but finite approximation errors.

The separate [bounded-forcing audit](bounded-forcing.md) evaluates 60 predeclared
configurations in both precisions, with independent continuous references and
panel-envelope reconstruction. Its checked command requires completion and
selected-case accuracy. The mathematical bound is conditional on supplied
global C2 curvature and excludes roundoff; its audit slack is not a solver bound.

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
