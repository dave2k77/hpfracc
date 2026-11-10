# Selected implicit scientific-acceptance policy

Status: Protocol frozen before measurement; selected CPU/CUDA reports rerun after
the NUM-004 arithmetic repair. Local full suites are verified; independent
review, remote multi-version CI and release gates are recorded separately.
This is a scoped Phase A policy, not phase closure, a universal error guarantee
or release sign-off.

## Purpose and separation

`benchmarks.numerical.implicit_validation --check` retains its established
measurement-integrity semantics, including coherent failed integrations. The
separate `benchmarks.numerical.scientific_acceptance` command evaluates selected
trajectories against independent continuous-reference errors and reconstructed
Volterra residuals. It must not reinterpret the original audit's green exit.

A policy pass for an expected rejection is not a successful trajectory. Reports
must show requirement, actual rejection reasons, scientific `accepted` and
`policy_passed` separately.

## Frozen protocol: implicit-scientific-v1

Each execution declares `dtype` (`float32` or `float64`) and
`weight_evaluation` (`direct` or `stable`). The command defaults to the solver's
`direct` evaluation; `stable` is an explicit opt-in, never an automatic fallback.
Both modes face the same criteria. An unevidenced combination may fail the gate;
its thresholds must not be relaxed to turn that measurement green.

All cases use the full-history implicit solver on dyadic uniform nodes over
[0, 1]. The model, initial state, component scales, independent host continuous
reference and 32-point panel-quadrature residual reconstruction are those in
`benchmarks/numerical/implicit_validation.py`. Newton uses eight iterations
except the intentionally failing control; rtol is 1e-5/1e-10 for float32/float64,
with atol = 0.01 * rtol. Times, states and dynamic order/strength inputs must
have the declared dtype; x64 is scoped off/on outside JIT and restored afterward.

| ID | Model | Order | Strength | Nodes | Iterations | Requirement | Max dimensionless error budget |
| --- | --- | --- | --- | --- | --- | --- | --- |
| weak_decay | half_decay | 0.5 | 1 | 257 | 8 | required success | 1e-3 |
| cubic | cubic | 0.5 | 1 | 129 | 8 | required success | 1e-3 float32; 2e-5 float64 |
| coupled | coupled | 0.5 | 1 | 129 | 8 | required success | 1e-3 float32; 2e-5 float64 |
| scaled | scaled | 0.5 | 1 | 129 | 8 | required success | 1e-3 float32; 2e-5 float64 |
| coarse_stiff_decay | half_decay | 0.5 | 30 | 17 | 8 | expected inaccurate rejection | 1e-3; additionally error > 0.3 and negative minimum |
| nonfinite_cubic | cubic | 0.3 | 30 | 17 | 1 | expected nonfinite rejection | no accuracy value is permitted |

Dimensionless errors divide each component by its declared scale: (1) for scalar
cases, (1, 1) for coupled, (1e-3, 1e3) for scaled. Order 0.3 is compared to its
actual representation in the declared dtype, not a rounded nominal label.

### Budget evidence and limits

- The weak-decay 1e-3 error budget and independent residual threshold are already
  exercised together by `history_validation.stable_implicit_row()` for the
  stable float32 configuration. This evidence does not validate direct weights.
- The 129-node nonlinear float64 budget is grounded in
  `test_nonlinear_refinement_with_converged_correctors`, for exactly these three
  manufactured cases, order and strength. It is not general nonlinear stability.
- The nonlinear/mixed-scale float32 1e-3 budget is a new conservative, declared
  policy target, not an existing case-specific validation result. Passing the
  new measurements is required before claiming this selected coverage.
- Historical direct float32 audit residuals exceed one even when continuous
  error meets its budget. New normalization fixes do not establish that direct
  history-weight cancellation disappeared; measure and retain any failures.

## Decision rules

Every row first needs coherent measurements, configuration and diagnostics.
Missing/nonfinite/negative error or residual metrics, impossible counts,
contradictory convergence flags, inconsistent component-normalized maxima and
endpoint error above maximum invalidate a finite measurement. Nonfinite
trajectories require coherent failure/nonfinite counts and absent accuracy
metrics; invalid measurements cannot count as expected numerical rejections.

Required-success rows need all of:

- A finite trajectory and reported Newton convergence, zero failure/nonfinite
  counts, reported maximum scaled residual <= 1.
- Independent maximum scaled residual <= 1 and zero residual-flag disagreements.
- Maximum dimensionless continuous-reference error within the frozen budget.

Rejection reasons are retained together: reported nonconvergence, independent
residual failure and continuous inaccuracy must not hide one another. The coarse
stiff negative control must remain finite, Newton-converged, negative and have
error > 0.3. Its float32 execution may also fail the independent residual check;
only the clean float64 control establishes residual convergence without accuracy.
The nonfinite control must have failed correctors and nonfinite diagnostics,
with no fabricated accuracy value. Finite under-iteration is a separate real
classifier regression, not an additional required-success case.

## Coverage and evidence integrity

The authoritative case manifest is fixed independently of report rows. Require
exact case IDs, multiplicity, dtype, weight evaluation, order, strength, grid,
iterations, Newton tolerances, role and budget. Recompute decisions and booleans;
never trust serialized `accepted` or `policy_passed` fields. Empty, missing,
duplicated, extra or relabelled cases fail policy.

Save raw times, trajectories and per-step Newton diagnostics alongside measured
rows, full manifest and digests. Record actual backend/devices, effective dtype
and x64 mode, versions, dirty source state and hashes of relevant sources,
including `_special.py` and `_weights.py`. These support reproducibility and
numerical reinspection, not cryptographic proof that an execution occurred.
A row-only checker cannot authenticate a fully coherent fabricated data set.

## Execution and recorded boundary

```bash
JAX_PLATFORMS=cpu uv run --frozen --extra dev python -m benchmarks.numerical.scientific_acceptance --dtype float32 --weight-evaluation stable --check --output-dir path/to/new-evidence
UV_PROJECT_ENVIRONMENT="$HOME/.local/share/hpfracc/venv-cuda13" JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --frozen --extra dev --extra cuda13 python -m benchmarks.numerical.scientific_acceptance --dtype float64 --weight-evaluation direct --check --output-dir path/to/new-evidence
```

The immutable
[pre-repair bundle](audits/scientific-acceptance-2026-11-10/README.md) preserves
eight reports and 48 measured configurations, including the two failing direct
float32 weak-decay residuals. Those nonzero exits are historical scientific
rejections, not executions relabelled as successful after the repair.

The [NUM-004 repair bundle](audits/num004-direct-precision-2026-11-10/README.md)
records eight new reports and 48 configurations. All eight reports pass the
same frozen policy: 32 required-success trajectories are accepted; all 16
negative controls have `accepted=False`, `policy_passed=True`. Float32 direct
weak-decay independent scaled residuals are `0.016331607987060983` on CPU and
`0.01743271498279139` on CUDA, below the unchanged limit of one. Explicit stable
evaluation retains its weak-decay values `0.052358017489466745` /
`0.01643224778294578`. These are selected-case results, not a general float32
nonlinear/stiff accuracy guarantee.

Omission of `--weight-evaluation` still means direct, not stable. NUM-004 changes
the working-precision evaluation of the same direct Diethelm weights;
**legacy direct floating-point roundoff is not preserved**. The stable
interval-moment route, full-history method, normalization, Newton updates and
tolerances are unchanged. No budget or manifest criterion above is relaxed.
The coefficient-corruption regression is a separate negative control, not a
new manifest case or a claim that current direct weights must fail.
Local Python 3.13.13 full CPU/CUDA suites each pass 850 tests, one performance
case deselected, with confirmed zero process exits. Independent review and
remote multi-version CI are separate; CI configuration alone does not establish
execution. See NUM-004 in the
[tracker](../developer/bug-resolution-tracker.md) and
[uniform-weight evaluation](uniform-weights.md).

No solver defaults, runtime dependencies or public numerical APIs change for
this policy. General adaptive calibration, additional stiff/nonlinear orders,
long horizons and Phase A sign-off remain separate gates.
