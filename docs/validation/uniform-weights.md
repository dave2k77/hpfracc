# Stable Uniform Solver Weight Evaluation

Status: Unreleased, opt-in numerical hardening. Solver APIs remain provisional;
`"stable"` describes roundoff evaluation, not an API tier or a larger integration
stability region.

## Select explicitly

```python
import hpfracc as hp

explicit_solver = hp.solvers.PredictorCorrector(
    dt=1/256, order=0.5, weight_evaluation="stable",
)
implicit_solver = hp.solvers.ImplicitPredictorCorrector(
    dt=1/256, order=0.5, newton_iterations=8, weight_evaluation="stable",
)
```

Both uniform deterministic configurations accept `"direct"` (the default) and
`"stable"`. The new setting is recorded in `SolverInfo.diagnostics`. Unknown
names raise `ValueError`. Existing positional configuration arguments retain
their meaning; the option is appended to the dataclass fields.

The default still calls the original direct power-difference formulas. Stable
evaluation uses the same full-history PECE discretization and separate endpoint
coefficient, with the same timestep/gamma normalization, time-grid requirements,
Newton update count and residual criterion. It does not truncate history or
enable float64 globally. No runtime speedup is claimed: stable interval evaluation
does more weight arithmetic, and total history work remains quadratic.

This option is for deterministic uniform explicit/implicit solvers. Nonuniform
and adaptive solvers already use stable interval moments and have no new selector.
`hp.prob.simulate_stochastic` rejects a configuration requesting `"stable"`: its
separate drift/noise kernels need their own validation. Vector solver orders
remain unsupported.

## Same weights through interval moments

At step `n`, interval `j` has unit-width lag bounds `[n-j-1,n-j]`. Let `I_j`
integrate `u^(alpha-1)` on that interval, and let `L_j,R_j` integrate it against
the left/right linear nodal bases in forward time. The unscaled weights are:

```text
predictor[j] = alpha * I_j
corrector[0] = alpha*(alpha+1)*L_0
corrector[j] = alpha*(alpha+1)*(L_j + R_(j-1)), 1 <= j < n.
```

The current endpoint coefficient is one and remains handled separately by the
existing solver. Future/history-empty entries are zero. On a uniform grid these
are algebraically the original Diethelm weights. The implementation reuses the
[stable interval-moment evaluation](../theory/solvers.md#graded-mesh-precision),
including its bounded series for narrow lag intervals and safe endpoint branches.
Combining positive moments avoids cancellation of large, nearly equal powers.

## Coefficient evidence

The [protocol](audits/uniform-weights-2026-09-15/protocol.json) fixes orders
`0.01,0.1,0.3,0.5,0.8,0.9,0.99`, steps `1,3,16,256,4096`, both evaluations and
both precisions. The 140 rows compare predictor/corrector arrays to independent
host panel quadrature. Unit tests additionally use 70-digit interval integrals
at boundary, interior and masked positions.

All stable rows passed the predeclared relative tolerances: `5e-6` for float32
and `5e-12` for float64. The largest measured stable corrector relative errors
were approximately `6.4e-7` and `2.0e-15`, respectively. These concern weights
before timestep/gamma normalization, not global trajectory errors.

Selected float32 maximum relative corrector errors:

| Order | Elapsed intervals | Direct | Stable |
| --- | --- | --- | --- |
| 0.1 | 256 | 0.1401 | 2.82e-7 |
| 0.5 | 256 | 0.01153 | 3.25e-7 |
| 0.5 | 4,096 | 3.647 | 3.25e-7 |
| 0.99 | 4,096 | 1.206 | 1.27e-7 |

Some direct long-history float32 weights become zero or negative. These are
retained as measurements, not counted as evidence of accurate direct evaluation.
The stable path has positive historical weights in the tested matrix without
clipping negative coefficients after calculation.

## Paired solver evidence

The solver matrix repeats the [implicit audit's](implicit-nonlinear.md) half-order
decay and scalar/coupled/mixed-scale manufactured problems, at both strengths
and all original grid/order settings, holding Newton iterations at eight. Both
evaluations are run, producing **240 configurations** across the two precisions.
Every trajectory in this selected matrix was finite and passed its working-
precision Newton residual criterion.

Diagnostic tolerances remain `rtol=1e-5, atol=1e-7` for float32 and
`rtol=1e-10, atol=1e-12` for float64. Comparisons use the same independent
float64 residual reconstruction and continuous references as the preceding audit.

For half-order decay at strength 30 and 257 nodes:

| Precision / evaluation | Independent max scaled residual | Disagreeing step flags | Endpoint absolute error |
| --- | --- | --- | --- |
| Float32 direct | 673.78 | 235 | 6.469e-5 |
| Float32 stable | 0.966 | 0 | 1.626e-5 |
| Float64 direct | 0.180 | 0 | 1.618e-5 |
| Float64 stable | 2.54e-4 | 0 | 1.618e-5 |

This addresses the dominant weight-roundoff discrepancy in that case. It does
not guarantee that every independent residual or continuous error improves.
For the mixed-scale coupled case at strength 30, order 0.5 and 257 nodes in
float32, the independent scaled residual rises from `0.496` to `1.112` while the
maximum dimensionless trajectory error falls from `5.12e-6` to `2.37e-6`.
Working-precision RHS evaluation, state scaling, normalization and accumulation
still matter. All such outcomes are retained.

The coarse-grid positivity counterexample also remains. At strength 30,
order 0.5 and 17 nodes, stable evaluation still produces the first state near
`-0.27416`, despite a positive continuous solution and a converged corrector.
The integration method's startup/discretization error is separate from weight
roundoff; changing evaluation alone cannot fix it.

## Tests and compatibility boundary

The new tests cover high-precision coefficient references, positive moments,
constant/affine identities, order gradients, small-grid agreement with the direct
formulas, and agreement with the nonuniform solver on an equispaced float64 grid.
They also check nonlinear explicit/implicit state/parameter/order finite
differences, JIT/eager consistency, trailing shapes, singleton behavior, bounded
graphs, configuration guards, metadata, and the stochastic rejection boundary.

Default versus explicitly requested direct trajectories are array-equal in
selected tests; the existing unrolled-reference and solver regressions remain
in place. Stable/direct agreement is a floating-point tolerance claim, not a
cross-backend bitwise claim. On the selected explicit polynomial problem, grid
refinement retains the method's approximately `1+alpha` behavior; roundoff
hardening is not a higher-order discretization.

## Reproduce and interpret

Use a new output directory for another experiment:

```powershell
$env:JAX_ENABLE_X64 = '0'
uv run --frozen python -m benchmarks.numerical.uniform_weight_validation --check --output-dir docs/validation/audits/uniform-new
uv run --frozen python -m benchmarks.numerical.uniform_weight_validation --mode solvers --check --output-dir docs/validation/audits/uniform-new
$env:JAX_ENABLE_X64 = '1'
uv run --frozen python -m benchmarks.numerical.uniform_weight_validation --dtype float64 --check --output-dir docs/validation/audits/uniform-new
uv run --frozen python -m benchmarks.numerical.uniform_weight_validation --mode solvers --dtype float64 --check --output-dir docs/validation/audits/uniform-new
```

All four recorded checked commands exited 0. Coefficient checks require complete
coverage, valid masks, and stable-reference accuracy/positivity at the declared
tolerances. Solver checks require complete, coherent measurements, not global
error, positivity, or independent residual at most one. The solver matrix focuses
on implicit behavior; explicit evidence is in the unit tests.

- [Float32 coefficients CSV](audits/uniform-weights-2026-09-15/uniform_coefficients_float32.csv)
  and [JSON/provenance](audits/uniform-weights-2026-09-15/uniform_coefficients_float32.json).
- [Float64 coefficients CSV](audits/uniform-weights-2026-09-15/uniform_coefficients_float64.csv)
  and [JSON/provenance](audits/uniform-weights-2026-09-15/uniform_coefficients_float64.json).
- [Float32 solver CSV](audits/uniform-weights-2026-09-15/uniform_solvers_float32.csv)
  and [JSON/provenance](audits/uniform-weights-2026-09-15/uniform_solvers_float32.json).
- [Float64 solver CSV](audits/uniform-weights-2026-09-15/uniform_solvers_float64.csv)
  and [JSON/provenance](audits/uniform-weights-2026-09-15/uniform_solvers_float64.json).

The records identify local Windows CPU execution, Python 3.13.13, JAX/jaxlib
0.10.1, configurations and source/CSV digests. Precision sweeps ran concurrently;
no timing or accelerator claim is made. Earlier audit bundles remain historical.
This is dirty-tree development evidence, not release sign-off or Phase A closure.

## Final development checks

- `uv run python -m pytest`: **653 passed**, 1 performance test deselected,
  in 996.88 s; confirmed process exit 0.
- Dedicated uniform-weight implementation/report tests: **62 passed**.
- The four checked report commands exited 0, covering **140 coefficient rows**
  and **240 paired solver configurations**.
- `uv run python -m benchmarks.numerical.validation_summary --check`: all
  **20 baseline/hardening rows passed**. The new matrices are separate reports.
- Strict docs, Ruff 0.16.7 on changed Python files, and whitespace checks passed.

The targeted coefficient-roundoff issue now has an opt-in implementation and
independent evidence. Broader mixed-scale arithmetic, error control, performance
and release-readiness gates remain open. This work remains local and uncommitted
on top of `37aaac6` and the preceding development changes.
