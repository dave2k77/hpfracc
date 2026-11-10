# Uniform Solver Weight Evaluation

Status: Unreleased numerical hardening: opt-in interval moments and the NUM-004
direct-arithmetic repair. Selected-gate and local full CPU/CUDA verification is
recorded in the repair bundle; review and release gates are separate. Solver
APIs remain provisional; `"stable"`
describes roundoff evaluation, not an API tier or a larger integration stability
region.

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

The default remains `"direct"`, but NUM-004 replaces its cancellation-prone
power differences with cancellation-safe evaluation of the same Diethelm
coefficients. **Legacy direct floating-point roundoff is not preserved.**
The explicit `"stable"` interval-moment route is unchanged. Both settings retain
the full-history PECE discretization and separate endpoint coefficient, with
the same timestep/Gamma normalization, time-grid requirements, Newton updates
and residual criteria. Neither truncates history nor enables float64 globally.
No runtime speedup is claimed; total history work remains quadratic.

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

## Historical coefficient evidence — original stable opt-in

The historical [protocol](audits/uniform-weights-2026-09-15/protocol.json) fixes orders
`0.01,0.1,0.3,0.5,0.8,0.9,0.99`, steps `1,3,16,256,4096`, both evaluations and
both precisions. The 140 rows compare predictor/corrector arrays to independent
host panel quadrature. Unit tests additionally use 70-digit interval integrals
at boundary, interior and masked positions.

All stable rows passed the predeclared relative tolerances: `5e-6` for float32
and `5e-12` for float64. The largest measured stable corrector relative errors
were approximately `6.4e-7` and `2.0e-15`, respectively. These concern weights
before timestep/gamma normalization, not global trajectory errors.

These measurements predate the direct-arithmetic repair. Here and in the paired
solver table below, direct means the **legacy power-difference implementation**,
not the current repaired default. Selected float32 maximum relative corrector errors:

| Order | Elapsed intervals | Legacy direct | Stable |
| --- | --- | --- | --- |
| 0.1 | 256 | 0.1401 | 2.82e-7 |
| 0.5 | 256 | 0.01153 | 3.25e-7 |
| 0.5 | 4,096 | 3.647 | 3.25e-7 |
| 0.99 | 4,096 | 1.206 | 1.27e-7 |

Some legacy direct long-history float32 weights became zero or negative. These
are retained as historical measurements, not current direct-evaluation results.
The stable path has positive historical weights in the tested matrix without
clipping negative coefficients after calculation.

## Historical paired solver evidence

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
| Float32 legacy direct | 673.78 | 235 | 6.469e-5 |
| Float32 stable | 0.966 | 0 | 1.626e-5 |
| Float64 legacy direct | 0.180 | 0 | 1.618e-5 |
| Float64 stable | 2.54e-4 | 0 | 1.618e-5 |

The historical stable opt-in addressed the dominant weight-roundoff discrepancy
in that recorded case. It does
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

## Working-precision normalization follow-up

Uniform deterministic explicit and implicit solvers also share an improved
evaluation of `dt**alpha / Gamma(1+alpha)` and its recurrence-derived corrector
scale, `predictor_scale / (1+alpha)`. Both direct and stable weight settings use
it. This changes floating-point normalization, not the quadrature scheme, Newton
iteration count, diagnostic thresholds, or the default history/weight selector.
Earlier audit matrices below and above predate this follow-up.

The CUDA float32 cubic first-step check exposed the distinction: Newton solved
its working-precision equation with a reported scaled residual near `0.033`,
while the host-float64 equation gave `1.795` against the unchanged limit of one.
The corrector scale's relative error was approximately `1.27e-6`, traced to
general-purpose float32 Gamma evaluation rather than an unconverged Newton solve.

The private normalization uses [DLMF 5.7.3](https://dlmf.nist.gov/5.7.E3) for
`log Gamma(1+alpha)`, evaluated in Horner form. On `0 <= alpha <= 1`, the
alternating-series remainder through degree 24 (float32) or 56 (float64) is
bounded by the first omitted term, below `1.2e-9` or `1.3e-19` respectively.
These are exact-arithmetic truncation bounds, not total floating-point or solver
error guarantees. No runtime mpmath/SciPy dependency, host conversion of traced
orders, or hidden enabling of x64 is introduced.

`tests/unit/test_solver_normalization.py` checks the constants against 80-digit
references, values and order derivatives against 70-digit references across the
supported interval, eager/JIT coefficients, dtype preservation, and an
independently reconstructed float32 cubic residual. The singular-Newton fixture
uses the actual normalization to keep its Jacobian exactly singular at working
precision rather than depending on the previous special function's roundoff.

The [local CUDA follow-up bundle](audits/gpu-normalization-2026-11-10/README.md)
records source/runtime provenance, paired old/new coefficients and checked CPU/
GPU aggregates. The selected independent cubic residual falls from `1.795` to
`0.0783` without relaxing its threshold. This is not a global accuracy guarantee.

## NUM-004 — cancellation-safe default direct arithmetic

The deterministic uniform `_pece_weights` predictor now evaluates the power
difference with `expm1`/`log1p`. Its boundary corrector uses a positive binomial
series through degree `K=28` in float32 or `K=56` in float64. Interior correctors
retain `M=14` or `M=28` even terms, respectively. Computation uses the actual
working dtype, without global x64 changes or a fallback to the stable selector.
The separate stochastic drift/noise weight helper is untouched.

For `0 < alpha < 1`, normalize each series by its positive leading term. For
boundary steps or interior lags at least two, let `r` be the reciprocal step or
lag, so `r <= 1/2`. Boundary successive terms have ratio at most `r`; interior
even terms have ratio at most `r**2`. Since the normalized sum is at least one,
geometric tails give conservative relative truncation bounds:

```text
interior: r**(2*M) / (1-r**2)
boundary: r**(K-1) / (1-r)
```

At `r=1/2` these bounds are approximately `4.9671e-9` / `1.4901e-8` for
float32 and `1.8504e-17` / `5.5511e-17` for float64 (interior / boundary), below
their respective machine epsilons. These bound **exact-arithmetic series
truncation only**, not floating-point accumulation, order derivatives or solver
error. First-step/lag-one cases have separate formulas. Inactive branches use
finite operands before `where`, with safe step/lag values at least two, so masked
future entries and singleton cases do not evaluate singular inactive powers or
logarithms.

The [new repair bundle](audits/num004-direct-precision-2026-11-10/README.md)
records eight new executions under the unchanged `implicit-scientific-v1`
manifest and budgets. All 32 required-success cases are accepted and all 16
negative controls remain rejected with passing policy flags. For float32
weak decay, direct independent maximum scaled residuals are
`0.016331607987060983` on CPU and `0.01743271498279139` on CUDA, below the
unchanged threshold of one. The stable path reproduces its prior residuals
`0.052358017489466745` / `0.01643224778294578`. The
[pre-repair scientific bundle](audits/scientific-acceptance-2026-11-10/README.md)
and all older tables retain the original failing arithmetic as historical
evidence; they are not overwritten or relabelled as repaired results.

`tests/unit/test_direct_weight_precision.py` targets independent coefficients
and default-path residuals. The stable-improvement comparison in
`test_uniform_weights.py` explicitly uses frozen legacy arithmetic, not the
repaired direct path. Default versus explicitly selected direct remains the
same configuration; neither is promised to reproduce old direct trajectories
bitwise. Local Python 3.13.13 full suites pass 850 tests on each CPU/CUDA backend
with one performance case deselected and confirmed zero process exits.
Independent review, remote multi-version CI and release gates are separate.
This repair does not establish arbitrary nonlinear/stiff float32
accuracy or remove normalization, model-evaluation, accumulation or coarse-grid
positivity limitations. Phase A remains open.

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

## Historical development checks — original stable opt-in

The following counts and source checkpoint belong to the original opt-in
implementation, not the NUM-004 repair or its pending candidate verification.

- `uv run python -m pytest`: **653 passed**, 1 performance test deselected,
  in 996.88 s; confirmed process exit 0.
- Dedicated uniform-weight implementation/report tests: **62 passed**.
- The four checked report commands exited 0, covering **140 coefficient rows**
  and **240 paired solver configurations**.
- `uv run python -m benchmarks.numerical.validation_summary --check`: all
  **20 baseline/hardening rows passed**. The new matrices are separate reports.
- Strict docs, Ruff 0.16.7 on changed Python files, and whitespace checks passed.

The targeted coefficient-roundoff issue received an opt-in implementation and
independent evidence. Broader mixed-scale arithmetic, error control, performance
and release-readiness gates remain open. That historical work was local and
uncommitted on top of `37aaac6` and the preceding development changes.
