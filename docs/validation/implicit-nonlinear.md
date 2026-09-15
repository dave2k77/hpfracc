# Nonlinear and Stiff Implicit-Solver Audit

Status: Unreleased development evidence, recorded 2026-09-15. This work extends
validation of the existing uniform fixed-iteration implicit method. It does not
change Newton updates, solver defaults, or the full-history discretization.

The subsequent [stable uniform-weight implementation](uniform-weights.md)
addresses the coefficient-roundoff issue through an explicit opt-in. The
measurements below preserve the preceding direct-evaluation audit.

## Questions and independent references

The [pre-execution protocol](audits/implicit-2026-09-15/protocol.json) fixes the
matrix after exploratory 17-node float64 probes. The audit separates:

1. agreement with a continuous solution;
2. convergence of the working-precision algebraic corrector;
3. its residual when reconstructed using independent float64 quadrature.

The four problems on `[0,1]` are:

| Case | Dynamics and exact solution |
| --- | --- |
| Half-order decay | `D^0.5 y = -k*y`, `y(0)=1`; `y(t)=erfcx(k*sqrt(t))`. This remains reliable at large negative Mittag-Leffler arguments. |
| Scalar cubic feedback | `D^alpha y = q(t)-k*(y^3-u(t)^3)`, with `u=1+t^2` and `q=2*t^(2-alpha)/Gamma(3-alpha)`. |
| Coupled cubic feedback | `D^alpha y = S*(q-R*((S^-1*y)^3-u^3))`, where `u=(1+t^2,0.5+t^3)` and `R=[[k+1,-k],[-k,k+1]]`. `q` contains the analytic Caputo derivatives of the two polynomial components. Here `S=I`. |
| Mixed-scale coupled feedback | The same coupled problem with `S=diag(1e-3,1e3)`. The continuous solution is `S*u`; dimensionless errors divide by these scales as represented in the input dtype. |

The coupled feedback is genuinely nonlinear and has nonzero off-diagonal state
derivatives. At the exact trajectory it vanishes. Continuous Caputo quadrature
tests independently validate the manufactured forcing and half-order decay
identity, including a wrong-RHS negative control.

For the discrete residual, a separate host routine integrates the piecewise-linear
RHS on each panel using 32-point Gauss-Legendre quadrature. The newest panel's
singular endpoint uses exact beta moments. Constant and affine moment tests
validate these weights. It imports no solver weight helpers and does not use a
fine-grid run of the same solver as its reference.

## Fixed sweep and check semantics

- Orders `0.3,0.5,0.8` for manufactured cases; `0.5` for half-order decay.
- Strengths `k=1,30`, node counts `17,65,257`, and Newton counts `1,3,8`.
- Float32 and float64, 180 configurations each. Dyadic grid spacings are exactly
  representable in both precisions.
- Diagnostic `newton_rtol=1e-5` in float32 and `1e-10` in float64;
  `newton_atol=0.01*newton_rtol`. These are audit settings, not new defaults.
- Compiled solves take dynamic initial state, strength, and order arguments.

Both checked commands exited 0: **360 complete, contract-consistent measurements**.
This does **not** mean every solve converged. `--check` requires the exact unique
matrix and coherent output shapes, dtypes, reported masks, residuals and counts.
It does not require all trajectories to be finite, all Newton solves to converge,
or continuous errors to satisfy a threshold. Finite unconverged trajectories retain
their labeled errors; nonfinite trajectories have empty accuracy/residual fields.

Independent residual disagreements are measured, rather than used as a blanket
cross-precision pass/fail gate. They use more accurate quadrature/RHS arithmetic
than the float32 solver's assembled equation. Scientific assertions for selected
cases live in the tests, not merely in this report's measurement-integrity gate.

## Results

| Precision | Newton iterations | Configurations | Finite trajectories | Reported fully converged trajectories |
| --- | --- | --- | --- | --- |
| Float32 | 1 | 60 | 51 | 14 |
| Float32 | 3 | 60 | 60 | 60 |
| Float32 | 8 | 60 | 60 | 60 |
| Float64 | 1 | 60 | 51 | 6 |
| Float64 | 3 | 60 | 60 | 50 |
| Float64 | 8 | 60 | 60 | 60 |

**The diagnostic tolerances differ by precision**, so the convergence counts are
not a ranking of float32 versus float64 accuracy. All 18 nonfinite runs used one
Newton iteration: strength 30 with order 0.3 at 17/65 nodes, or order 0.5 at 17
nodes, in each of the three nonlinear cases and both precisions.

One Newton step can also return finite but grossly inaccurate nonlinear results.
For scalar cubic feedback at order 0.8, strength 30 and 17 nodes, the float64 max
dimensionless error is about `2.51e5`. Eight iterations reduce it to `4.69e-5`.
The report preserves both results. Three iterations can already give small
trajectory errors while failing the deliberately strict float64 residual check;
algebraic and integration tolerances control different quantities.

### Converged correctors do not guarantee positivity or global accuracy

For scalar decay, define `mu=k*h^alpha/Gamma(alpha+2)`. The exact first discrete
implicit step is:

```text
y_1 = (1-alpha*mu)/(1+mu).
```

It becomes negative when `alpha*mu>1`, even though the continuous solution is
positive. At order 0.5, strength 30 and 17 nodes, the first state is approximately
`-0.27416087` and the maximum trajectory error is `0.34873456`. One and eight
Newton iterations produce the same discrete trajectory within roundoff and
both satisfy the residual criterion.

Refining to 257 nodes makes the first state positive and reduces the float64
max error to `0.14714573`, but the endpoint error is only `1.62e-5`. The startup
error and endpoint error therefore tell very different stories. More Newton
iterations cannot repair this coarse-grid discretization error. This evidence
does not justify clipping states or promoting the method as positivity-preserving.

### Float32 weight/history precision is a separate limit

The independent residual uses nominal uniform nodes, host float64 RHS values and
panel-integral weights. The solver reports the residual of its own working-
precision assembled equation. Differences include coefficient evaluation,
normalization, dynamics and accumulation roundoff; they are not automatically
an error in the diagnostic implementation.

For half-order decay at strength 30, 257 nodes and eight iterations:

| Precision | Reported max scaled residual | Independently reconstructed max | Disagreeing step flags |
| --- | --- | --- | --- |
| Float32 | 0.00636 | 673.78 | 235 |
| Float64 | 1.17e-6 | 0.17983 | 0 |

A separate eager coefficient probe at order 0.5 and 257 nodes compared the
current uniform weight formula with the independent final-row quadrature. It
found max absolute error `2.60e-5` and max relative error approximately **1.15%**
in float32. The uniform second-power differences remain cancellation-sensitive,
unlike the already-hardened nonuniform interval moments. More Newton iterations
leave these weights unchanged.

Improved uniform weight evaluation is a concrete follow-up. The reconstruction
also shows why a very small working-precision residual alone cannot certify the
accuracy of the underlying product integration.

## Validation added

`tests/unit/test_implicit_validation.py` adds 19 tests/parameterized cases:

- independent continuous references and positive constant/affine panel moments;
- the negative first-step counterexample and iteration independence for linear
  dynamics;
- a scalar first-step cubic root checked with independent bracketing;
- selected nonlinear refinement with observed near-second-order max errors on
  smooth manufactured half-order cases at strength 1 and 33/65/129 nodes;
- normalized trajectory agreement under component rescaling and active nonlinear
  coupling checks;
- JIT/eager agreement and initial-state, strength and order finite-difference
  gradients for cubic and mixed-scale coupled cases at orders 0.3, 0.5 and 0.8;
  these tests use strength 3 and 33 nodes, with manufactured forcing held fixed
  during the order perturbation;
- nonfinite-error suppression, corrupted diagnostic detection, sensitivity of
  the independent residual to corrupted states, and exact matrix coverage.

These selected cases expand the evidence boundary; they do not establish general
nonlinear stiffness robustness, convergence for arbitrary Newton budgets, or
controller/solver-history compression capabilities.

## Reproduce and artifacts

Use a new output directory for another experiment:

```powershell
$env:JAX_ENABLE_X64 = '0'
uv run --frozen python -m benchmarks.numerical.implicit_validation --check --output-dir docs/validation/audits/implicit-new
$env:JAX_ENABLE_X64 = '1'
uv run --frozen python -m benchmarks.numerical.implicit_validation --dtype float64 --check --output-dir docs/validation/audits/implicit-new
```

- [Float32 CSV](audits/implicit-2026-09-15/implicit_validation_float32.csv) and
  [JSON, counts and provenance](audits/implicit-2026-09-15/implicit_validation_float32.json).
- [Float64 CSV](audits/implicit-2026-09-15/implicit_validation_float64.csv) and
  [JSON, counts and provenance](audits/implicit-2026-09-15/implicit_validation_float64.json).

The run used Windows CPU, Python 3.13.13 and JAX/jaxlib 0.10.1. Both accuracy
sweeps ran concurrently; no timing/performance conclusions are drawn. Source
digests, CSV digests, configuration and dirty-tree provenance identify this
development snapshot on top of `37aaac6`. The solver source is unchanged by the
audit. This is not a clean-candidate release dossier or Phase A closure.

## Final development checks

- `uv run python -m pytest`: **591 passed**, 1 performance test deselected,
  in 935.16 s; confirmed exit code 0.
- Dedicated implicit audit tests: **19 passed**.
- Both implicit audit `--check` commands exited 0 with all 360 measurements,
  including the explicitly labeled unconverged/nonfinite results.
- `uv run python -m benchmarks.numerical.validation_summary --check`: all
  **20 baseline/hardening rows passed**. The larger implicit matrix is a
  separate report, not extra coverage silently attributed to the aggregate.
- Strict docs, Ruff 0.16.7 on the new Python files, and whitespace checks passed.

The next core-hardening target is stable evaluation of uniform-grid history
weights, using the independent quadrature and float32 discrepancy above as
validation anchors. Coarse-grid positivity loss remains a distinct method-level
limitation. All new work remains local and uncommitted.
