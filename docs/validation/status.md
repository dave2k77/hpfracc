# Validation Status

Status: Current development validation boundary.

Active phase: Phase A

Latest tagged release: `0.1.0a1` (2026-06-26).

HPFRACC validation is scoped to research software claims for fractional
operators, deterministic fractional differential equation solvers, differentiable
workflows, and small experimental probabilistic utilities. This page separates
validated v0.1 behavior from provisional or out-of-scope capabilities.

HPFRACC is not clinical, diagnostic, or subject-specific decision software.
Validation here does not establish biological realism or fitness for medical
use.

## Released Features and Current Evidence

The features in this table shipped through `0.1.0a1`; the evidence references
describe the current development tests, not a newly archived release dossier.
The additional hardening described below is **unreleased** and must not be
attributed to the a1 tag. The [roadmap](../developer/roadmap.md#current-development-state)
separates implemented capabilities from the remaining Phase A completion gates.
Selected numerical validation does not promote provisional public APIs to stable.

| Area | Current status | Evidence |
| --- | --- | --- |
| Caputo derivative of constants | Validated for the v0.1 full-history L1 implementation. | Unit tests and `uv run python -m benchmarks.numerical.operator_validation.report`. |
| Caputo power-law references | Validated against analytic `t**power` references with refinement checks. | Unit tests and operator validation report. |
| Observed convergence order | Validated that the Caputo L1 operator attains max-norm order `2 - alpha`, the Grunwald-Letnikov / Riemann-Liouville operator attains first order `O(h)`, and the predictor-corrector solver attains endpoint order `1 + alpha`, by float64 grid-refinement slope estimates. | Unit tests in `tests/unit/test_convergence_order.py` and `uv run python -m benchmarks.numerical.convergence`. |
| FFT-accelerated history convolution | Validated as mathematically equivalent to the full-history reference on uniform grids for Caputo and Grunwald-Letnikov operators, including JIT and autodiff paths. | Unit tests in `tests/unit/test_fft_history.py` and `uv run python -m benchmarks.numerical.history_comparison`. |
| Short-memory history truncation | Validated to stay within a bounded relative error on smooth power-law signals for selected alpha values and window sizes. | Unit tests in `tests/unit/test_history_compression.py`. |
| Sum-of-exponentials kernel compression | Validated to stay within a bounded relative error on smooth power-law signals for selected alpha values and pole counts. | Unit tests in `tests/unit/test_history_compression.py`. |
| Riemann-Liouville correctness | Validated against the analytic RL power-law reference and the decisive constant case (RL of a constant is `t**(-alpha) / Gamma(1 - alpha)`, nonzero, distinguishing it from Caputo). Replaces the earlier tautological RL-equals-GL check. | Unit tests in `tests/unit/test_fractional_operators.py` and `uv run python -m benchmarks.numerical.operator_validation.report`. |
| JAX JIT compatibility | Smoke-tested for representative operators and solver calls. | Unit tests for JIT-compatible operator and solver paths. |
| Input/parameter differentiability | Validated with finite-difference checks for representative operator and solver quantities. | `uv run python -m benchmarks.numerical.gradient_checks` and unit tests. |
| Fractional-order (`alpha`) differentiability | Validated with finite-difference checks across a range of `alpha` for the Caputo, Grunwald-Letnikov, and Riemann-Liouville operators and the predictor-corrector solver endpoint, including a finiteness/JIT smoke test. The operators use a NaN-safe power so the gradient stays finite where a weight raises a zero base. | Unit tests in `tests/unit/test_alpha_gradients.py` and `uv run python -m benchmarks.numerical.gradient_checks`. |
| Vector / per-state fractional orders | Validated for the Caputo, Grunwald-Letnikov, and Riemann-Liouville operators: a per-state order array decouples component-by-component (each component matches the scalar operator at its own order), a uniform vector reproduces the scalar result exactly, the `full` / `fft` / `short_memory` history methods agree, and gradients with respect to the order vector match finite differences (`soe` rejects vector orders; the fixed-step solver remains scalar). | Unit tests in `tests/unit/test_vector_orders.py`. |
| Non-uniform / graded Caputo grids | Validated for the Caputo operator with a `t=` node array: the non-uniform L1 weights reduce to the uniform weights within tested floating-point tolerances on an equispaced grid; analytic power-law errors decrease under refinement; a graded mesh improves accuracy for selected weakly singular inputs; and order-gradients match finite differences. Full-history and scalar-order only; non-`full` history, vector orders, and GL/RL with `t=` raise. `PredictorCorrector` remains uniform-grid; nonuniform solvers are separate configurations. | Unit tests in `tests/unit/test_nonuniform_grids.py`. |
| Scalar linear Caputo FDE refinement | Validated against a truncated Mittag-Leffler reference for selected grids. | `uv run python -m benchmarks.numerical.solver_validation.report`. |
| Non-uniform / graded-mesh Caputo solver (`NonUniformPredictorCorrector`) | Validated agreement with uniform `PredictorCorrector` within tested tolerances on an equispaced grid, Mittag-Leffler refinement on graded meshes, and improved max-norm convergence for selected origin-singular problems; finite initial-state/parameter gradients, JIT consistency, and an `n`-independent graph are tested. Explicit and full-history. Adaptive stepping exists as `AdaptivePredictorCorrector`; SOE-accelerated solver history remains future work. | Unit tests in `tests/unit/test_nonuniform_solver.py`. |
| Implicit Caputo solver (`ImplicitPredictorCorrector`) | Validated Mittag-Leffler refinement at mild rates and bounded decay on a selected stiff linear problem where explicit PECE diverges; a moderately stiff comparison uses a fine-grid self-refinement reference. Linear-dynamics iteration independence, finite initial-state/parameter gradients, and an `n`-independent graph are tested. Full-history and uniform-grid; an adaptive/implicit combination remains future work. These cases do not establish general nonlinear stiffness robustness. | Unit tests in `tests/unit/test_implicit_solver.py`. |
| Adaptive-step Caputo solver (`AdaptivePredictorCorrector`) | Validated that the a-posteriori adaptive solver, which chooses its mesh from a step-doubling (Richardson) local-error estimate, tracks tolerance (the error against a Mittag-Leffler reference decreases as `rtol` is tightened) and earns its keep on the `t^alpha` origin singularity (it clusters nodes near `t=0` and reaches an accuracy a uniform fixed-step grid of several times as many nodes cannot, because the uniform global error stalls). Because it advances with the single full step (no local extrapolation), the realized trajectory is identical to `NonUniformPredictorCorrector` on the realized mesh, so the frozen-mesh gradient equals that solver's gradient there and is finite-difference checked (the step-size controller and accept/reject decisions are not differentiated); `jit` is consistent and the masked-scan graph is bounded by `max_steps` independently of the problem. Correct full-history `O(N^2)` reference; SOE-accelerated history (cheap rejections), dense output, and an adaptive+implicit combination are future. | Unit tests in `tests/unit/test_adaptive_solver.py`. |
| Basic numerical stability checks | Smoke-tested for constant zero response and non-amplifying scalar decay. | `uv run python -m benchmarks.numerical.stability`. |
| Baseline benchmark context | Local CPU-oriented timing rows include backend/platform context. | `uv run python -m benchmarks.numerical.baseline`. |
| Stochastic reproducibility mechanics | Reproducibility with fixed PRNG keys and variation across keys are tested for the additive-noise helper. | Unit tests in `tests/unit/test_probabilistic_phase5.py`. |
| Additive-noise FSDE variance | For the linear additive FSDE (`alpha > 1/2`), the simulated variance matches the analytic variance `sigma^2 * integral [tau^(alpha-1) E_{alpha,alpha}(lambda tau^alpha)]^2 dtau` (Monte-Carlo check) and the sample mean reproduces the deterministic Mittag-Leffler solution. The solver rejects `alpha <= 1/2`. | Unit tests in `tests/unit/test_probabilistic_phase5.py`. |
| Scalar-grid probabilistic calibration behavior | Tested for Gaussian likelihood preference, nearby scalar parameter recovery, normalized posterior weights, posterior-predictive summary shape, and posterior-weighted credible intervals (the band tracks the posterior and stays within trajectory support). | Unit tests in `tests/unit/test_probabilistic_phase5.py` and the calibration example. |
| Runtime provenance capture | Tested for JSON-compatible runtime context and graceful operation without git. | Unit tests in `tests/unit/test_provenance.py`. |

## Unreleased Hardening Evidence

### Input and Grid Validation

`tests/unit/test_solver_input_validation.py` covers finite, strictly increasing
solver grids, uniform spacing at tiny scales, invalid timesteps, and adaptive
dtype normalization. Numeric validation requiring concrete values is eager-only;
compiled callers must supply validated inputs. Whole-result dataclasses are not
registered JAX pytrees: compiled examples return array fields and numeric
diagnostics rather than complete result objects.

### Graded-Mesh Roundoff Checks

`tests/unit/test_graded_mesh_precision.py` compares nonuniform solver weights
against 70-digit mpmath interval-integral references evaluated on the actual
float32/float64 input nodes. It tests positivity, constant/affine moment
identities, series-switch and near-endpoint cases, the Caputo operator's earliest
history contribution, and JIT/order gradients. The shared stable evaluation
retains the original full-history product-integration scheme; it does not imply
general float32 accuracy for arbitrarily graded or long-horizon problems.

### Implicit Residual Diagnostics

`tests/unit/test_implicit_diagnostics.py` checks under-iterated nonlinear
correctors against an independently reconstructed first-step residual, successful
nonlinear convergence with additional iterations, componentwise thresholds,
linear solves, and non-finite/singular solves. Tests cover eager strict-JSON
diagnostics and JIT-accessible status arrays, and verify that changing diagnostic
tolerances leaves the trajectory and its gradients unchanged. This evidence
concerns algebraic residual reporting, not general nonlinear solver robustness
or global error control.

### Adaptive Completion Diagnostics

`tests/unit/test_adaptive_diagnostics.py` checks JIT-accessible completion flags
and output masks, completion on the final allowed attempt, rejected-step budget
accounting, retained prefixes after non-finite trials, initial-data failures,
unrepresentable midpoints, minimum-step tolerance failures, tiny time horizons,
lossy integer-bound conversion (including integer maxima), mixed-precision
initial-state storage, and zero-attempt singleton completion. Existing adaptive
tests retain the
accuracy/refinement and frozen-mesh gradient checks. Operational success is not
a guarantee of global error within `rtol`/`atol`.

`tests/unit/test_integer_time_grids.py` checks wide signed-integer spans against
constant-forcing solver and linear-signal Caputo references, in eager and JIT
execution. Monotonicity checks compare adjacent nodes without subtraction;
positive integer lag/interval differences avoid signed overflow before conversion
to floating point.

### Adaptive Tolerance Calibration Audit

The [independent calibration audit](adaptive-calibration.md) measures scalar
constant/quadratic forcing, exact half-order decay, and manufactured feedback in
float32 and float64. It records global errors at accepted nodes separately from
completion and the trial indicator. Nontrivial-case absolute errors decrease
under the tested tolerance refinements, but reference-scaled errors exceed one.

An exact first-panel quadratic-forcing derivation and executable acceptance
counterexample show that the current indicator is not calibrated to the retained
coarse-step error. The audit preserves numerical solver behavior; controller
correction remains an open Phase A gate. Neither a green audit command nor
operational success certifies the requested global accuracy.

The [startup-scaling follow-up](adaptive-startup.md) validates selected weak-forcing
and autonomous leading coefficients against independent references. It also
records a smooth cubic forcing with zero coarse/fine difference but nonzero
error. Neither an alpha-only multiplier nor replacing the controller exponent
alone resolves that sampling blind spot. The prototype below uses additional
information; its broader validation remains open.

### Experimental Subdivision Prototype

`AdaptivePredictorCorrector(error_estimator="subdivision")` is an opt-in
prototype using additional quarter-step probes, componentwise max normalization,
and bounded step factors. The default remains unchanged. Tests cover rejection
of the cubic blind-spot trial, analytic accuracy, public padding/last-slot bounds,
quarter-only nonfinite and resolution failures, dtypes, and successful JIT/gradient
behavior. The [measured comparison](adaptive-subdivision.md) records 32 completed
scalar runs across two precisions and modes; some global errors still exceed
the reference tolerance scale. This is not promotion to a calibrated solver.

## Provisional Capabilities

The following capabilities exist, but should not be treated as validated stable
research infrastructure yet:

- Numerical behavior for long time horizons or very large state dimensions.
- Stochastic fractional differential equation accuracy beyond the linear
  additive-noise variance check: nonlinear drift, state-dependent
  (multiplicative) diffusion, and `alpha <= 1/2` noise are not validated.
- Broad Bayesian inference workflows beyond scalar-grid Gaussian calibration.
- Performance claims outside the local benchmark context.
- Memory-efficiency claims for short-memory/SOE: the current implementations
  approximate weights but still apply them through dense full-history kernels.
  Selected signal-error checks do not establish bounded-memory execution or
  universal truncation/compression error bounds.
- Biological, neural, EEG, clinical, diagnostic, or subject-specific realism.
- Placeholder domain namespaces such as `hpfracc.brain`, `hpfracc.observe`,
  `hpfracc.train`, `hpfracc.data`, and `hpfracc.viz`.

## Aggregate Report Coverage

The aggregate summary now includes the fixed core-hardening report in addition
to the original baseline rows. Run either report as an executable check:

```bash
uv run python -m benchmarks.numerical.validation_summary --check
uv run python -m benchmarks.numerical.core_hardening --check
```

New rows cover graded history precision, nonuniform refinement, implicit
convergence/failure diagnostics, and adaptive completion/failure/JIT status.
Intentional solver failures count as passing validation only when the expected
failure is detected. Case configurations and tolerances are recorded per row;
the [methodology](methodology.md#aggregate-validation-harness) defines the scope.
Other features, including broader history acceleration and vector-order behavior,
still rely on their separate tests/reports rather than these new summary cases.

## Required Validation Commands

Run the aggregate validation summary:

```bash
uv run python -m benchmarks.numerical.validation_summary --check
```

Run detailed operator and solver reports:

```bash
uv run python -m benchmarks.numerical.operator_validation.report
uv run python -m benchmarks.numerical.solver_validation.report
```

Run component validation checks when detailed rows are needed:

```bash
uv run python -m benchmarks.numerical.gradient_checks
uv run python -m benchmarks.numerical.stability
```

Run the observed-convergence-order report (estimates empirical orders against
the expected `2 - alpha` operator and `1 + alpha` solver rates):

```bash
uv run python -m benchmarks.numerical.convergence
```

Run the baseline benchmark for local timing context:

```bash
uv run python -m benchmarks.numerical.baseline
```

See [Validation Methodology](methodology.md) for report formats, command
options, and provenance capture guidance. See the
[v0.1.0a0 Report Bundle](reports/v0.1.0a0/README.md) for historical artifacts from
the original alpha baseline. Those files are not evidence for the a1 extensions
or unreleased hardening and must not be relabeled as a new candidate bundle.

## Interpreting Validation Output

Validation reports write plain CSV to stdout. A passing validation row means the
specific documented case passed its configured tolerance. It does not imply a
general theorem about all fractional systems, all discretizations, all time
horizons, or all hardware backends.

When saving validation artifacts for release notes, save runtime provenance as a
separate JSON artifact next to the CSV output:

```bash
uv run python - <<'PY'
import json
import hpfracc as hp

print(json.dumps(hp.config.current_provenance().to_dict(), indent=2))
PY
```

## Release-Readiness Rule

For every candidate, a v0.1 alpha release should not make a claim unless it is
covered by one of:

1. a unit test;
2. a validation report command;
3. a documented benchmark with runtime context;
4. an explicit provisional/experimental label that limits the claim rather than
   substituting for missing validation.

Local development checks are not an executed release gate. Preserve the source
revision, dirty state, configuration, environment, and report outputs; use the
[candidate checklist](../developer/release-checklist.md) for final clean-source
verification and artifact capture. Phase A closure additionally requires its
remaining roadmap gates, not just passing the aggregate report.