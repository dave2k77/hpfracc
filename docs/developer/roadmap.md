# Phase Development Roadmap

HPFRACC development is milestone-gated. Each phase must produce validation
evidence before later scientific or domain-specific layers depend on it.

## Current Development State

Active phase: Phase A

Latest tagged release: `0.1.0a1` (2026-06-26).

Phases 0-6 delivered the original `0.1.0a0` baseline. The `0.1.0a1` release
added Phase A capabilities; it did **not** close Phase A. The hardening changes
listed below are **unreleased**, even while package metadata still reads
`0.1.0a1`. Use the changelog, commit and working-tree state to distinguish the
development code from that tag. Phases B-D remain planned.

This page is the current milestone map; the
[validation status](../validation/status.md) defines the evidence boundary and
the [release checklist](release-checklist.md) is an unexecuted candidate template,
not a record that the current development tree is ready to tag.

The post-alpha roadmap below (Phases A-D) supersedes the earlier
Phases 7-9 sketch. It maps the path from the released v0.1.0a0 "minimal viable
scientific kernel" to the full layered blueprint in
`hpfracc_design_specification.pdf`, sequenced Core-first per the v0.1 priority
order. Four standing decisions constrain it:

1. Core-first: harden the mathematical engine before the brain stack.
2. No NumPyro: the probabilistic layer stays JAX-native without an additional
   probabilistic backend dependency,
   evolving the hand-rolled approach toward SVI/MCMC while preserving the public
   `hp.prob` surface.
3. Additive growth: keep the flat `ops/`, `solvers/`, `prob/` layout and add new
   sibling modules; do not refactor into the blueprint's granular directory tree.
4. Preserve the released API: all additions are additive, with deprecation paths
   for any later incompatible change.

## Phase 0: Project Spine and Governance

Status: Complete.

Goal: make the repository coherent, installable, testable, and explicit about
research boundaries.

Deliverables:

- MIT-licensed package scaffold.
- JAX-only v0.1 numerical backend decision.
- Research usage policy and non-clinical disclaimer.
- Public namespace skeleton.
- CI for imports, unit tests, and docs.
- Roadmap and architectural decision records.

Exit criteria:

- `import hpfracc as hp` works.
- Unit tests pass.
- Docs build is configured.
- Research-only positioning is visible in docs and README.

## Phase 1: Fractional Operator Core

Status: Complete.

Goal: implement validated Riemann-Liouville, Caputo, and Grunwald-Letnikov
operators on uniform grids for scalar `0 < alpha < 1`.

Deliverables:

- Uniform-grid GL derivative.
- Uniform-grid RL derivative using the documented GL discretisation.
- Uniform-grid Caputo derivative using the L1 scheme.
- Analytic validation tests for constants and polynomials.
- JIT and differentiation smoke tests where JAX is available.
- Theory documentation for definitions, assumptions, and limitations.
- Operator metadata/result contracts.
- Validation report command.
- Operator scaling smoke benchmark.

Exit criteria:

- Operator tests pass on CPU.
- Operators support leading time axis and arbitrary trailing state dimensions.
- `alpha` gradients are documented as provisional.
- Complexity and full-history limitations are documented.

## Phase 2: Deterministic Caputo FDE Solver

Goal: implement the first credible deterministic fractional dynamical-system
solver.

Status: Complete.

Deliverables:

- Fixed-step Caputo predictor-corrector solver for scalar `0 < alpha < 1`.
- `simulate(...)` API following the engineering contract.
- `PredictorCorrector` solver configuration object.
- Structured `SimulationResult` and `SolverInfo` diagnostics.
- Uniform-grid validation and early failure for nonuniform grids.
- Validation against scalar reference FDEs.
- Gradients with respect to initial state and model parameters.
- Solver validation report command.
- Minimal fractional ODE example.

Exit criteria:

- Timestep refinement behavior is demonstrated.
- JIT/non-JIT consistency is tested.
- Solver limitations and failure modes are documented.
- `python -m pytest` passes.
- `mkdocs build --strict` passes.

Implementation checklist:

- Add `hp.solvers.PredictorCorrector`.
- Add `hp.solvers.simulate`.
- Accept callable dynamics `f(t, state, params, *, rng_key=None, inputs=None)`.
- Support states with arbitrary trailing dimensions.
- Return trajectories with leading time axis.
- Record fractional order, timestep, step count, method, and warnings in
  `SolverInfo`.
- Validate against `D_C^alpha y = lambda y`, `y(0)=1`, using a truncated
  Mittag-Leffler reference.
- Add gradient smoke tests with respect to `initial_state` and a scalar model
  parameter.

## Phase 3: Numerical Validation and Benchmark Harness

Status: Complete.

Goal: make correctness and performance claims reproducible.

Deliverables:

- Operator convergence reports.
- Solver refinement reports.
- Finite-difference gradient checks.
- CPU baseline benchmarks.
- Optional single GPU/TPU benchmarks when hardware is available.

Exit criteria:

- Validation commands produce repeatable summaries.
- Performance claims include benchmark context.
- Validation status is visible in docs.

## Phase 4: Differentiable Scientific Model Layer

Status: Complete.

Goal: support trainable fractional dynamical models after solver validation.

Deliverables:

- Experimental `hp.nn` model conventions.
- First neural FODE wrapper.
- Minimal training example on synthetic data.
- Equinox integration decision after core APIs stabilize.

Exit criteria:

- Gradients flow through model parameters and initial conditions.
- A small synthetic recovery example is tested.

## Phase 5: Stochastic and Probabilistic Layer

Status: Complete.

Goal: add uncertainty-aware workflows after deterministic differentiability is
solid.

Deliverables:

- Experimental additive-noise FSDE support.
- Scalar-grid Gaussian calibration workflow.
- Posterior predictive utilities.
- Small reproducible probabilistic calibration example.
- Hand-rolled scalar-grid inference; the earlier optional NumPyro proposal is
  superseded by the NumPyro-free post-alpha decision above.

Exit criteria:

- One reproducible probabilistic calibration example.
- Posterior diagnostics and assumptions are documented.

## Phase 6: Research-Ready v0.1 Alpha

Status: Complete.

Goal: make the completed numerical, differentiable-model, and experimental
probabilistic foundations coherent enough for a credible v0.1 alpha release
before adding domain-specific brain-model layers.

Deliverables:

- Documentation consistency pass across README, theory pages, API contracts,
  validation methodology, and developer notes.
- Runnable examples for operators, deterministic solvers, differentiable
  training, and probabilistic calibration.
- Explicit API stability tier labels for public namespaces and objects.
- Provenance metadata review for result objects, validation reports, and
  examples.
- Release checklist covering tests, docs, validation reports, benchmarks,
  citation metadata, and research-use disclaimers.

Exit criteria:

- `python -m pytest` passes.
- `mkdocs build --strict` passes.
- Validation and benchmark commands are documented and runnable.
- Public API stability expectations are visible in docs.
- Release-readiness gaps are tracked before the v0.1 alpha tag.

Task backlog: see [Phase 6 Alpha Backlog](phase-6-research-ready-alpha.md).
Release gate: see the [v0.1 Alpha Release Checklist](release-checklist.md).

## Phase A: Harden the Numerical Core

Status: Active; not closed.

Goal: make the mathematical engine broad, fast, and trustworthy enough to carry
the SciML and brain layers before any domain code depends on it.

### Released capabilities through 0.1.0a1

| Area | Implemented boundary |
| --- | --- |
| History methods | Uniform-grid operator `full`, `fft`, `short_memory`, and `soe` paths are explicit opt-ins except the full-history default. FFT evaluates the same discrete convolution up to floating-point error. Short-memory and SOE approximate weights but still use dense full-history application; they are not yet bounded-memory implementations. |
| Orders and grids | Uniform operators accept scalar or per-state orders with `full`, `fft`, and `short_memory`; SOE is scalar-only. Nonuniform Caputo L1 is scalar-order, full-history only. GL/RL remain uniform-grid operators. |
| Deterministic solvers | `PredictorCorrector` and `ImplicitPredictorCorrector` use uniform grids; `NonUniformPredictorCorrector` uses supplied nodes; `AdaptivePredictorCorrector` chooses a mesh from the supplied bounds. All remain scalar-order and full-history. Implicit stepping is uniform only, not an adaptive/implicit combination. |
| Differentiation | Selected state, parameter, and order gradients have finite-difference/JIT evidence. Adaptive derivatives are frozen-mesh sensitivities, not derivatives of the controller or accept/reject decisions. Validated cases do not promote the public APIs out of their provisional stability tier. |
| Test tiers | Unit, property, golden-regression, and separately selected performance tests exist. Their presence is not proof that every Phase A combination or performance claim is covered. |

<a id="phase-a-unreleased"></a>

### Unreleased hardening

- Shared solver validation for finite, strictly increasing grids, tiny uniform
  spacings, and safe integer differences; adaptive time/state dtype handling.
- Stable graded-mesh interval moments shared by nonuniform Caputo and
  nonuniform/adaptive solvers, with independent high-precision references.
- Implicit algebraic residual diagnostics and adaptive operational
  completion/failure diagnostics, including JIT-readable arrays and accepted
  output masks. Neither is a global integration-error guarantee.
- Fixed core-hardening report cases added to the existing aggregate CSV schema;
  `--check` makes failed or empty reports return a nonzero exit status. Corruption
  tests check report sensitivity, including independent implicit residuals.
- Coupled adaptive accuracy/order/scale evidence and frozen-mesh gradients extend
  the scalar stress and runtime audits; subdivision remains experimental.
- Short-memory operators now build retained weights only and use direct windowed
  convolution. Independent truncation/gradient tests and local float32 forward
  timing/XLA buffer measurements are recorded in
  [windowed history](../validation/windowed-history.md).
- A pure-JAX reproducer isolates the Windows float64 native-convolution failure.
  Windowed application now uses an equivalent lag scan, with complete benchmark
  sweeps in both precisions; see [the follow-up](../validation/native-convolution.md).
- SOE operator application is now recursive, with a named `scaled_tail` fit using
  streamed sufficient statistics and a positive GL-prefix/increment identity.
  Independent analytic/gradient checks and full accuracy/scaling matrices are
  recorded in [SOE history](../validation/soe-history.md). Legacy fitting remains
  selectable and default, with its poor long-sequence regimes explicitly recorded.
- A [nonlinear/stiff implicit audit](../validation/implicit-nonlinear.md) adds
  independent continuous and discrete-residual references, 360 measured cases,
  and selected coupled state/parameter/order gradients. It documents coarse-grid
  positivity loss and float32 uniform-weight roundoff separately from Newton
  convergence; solver updates and defaults are unchanged.
- Uniform explicit/implicit solvers now expose `weight_evaluation="stable"` as
  an opt-in evaluation of the same coefficients, with direct evaluation still
  default. The [paired evidence](../validation/uniform-weights.md) addresses the
  float32 weight-roundoff issue with independent coefficient, solver and gradient
  checks. Model/accumulation precision and method-level positivity limits remain.
- Consolidation adds locked Ruff/uv CI checks, lazy runtime provenance imports,
  a [capability matrix](../validation/capabilities.md), and seven fixed SOE/stable-
  weight checks in the aggregate report. The numerical checkpoint is `e19f0c1`;
  consolidation remains unreleased and is not candidate sign-off.
- Experimental `BoundedForcing` / `AdaptiveForcingQuadrature` adds a restricted
  full-history nodal interpolation bound for prescribed C2 forcing, conditional
  on caller-supplied global curvature. The [60-case audit](../validation/bounded-forcing.md)
  and independent gradient/failure tests support this boundary; roundoff and
  state-feedback error propagation remain outside it. An eighth fixed extension
  gate brings the current aggregate to 28 rows.

The implementation and regression evidence live in the source tree and are
described in [validation status](../validation/status.md) and the changelog's
Unreleased section. Local development checks are not a clean-candidate release
sign-off, accelerator validation, or a complete Phase A evidence dossier.

<a id="phase-a-gates"></a>

### Remaining Phase A gates

1. Audit each new scheme/history path against the analytic, convergence,
   gradient, and JIT requirements below. Record supported combinations and
   case-specific tolerances; selected smoke tests or report rows are not blanket
   coverage. In particular, extend independent adaptive tolerance-calibration
   and nonlinear/stiff implicit evidence beyond the controlled manufactured and
   half-order cases. Stable uniform weight evaluation now addresses the identified
   coefficient-roundoff target through an explicit opt-in. Extend precision
   evidence beyond these cases, including mixed-scale model/accumulation effects;
   reported Newton convergence alone does not certify positivity or global error.
   The [initial adaptive audit](../validation/adaptive-calibration.md) now
   demonstrates first-panel indicator underestimation and separates global
   errors from operational completion; a justified calibration correction
   remains open, rather than being inferred from decreasing refinement errors.
   The [startup follow-up](../validation/adaptive-startup.md) supplies the
   weak-forcing/autonomous scaling and a zero-indicator cubic counterexample.
   An [experimental subdivision prototype](../validation/adaptive-subdivision.md)
   now uses additional quarter-step information behind an explicit opt-in.
    Broader calibration, nonlinear/stiff evidence, and matched-work evaluation
    remain gates before promotion; the default and the phase status are unchanged.
    A [conditional prescribed-forcing bound](../validation/bounded-forcing.md)
    now provides a justified restricted target with independent evidence, using
    supplied global curvature rather than inferred pair discrepancies. Extending
    calibrated control to weak forcing and nonlinear feedback, and bounding
    working-precision error, remain open.
2. Complete the memory-efficiency evidence: windowed short-memory and recursive
   SOE application are implemented with selected CPU evidence. Extend
   approximation calibration, tolerance-driven fitting/selection, and
   reverse-mode/accelerator and peak-memory measurements. SOE remains scalar-order;
   do not infer per-state support from other operator histories. The Windows
   native crash is avoided by the lag scan; its pure-JAX reproducer remains
   available for upstream investigation. Selected signal bounds
   are not universal error guarantees. Solver histories remain uncompressed.
3. Reconcile any remaining grid/order and differentiation scope with the actual
   supported combinations. Per-state solver orders are not implemented; do not
   infer them from per-state operator support. Record explicit deferrals rather
   than silently treating partial coverage as completion.
4. Capture a reviewed candidate's validation/benchmark outputs with provenance.
   The committed `v0.1.0a0` bundle is historical, not evidence for `0.1.0a1` or
   these unreleased changes. CI and lint now use the locked uv environment;
   verify the actual remote checks for the reviewed candidate and preserve its
   outputs. A configured workflow and local checks are not an executed release
   dossier or accelerator evidence.

Full-result dataclass JIT return support and runtime numeric validation of traced
inputs remain limitations. Current compiled callers should return array fields
and numeric diagnostics and supply prevalidated inputs. These limitations must
stay visible unless deliberately implemented and validated.

Original Phase A deliverables (retained as the completion target):

- Memory-efficient kernels (opt-in, never silently substituted for full-history):
  FFT-accelerated history convolution, short-memory truncation, sum-of-exponentials
  compression.
- Solver breadth: adaptive-step control and an implicit scheme for stiff regimes,
  alongside the existing predictor-corrector, reusing the `lax.scan` history-buffer
  pattern and returning the same `SimulationResult`.
- Grid and order generality: nonuniform time grids and vector / per-state
  fractional orders extending the original scalar `0 < alpha < 1` baseline.
- Validated gradients with respect to fractional order `alpha` (promoted from
  provisional).
- Test-tier build-out: `tests/property/` (hypothesis identities),
  `tests/regression/` (golden artifacts), `tests/performance/`, plus convergence
  checks for every new scheme.

Exit criteria:

- Every new operator/solver has analytic, convergence-order, gradient, and JIT
  validation.
- FFT paths agree with full-history within documented floating-point tolerances;
  approximate paths have documented, validated error bounds versus full-history
  for their claimed scope, with measured memory/scaling evidence.
- `python -m pytest` and `mkdocs build --strict` pass; the validation summary
  covers the new areas; `docs/validation/status.md` is updated.

## Phase B: SciML Depth

Goal: turn the single `NeuralFODE` into a real differentiable-modelling layer and
grow inference natively (no NumPyro).

Deliverables:

- `hp.nn`: `NeuralFSDE` on the experimental additive-noise solver after the
  required stochastic validation, PINN-style residual
  losses, and graph-coupled dynamics on `(n_nodes, n_nodes)` connectivity.
- `hp.train`: `Trainer`, losses, loops, callbacks, checkpoints, with Optax as an
  optional optimizer dependency and a pure-pytree fallback.
- `hp.prob`: extend the hand-rolled JAX-native layer toward variational inference
  and a simple MCMC sampler, preserving the existing
  `hp.prob.grid_calibrate_scalar` / `hp.prob.posterior_predictive` APIs.
  A generic `fit` entry point would be new, not an existing public contract.
- Probabilistic validation: parameter recovery, posterior predictive checks, and
  simulation-based calibration on controlled examples.

Exit criteria:

- Gradients flow through all new model types.
- One synthetic recovery example per model class.
- SVI/MCMC validated on a closed-form posterior; `tests/integration/` covers a
  fit-to-predict round trip.

## Phase C: Brain MVP - Observation, Metrics, Phantom Brain

Goal: deliver the blueprint headline, the end-to-end phantom-brain to EEG to
criticality demonstration.

Deliverables:

- `hp.brain`: `connectomes.random_modular`, `node_models.FractionalWilsonCowan`,
  and `phantom.PhantomBrain` composing connectome, node model, and observation
  model, runnable through `hp.solvers.simulate`.
- `hp.observe`: `eeg.LinearLeadField` and `observe.run` for the latent-to-sensor
  forward map, in deterministic and probabilistic observation modes.
- `hp.metrics`: trajectory, spectral, and uncertainty summaries, and the headline
  `criticality.report(eeg, metrics=["psd_slope", "dfa", "avalanche",
  "branching_ratio"])` returning structured output that separates point from
  interval estimates.
- `hp.data`: synthetic EEG-like dataset generation from tunable phantom brains.
- Domain validation: regime control (subcritical, near-critical, supercritical),
  spectral-slope control, and avalanche/branching where model-appropriate, with a
  `benchmarks/criticality/` suite.

Guardrails: every brain and metric output carries the non-clinical, model-relative
disclaimer; "ground truth" stays mechanism-relative; criticality signatures are
documented as model- and measurement-dependent.

Exit criteria:

- The phantom-brain example runs end to end.
- Metric assumptions and failure modes are documented; regime-control tests pass;
  all brain APIs are labeled experimental.

## Phase D: Reproducibility Infrastructure, Scaling, and Research Release

Goal: make the expanded platform reproducible, benchmarked, and publishable.

Deliverables:

- Experiment infrastructure: `experiments/{configs,scripts,reports,manifests}`,
  typed `ExperimentConfig` and manifest objects, and `hp.cli`.
- Benchmark suites: `benchmarks/{inference,scaling,criticality}` beyond today's
  `numerical/`, with compile time, runtime, memory, node-count and
  sequence-length scaling, and batching efficiency.
- `hp.viz`: trajectory, spectra, network, posterior, and benchmark plotting
  helpers.
- A multi-device design ADR (`pmap`/`pjit`/sharding), forward-compatible without
  required implementation.
- Research cookbooks and tutorials, plus a full API reference.
- A versioned research release graduating from the pre-alpha line, with complete validation
  and performance reports and provenance artifacts.

Exit criteria:

- Validation and benchmark reports are complete across all blueprint layers.
- Experiment manifests reproduce a published-style result.
- Docs and citation metadata are ready; the release checklist is green.
