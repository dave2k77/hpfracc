# Phase Development Roadmap

HPFRACC development is milestone-gated. Each phase must produce validation
evidence before later scientific or domain-specific layers depend on it.

Current status:

- Phase 0: Complete in commit `3f46a84`.
- Phase 1: Complete in commit `3f46a84`.
- Phase 2: Complete in the current working tree.
- Phase 3: Complete in the current working tree.
- Phase 4: Complete in the current working tree.
- Phase 5: Complete in the current working tree.
- Active phase: Phase 6, research-ready v0.1 alpha.
- Next domain phase: Phase 7, neural mass and neural field foundations.

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
- NumPyro integration deferred until small probabilistic contracts stabilize.

Exit criteria:

- One reproducible probabilistic calibration example.
- Posterior diagnostics and assumptions are documented.

## Phase 6: Research-Ready v0.1 Alpha

Status: Active.

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

## Phase 7: Neural Mass and Neural Field Foundations

Status: Planned.

Goal: introduce domain-relevant dynamical systems downstream of the numerical
core.

Deliverables:

- Experimental fractional Wilson-Cowan model.
- Neural field design note or prototype.
- Regime and sensitivity tests.

Exit criteria:

- Model-relative behavior is documented.
- No biological or clinical realism claims are made.

## Phase 8: Observation, Metrics, and Phantom-Brain MVP

Goal: build the first end-to-end research demonstration.

Deliverables:

- Linear observation model.
- PSD slope, DFA, and minimal avalanche/branching metrics.
- Small phantom-brain network of validated neural mass nodes.

Exit criteria:

- Synthetic EEG-like demonstration runs end to end.
- Metric assumptions and warnings are explicit.

## Phase 9: Scaling, Release, and Publication Readiness

Goal: prepare a credible open-source research release.

Deliverables:

- Optional compressed-memory strategies.
- Multi-device design ADR.
- Benchmark and validation reports.
- API stability review.
- Versioned `0.1.0` research release.

Exit criteria:

- Validation report is complete.
- Benchmark report is complete.
- Docs and citation metadata are ready.
