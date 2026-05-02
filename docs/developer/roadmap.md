# Phase Development Roadmap

HPFRACC development is milestone-gated. Each phase must produce validation
evidence before later scientific or domain-specific layers depend on it.

## Phase 0: Project Spine and Governance

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

Goal: implement validated Riemann-Liouville, Caputo, and Grunwald-Letnikov
operators on uniform grids for scalar `0 < alpha < 1`.

Deliverables:

- Uniform-grid GL derivative.
- Uniform-grid RL derivative using the documented GL discretisation.
- Uniform-grid Caputo derivative using the L1 scheme.
- Analytic validation tests for constants and polynomials.
- JIT and differentiation smoke tests where JAX is available.
- Theory documentation for definitions, assumptions, and limitations.

Exit criteria:

- Operator tests pass on CPU.
- Operators support leading time axis and arbitrary trailing state dimensions.
- `alpha` gradients are documented as provisional.
- Complexity and full-history limitations are documented.

## Phase 2: Deterministic Caputo FDE Solver

Goal: implement the first credible deterministic fractional dynamical-system
solver.

Deliverables:

- Fixed-step Caputo predictor-corrector solver.
- `simulate(...)` API following the engineering contract.
- Structured `SimulationResult` and solver diagnostics.
- Validation against scalar reference FDEs.
- Gradients with respect to initial state and model parameters.

Exit criteria:

- Timestep refinement behavior is demonstrated.
- JIT/non-JIT consistency is tested.
- Solver limitations and failure modes are documented.

## Phase 3: Numerical Validation and Benchmark Harness

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

Goal: add uncertainty-aware workflows after deterministic differentiability is
solid.

Deliverables:

- Experimental FSDE support.
- NumPyro-backed calibration workflow.
- Posterior predictive utilities.
- Small simulation-based calibration examples where feasible.

Exit criteria:

- One reproducible probabilistic calibration example.
- Posterior diagnostics and assumptions are documented.

## Phase 6: Neural Mass and Neural Field Foundations

Goal: introduce domain-relevant dynamical systems downstream of the numerical
core.

Deliverables:

- Experimental fractional Wilson-Cowan model.
- Neural field design note or prototype.
- Regime and sensitivity tests.

Exit criteria:

- Model-relative behavior is documented.
- No biological or clinical realism claims are made.

## Phase 7: Observation, Metrics, and Phantom-Brain MVP

Goal: build the first end-to-end research demonstration.

Deliverables:

- Linear observation model.
- PSD slope, DFA, and minimal avalanche/branching metrics.
- Small phantom-brain network of validated neural mass nodes.

Exit criteria:

- Synthetic EEG-like demonstration runs end to end.
- Metric assumptions and warnings are explicit.

## Phase 8: Scaling, Release, and Publication Readiness

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

