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
- Phase 6: Complete in the current working tree (v0.1.0a0 released and tagged).
- Active phase: Phase A, hardening the numerical core.

The post-alpha roadmap below (Phases A-D) supersedes the earlier
Phases 7-9 sketch. It maps the path from the released v0.1.0a0 "minimal viable
scientific kernel" to the full layered blueprint in
`hpfracc_design_specification.pdf`, sequenced Core-first per the v0.1 priority
order. Four standing decisions constrain it:

1. Core-first: harden the mathematical engine before the brain stack.
2. No NumPyro: the probabilistic layer stays dependency-free and JAX-native,
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
- NumPyro integration deferred until small probabilistic contracts stabilize.

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

Status: Active.

Goal: make the mathematical engine broad, fast, and trustworthy enough to carry
the SciML and brain layers before any domain code depends on it.

Deliverables:

- Memory-efficient kernels (opt-in, never silently substituted for full-history):
  FFT-accelerated history convolution, short-memory truncation, sum-of-exponentials
  compression.
- Solver breadth: adaptive-step control and an implicit scheme for stiff regimes,
  alongside the existing predictor-corrector, reusing the `lax.scan` history-buffer
  pattern and returning the same `SimulationResult`.
- Grid and order generality: nonuniform time grids and vector / per-state
  fractional orders extending today's scalar `0 < alpha < 1`.
- Validated gradients with respect to fractional order `alpha` (promoted from
  provisional).
- Test-tier build-out: `tests/property/` (hypothesis identities),
  `tests/regression/` (golden artifacts), `tests/performance/`, plus convergence
  checks for every new scheme.

Exit criteria:

- Every new operator/solver has analytic, convergence-order, gradient, and JIT
  validation.
- Compressed-memory paths are proven bit-equivalent or bounded versus full-history.
- `python -m pytest` and `mkdocs build --strict` pass; the validation summary
  covers the new areas; `docs/validation/status.md` is updated.

## Phase B: SciML Depth

Goal: turn the single `NeuralFODE` into a real differentiable-modelling layer and
grow inference natively (no NumPyro).

Deliverables:

- `hp.nn`: `NeuralFSDE` on the Phase A stochastic solver, PINN-style residual
  losses, and graph-coupled dynamics on `(n_nodes, n_nodes)` connectivity.
- `hp.train`: `Trainer`, losses, loops, callbacks, checkpoints, with Optax as an
  optional optimizer dependency and a pure-pytree fallback.
- `hp.prob`: extend the hand-rolled JAX-native layer toward variational inference
  and a simple MCMC sampler, preserving the public `hp.prob.fit` /
  `hp.prob.posterior_predictive` shape.
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
- A versioned research release graduating from `0.1.0a0`, with complete validation
  and performance reports and provenance artifacts.

Exit criteria:

- Validation and benchmark reports are complete across all blueprint layers.
- Experiment manifests reproduce a published-style result.
- Docs and citation metadata are ready; the release checklist is green.
