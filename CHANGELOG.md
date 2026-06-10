# Changelog

## 0.1.0a0

Status: pre-alpha candidate, released 2026-06-10.

- Initial project skeleton.
- Added full-history fractional operators, operator validation, and scaling
  smoke benchmarks.
- Added fixed-step Caputo predictor-corrector solver, solver validation report,
  and minimal fractional ODE example.
- Added aggregate numerical validation summary, finite-difference gradient
  checks, stability checks, and CPU-oriented baseline benchmarks.
- Added experimental `hp.nn.NeuralFODE`, trajectory MSE helpers, pytree SGD, and
  a synthetic fractional ODE training example.
- Added experimental stochastic simulation, scalar-grid Gaussian calibration,
  posterior predictive summaries, and a probabilistic calibration example.
- Added public API stability-tier documentation for the pre-alpha API surface.
- Added runtime provenance capture with `hp.config.current_provenance()`.
- Added Phase 6 alpha release-readiness documentation, validation status
  summary, release checklist, and smoke-tested example documentation.

### Numerical correctness fixes (2026-06-10 core review)

- Added an observed-convergence-order harness
  (`benchmarks.numerical.convergence`, `tests/unit/test_convergence_order.py`)
  verifying the Caputo L1 operator attains max-norm order `2 - alpha`, the
  Grunwald-Letnikov / Riemann-Liouville operators attain first order `O(h)`, and
  the predictor-corrector solver attains endpoint order `1 + alpha`.
- Replaced the tautological Riemann-Liouville-equals-Grunwald-Letnikov check
  with genuine analytic validation: added `hp.ops.riemann_liouville_power_law`
  and the decisive constant case (`t**(-alpha) / Gamma(1 - alpha)`,
  distinguishing RL from Caputo).
- Fixed `hp.prob.posterior_predictive` credible intervals to respect posterior
  weights via a new `hp.prob.weighted_quantile`; previously the bands ignored the
  posterior entirely.
- Re-expressed the Caputo predictor-corrector solver with `jax.lax.scan` over a
  preallocated history buffer for a bounded autodiff graph and tractable
  reverse-mode differentiation; also widened the time-grid uniformity tolerance
  to a ULP-scaled atol so large uniform float32 grids are accepted.
- Derived and validated the additive-noise FSDE noise scaling as a fractional
  Euler-Maruyama scheme with exact per-interval kernel-variance weights
  (validated against the analytic variance by Monte-Carlo); restricted to
  `alpha > 1/2`.
