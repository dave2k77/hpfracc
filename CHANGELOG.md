# Changelog

## Unreleased

Post-alpha Phase A (harden the numerical core).

- Added non-uniform / graded time grids to the `caputo` operator. `caputo` now
  accepts a `t=` array of (strictly increasing) time nodes as an alternative to a
  uniform `dt`; exactly one of the two must be given. The non-uniform path uses
  the L1 product-integration weights derived from the actual node spacings, which
  reduce exactly to the uniform `b_k` weights on an equispaced grid and let a mesh
  graded toward `t=0` recover accuracy for weakly-singular inputs. It is
  full-history only and scalar-order only; `history` other than `full`, a vector
  order, and `grunwald_letnikov` / `riemann_liouville` with `t=` all raise
  `NotImplementedError` (GL is a uniform-shift operator with no non-uniform
  analog). Gradients with respect to the order remain validated on the
  non-uniform path. Added `tests/unit/test_nonuniform_grids.py`. The fixed-step
  solver remains uniform.
- Added vector / per-state fractional orders to the `caputo`,
  `grunwald_letnikov`, and `riemann_liouville` operators. The `order` argument now
  accepts a per-state array broadcastable to the trailing state shape in addition
  to a scalar; each state component is differentiated independently (validated by
  a per-component decoupling property), the uniform-vector case reproduces the
  scalar result exactly, and gradients with respect to the order *vector* are
  validated against finite differences. The `full`, `fft`, and `short_memory`
  history methods support per-state orders; `soe` raises `NotImplementedError`.
  Added `tests/unit/test_vector_orders.py`. The fixed-step solver remains scalar.
- Validated gradients with respect to the fractional order `alpha`, promoting
  them from provisional. The operators now build their L1 normalisation and
  weights with `jax.scipy.special.gamma` and a NaN-safe power (instead of
  `math.gamma`), and order validation no longer coerces `alpha` to a Python
  `float` on traced paths (`hpfracc.ops.orders.as_order`), so `jax.grad` flows
  through `alpha` for the Caputo, Grunwald-Letnikov, and Riemann-Liouville
  operators and the predictor-corrector solver endpoint. Added
  `tests/unit/test_alpha_gradients.py`, alpha-gradient rows in
  `benchmarks.numerical.gradient_checks`, and an updated validation status.

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
