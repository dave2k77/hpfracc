# Changelog

## Unreleased

Post-alpha Phase A (harden the numerical core).

- Added an a-posteriori adaptive-step Caputo solver
  `hp.solvers.AdaptivePredictorCorrector`. It chooses its own mesh from a
  step-doubling (Richardson) local-error estimate -- one step of `h` against two of
  `h/2` -- and accepts / shrinks / grows the step to track `rtol`/`atol`, clustering
  nodes where the solution is hard to resolve. (The predictor/corrector difference,
  the usual integer-order estimate, is *not* usable here: because every step
  re-sums the whole history it behaves like the global error and would force the
  step to zero.) `simulate`'s `ts` then supplies only the integration bounds
  `(ts[0], ts[-1])`. It advances with the single full step (no local
  extrapolation), so the realized trajectory is identical to
  `NonUniformPredictorCorrector` run on the realized mesh and the gradient is the
  exact frozen-mesh sensitivity (finite-difference checked); the step-size
  controller and accept/reject decisions are not differentiated (the diffrax
  discretize-then-optimize pattern). The integration runs as a fixed-length masked
  `lax.scan` of `max_steps` iterations, so it stays `jit`-traceable with a bounded
  autodiff graph. On a problem with a weak `t**order` origin singularity it reaches
  an accuracy the uniform fixed-step solver cannot reach even with several times as
  many nodes (the uniform global error stalls). It remains a correct full-history
  `O(N**2)` reference -- rejected steps re-evaluate history; a sum-of-exponentials
  acceleration (cheap rejections, `O(M)` per step), dense output, and an
  adaptive+implicit combination remain future extensions. Added
  `tests/unit/test_adaptive_solver.py`.
- Added a non-uniform / graded-mesh Caputo solver
  `hp.solvers.NonUniformPredictorCorrector`. It uses the variable-step fractional
  Adams-Bashforth-Moulton weights built from the actual node spacings of `ts` (no
  scalar `dt`), so any strictly-increasing time grid is admissible; on an
  equispaced grid it reduces exactly to `PredictorCorrector`. An a-priori mesh
  graded toward `t=0` (`t_j = T (j/N)**r`) restores the convergence the uniform
  grid loses to the solution's weak `t**order` origin singularity. `simulate`
  dispatches on the solver type; the solver stays explicit one-pass PECE,
  full-history, `jit`-traceable with a bounded autodiff graph, and differentiable.
  Added `tests/unit/test_nonuniform_solver.py`. A-posteriori adaptive step-size
  control and an SOE-accelerated non-uniform history remain future extensions, as
  does the implicit solver on non-uniform grids.
- Added an implicit Caputo solver `hp.solvers.ImplicitPredictorCorrector`. It
  solves the fractional Adams-Moulton corrector equation
  `y_n = C + corrector_scale * f(t_n, y_n)` at each step with a fixed-iteration
  Newton solve (seeded by the explicit predictor) instead of the explicit
  one-pass PECE correction, giving a much larger stability region: on a stiff
  linear problem where the explicit `PredictorCorrector` diverges, the implicit
  method stays bounded and tracks the decaying solution. It shares the explicit
  solver's predictor/corrector history weights, stays full-history, fixed-step,
  uniform-grid, `jit`-traceable with a bounded autodiff graph, and differentiable
  in the initial state and parameters; `simulate` dispatches on the solver type.
  Added `tests/unit/test_implicit_solver.py`. Adaptive step-size control remains a
  future extension.
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
