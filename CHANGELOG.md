# Changelog

## Unreleased

- Replaced dense short-memory operator application with direct windowed
  convolution and retained-window-only weights, preserving the explicit
  truncation rule and scalar/per-state order support. Added independent
  truncated-sum/tail-bound, analytic cutoff, complex-signal, JIT and gradient
  checks, plus local float32 forward timing/XLA buffer evidence. SOE remains
  dense; repeated float64 benchmarking hit a native Windows JAX access violation
  recorded separately from passing numerical tests.
- Added a coupled nonlinear adaptive audit across component scales, fractional
  orders, tolerances, estimator modes, and both precisions (48 completed runs).
  Added frozen-mesh state/parameter/order gradient checks and a compiled
  quarter-only failure-prefix regression. Subdivision remains experimental;
  some global reference-scaled errors still exceed one.

- Added experimental opt-in `error_estimator="subdivision"` to the adaptive
  solver: quarter-step probes, maximum componentwise pair discrepancies, and
  bounded-factor step proposals. It detects the tested cubic cancellation while
  retaining coarse PECE states, frozen-mesh gradients, and existing public
  padding/status contracts. The default step-doubling behavior is unchanged.
  Added precision/failure/JIT/gradient regressions and a paired comparison;
  the new indicator remains heuristic, not a calibrated error bound.

- Derived and checked retained-coarse startup scaling for weak power forcing
  and smooth autonomous dynamics, with independent local-series references.
  Added an executable smooth cubic counterexample whose coarse/fine indicator
  vanishes despite nonzero error, plus raw measurements and provenance.
  This follow-up does not change controller behavior; a safeguarded correction
  requires more than an alpha-only multiplier or exponent change.

- Added an independent adaptive tolerance-calibration audit with float32/float64
  exact-solution sweeps and a first-panel quadratic-forcing counterexample.
  Recorded source/environment provenance and separated global error from
  operational completion. Clarified that the step-doubling indicator is not a
  calibrated retained-coarse error bound; numerical controller updates are
  unchanged pending a justified correction.

- Reconciled the roadmap, engineering contract, validation boundary, and release
  checklist: distinguish tagged a1 capabilities from unreleased hardening,
  retain Phase A's open gates, and label the original Phase 2/6 plans as
  historical. Added documentation-status and simulation-signature guards;
  the candidate checklist now requires checked reports and separate evidence.

- Extended the aggregate numerical report with fixed core-hardening cases for
  graded-mesh precision, nonuniform refinement, implicit residual diagnostics,
  and adaptive completion/failure/JIT status. Added the focused
  `benchmarks.numerical.core_hardening` command and opt-in `--check` exit status
  for both reports, preserving existing CSV columns and baseline case names.

- Hardened shared solver time-grid validation: reject non-finite nodes (including
  single-node grids), repeated/decreasing nodes, and non-finite uniform timesteps.
  Uniform-grid rounding tolerances now scale with the actual grid magnitude,
  rejecting spacing mismatches on tiny grids without rejecting valid float32
  grids. The shared checks cover explicit, implicit, and stochastic uniform paths.
- Fixed adaptive integration with integer time bounds and mixed time/state
  precision by keeping mesh buffers and step-size control in a common floating
  dtype. Nonuniform/adaptive weight normalization now respects the history dtype
  when x64 is enabled. Added eager, JIT, and gradient regression coverage and
  documented the boundary between eager validation and traced execution.

- Stabilized nonuniform/adaptive solver and Caputo L1 interval weights on graded
  meshes. Shared power-kernel moments use a bounded binomial series for narrow
  intervals and stable closed forms elsewhere, retaining the original
  full-history schemes without clipping negative weights or forcing float64.
  Added high-precision reference, positivity, moment, endpoint, and autodiff
  regressions, including zero-width trial intervals in completed adaptive scans.

- Added implicit Newton residual diagnostics with configurable `newton_rtol`
  and `newton_atol`, per-step convergence flags/scaled residuals, failure counts,
  and structured warnings. Status arrays remain available under JIT; eager
  diagnostics export non-finite residuals as JSON null with explicit failure
  flags. The fixed iteration count and numerical updates are unchanged.

- Added adaptive completion/failure flags, attempt counts, and a fixed-shape
  output mask, all accessible under JIT. Non-finite trials and step-size failures
  now stop without appending invalid states; exhausted budgets produce structured
  warnings and retain the accepted prefix. Removed premature completion on tiny
  time spans and forced acceptance of over-tolerance minimum steps. The step-size
  failure threshold scales with the integration span; numerical controller
  configuration values must be finite. Lossy integer time-bound conversions
  report precision failure rather than false completion, and history buffers
  preserve initial-state precision when dynamics return a narrower dtype.
- Fixed signed-integer overflow in grid monotonicity/spacing validation and
  nonuniform operator/solver lag calculations, with analytic eager/JIT checks
  on valid integer spans wider than the signed dtype range.

## 0.1.0a1

Status: pre-alpha, released 2026-06-26.

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
