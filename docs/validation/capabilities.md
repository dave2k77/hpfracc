# Capability and Evidence Matrix

Status: Current development boundary after numerical checkpoint `e19f0c1` and
the validation/tooling consolidation and conditional prescribed-forcing extension.
Core operator/solver APIs remain provisional except explicitly experimental
extensions; model/probabilistic layers remain experimental.

**Implemented support and numerical evidence are different columns.** A passing
case does not validate every cross-product of grid, order, history, precision,
signal and transformation. The linked tests/audits define the actual cases.

## Operator combinations

All operators take arrays with leading time axis `(time, ...)`, and orders in
`0 < alpha < 1`. Per-state orders broadcast to the trailing state shape, not to
the time axis. RL uses the GL discretization with zero prehistory.

| Operator / grid | History | Order shape | Principal evidence and limits |
| --- | --- | --- | --- |
| Uniform Caputo L1 | `full` | Scalar or per-state | Constants, power laws, selected `2-alpha` convergence, input/order gradients and JIT. Dense reference application. |
| Uniform GL/RL | `full` | Scalar or per-state | Independent RL constants/power laws, first-order convergence, input/order gradients and JIT. Dense reference application. |
| Uniform Caputo/GL/RL | `fft` | Scalar or per-state | Forward equivalence to the reference, selected JIT/input gradients, and dedicated scalar/per-state order-gradient finite differences in explicitly checked float32/float64 execution on CPU/CUDA. Real-input FFT implementation; selected cases do not establish arbitrary-shape or long-horizon coverage. |
| Uniform Caputo/GL/RL | `short_memory` | Scalar or per-state | Independent truncated sums, discarded-tail bounds, real/complex signals, JIT/outer batching, and Caputo/GL input/per-state-order gradients. Windowed execution; physical history shrinks if `dt` is refined at fixed window. |
| Uniform Caputo/GL/RL | `soe`, `soe_fit="legacy"` | Scalar only | Recursive fitted-sum consistency and selected scalar input/order gradients. Original fitting policy remains default and has documented poor long-sequence/conditioning regimes. |
| Uniform Caputo/GL/RL | `soe`, `soe_fit="scaled_tail"` | Scalar only | Streamed fitting, independent fitted-sum and analytic checks, conditional absolute bounds, real/complex application and selected scalar input/order gradients. Exact first weight; GL/RL uses a positive-prefix/increment identity. Fixed poles/iterations do not imply an error tolerance. |
| Nonuniform/graded Caputo L1, `t=nodes` | `full` only | Scalar only | Uniform-grid consistency, power-law/graded refinement, high-precision interval moments, selected order gradients and JIT. |

Nonuniform GL/RL, nonuniform accelerated/truncated histories, vector SOE orders,
time-varying order fields and orders outside `(0,1)` are not supported by these
contracts. Accepting a broadcastable array is not evidence of a new mathematical
order convention.

Source evidence:

- `tests/unit/test_fractional_operators.py`, `test_convergence_order.py`,
  `test_alpha_gradients.py`, `test_vector_orders.py`;
- `tests/unit/test_fft_history.py`, `test_nonuniform_grids.py`,
  `test_graded_mesh_precision.py`, `test_integer_time_grids.py`;
- `tests/unit/test_fft_order_gradients.py` checks selected scalar/per-state FFT
  order derivatives against finite differences and full-history derivatives;
- The [FFT precision correction](audits/fft-order-gradients-2026-11-10/README.md)
  preserves the initially failing dtype assertions and corrected CPU/CUDA runs;
- [Windowed history](windowed-history.md), [recursive SOE](soe-history.md),
  `tests/unit/test_windowed_history.py` and `test_soe_recursive.py`.

## Deterministic solver combinations

All current deterministic solvers use scalar order and full history. State paths
convert the initial state to an array; general nested **state** pytrees are not
implemented merely because annotations use `PyTree`. Parameter and input pytrees
are supported where the model and documented input convention support them.

| Configuration | Grid / evaluation | Differentiation and evidence boundary |
| --- | --- | --- |
| `PredictorCorrector` | Uniform, matching positive `dt`; `weight_evaluation="direct"` default or `"stable"` | Selected analytic/refinement, state/parameter/order gradients, trailing shapes and JIT. Stable evaluation changes roundoff, not the PECE method/order. |
| `ImplicitPredictorCorrector` | Uniform, matching `dt`; direct or stable weights; fixed Newton iterations | Selected linear/stiff and nonlinear/coupled references, state/parameter/order gradients and residual diagnostics. Coarse steps can lose positivity even with a converged corrector. |
| `NonUniformPredictorCorrector` | Supplied increasing nodes; stable interval moments | Uniform consistency, graded refinement, state/parameter gradient checks and JIT. Do not infer blanket order-gradient coverage from the uniform solver tests. |
| `AdaptivePredictorCorrector` | Bounds supplied, mesh selected internally; default step doubling or experimental subdivision | Selected reference problems and frozen-mesh state/parameter/order sensitivities. Controller decisions and mesh topology are not differentiated. Completion is not a global-error guarantee. |
| `AdaptiveForcingQuadrature` with `BoundedForcing` (experimental) | Real float32/float64 bounds/states, adaptive mesh; prescribed C2 forcing and concrete global curvature bound | Conditional full-history nodal interpolation envelope, excluding roundoff/RHS-evaluation error. Independent 60-case audit, nonuniform consistency, tensor states, JIT and frozen-mesh state/parameter/order gradients. No state-feedback or dense-output bound. |

The stable uniform setting is a precision option, not an API stability-tier
promotion. Uniform direct/stable evidence is in [uniform weights](uniform-weights.md);
nonlinear limitations are in [the implicit audit](implicit-nonlinear.md).
The [selected scientific-acceptance policy](scientific-acceptance.md) separately
checks coherent measurements, reported/independent residuals and continuous
errors. Its historical direct float32 weak-decay failures are retained in the
[pre-repair bundle](audits/scientific-acceptance-2026-11-10/README.md). After
NUM-004's cancellation-safe direct-weight repair, all eight
[new report variants](audits/num004-direct-precision-2026-11-10/README.md) pass
the unchanged policy: 32 required-success cases are accepted, and 16 negative
controls remain rejected with passing policy flags. **Legacy direct
floating-point roundoff is not preserved.** The default direct selector,
full-history method, normalization, Newton updates/tolerances and explicit
stable interval-moment path are unchanged; the separate stochastic kernel is
untouched. This is selected-gate evidence, not arbitrary nonlinear/stiff
float32 accuracy. Local full CPU/CUDA suites pass; independent review and remote
multi-version/release gates are separate. Expected
rejections are never counted as accepted trajectories.

The later [NUM-005 accumulation candidate](audits/num005-stable-precision-2026-11-10/README.md)
replaces shared history contraction with an explicit working-dtype weighted
reduction. Its original four regressions and 25 compatibility cases pass on
CPU/CUDA, including private-helper complex semantics and shared solver paths;
this does not establish public complex implicit support. All eight frozen-policy
reports pass without changing their roles or budgets. Full CPU/CUDA suites pass
879 tests each; review and release are separate. Captured reduction-emitter
variation excludes a cross-compilation bitwise guarantee; it is not a broader
stiffness or adaptive-accuracy guarantee.

Adaptive limitations and paired-work evidence are in
[calibration](adaptive-calibration.md), [startup](adaptive-startup.md),
[subdivision](adaptive-subdivision.md), [runtime](adaptive-runtime.md),
[dense runtime](adaptive-runtime-dense.md), and [coupled cases](adaptive-coupled.md).
The restricted curvature-based contract and its assumptions are in
[bounded prescribed forcing](bounded-forcing.md).

Per-state solver orders, compressed solver histories, an adaptive/implicit
combination and dense output remain unimplemented. Adaptive results are padded;
use the accepted-prefix mask. Time-series input interpolation on adaptive meshes
is not supplied automatically.

## Precision, transformations and result containers

| Concern | Current boundary |
| --- | --- |
| Float32 / float64 | Both have case-specific CPU evidence. Float64 requires JAX x64 enabled. Input/order/default-dtype promotion can affect actual execution dtype; inspect returned arrays and record the x64 setting. Legacy SOE fitting uses JAX's default floating dtype. |
| Accelerator execution | JAX-native code targets a single accelerator. Selected CUDA float32 dense GL/RL history checks use independent discrete sums; dense histories and legacy SOE fitting request highest matmul precision. Uniform solver normalization has independent value, order-gradient and implicit residual checks in float32/float64. CPU tolerances/timings and selected checks are not general GPU/TPU validation. |
| JIT validation | Shape checks can run while tracing. Value-dependent order/grid checks generally require eager inputs; compiled callers must supply validated values. |
| Whole results | Frozen dataclasses are not registered whole-result JAX pytrees. Return numerical array fields and numeric diagnostics from compiled wrappers. |
| Serialization | Metadata has `to_dict()` exports where values are concrete; trajectory arrays need explicit serialization. Runtime context is captured explicitly with `current_provenance()`. |
| Import behavior | Package import and metadata dataclasses do not load JAX. Numerical calls and explicit runtime-provenance capture request it lazily. |
| Gradients | Evidence is target/path-specific. A differentiable JAX expression or a finite smoke-test gradient is not automatically a validated sensitivity across all configurations. Adaptive derivatives hold the realized mesh fixed. |

## Experimental layers

| Surface | Implemented / validated boundary |
| --- | --- |
| `nn.NeuralFODE` and loss/SGD helpers | Explicit parameter pytrees, solver-backed trajectories and optional observation transform. Synthetic recovery/gradient tests exercise the uniform baseline. Other solver combinations inherit their constraints; adaptive padding needs explicit loss handling. |
| `prob.simulate_stochastic` | Additive-noise fractional simulation on a uniform grid, scalar `1/2 < alpha < 1`, direct weight evaluation only. Fixed-key reproducibility and selected linear mean/variance evidence. Multiplicative/nonlinear stochastic accuracy and broad stochastic sensitivity claims remain outside this evidence. |
| Scalar-grid calibration / prediction | Gaussian likelihood, normalized scalar-grid posterior weights and weighted predictive summaries. Python orchestration is not a general JIT inference engine; SVI/MCMC remain future work. |
| Brain/observation/training/data/viz namespaces | Importable placeholders beyond the small implemented helpers above. No domain-model functionality or biological validity is implied. |

Evidence: `tests/unit/test_nn_fode.py`, `test_probabilistic_phase5.py` and the
versioned examples. The package remains research software with the existing
domain-model guardrails.

## Executable gates versus measurement audits

After `uv sync --locked --extra dev`, the aggregate command is:

```bash
uv run --frozen python -m benchmarks.numerical.validation_summary --check
```

Its **28 fixed rows** comprise eight baseline, twelve core-hardening and eight
history/bounded-forcing extension cases. The new rows check scaled-tail Caputo/RL analytic
behavior, recursive fitted-sum consistency, an order gradient, stable uniform
coefficients, explicit/implicit stable solver behavior, and a prescribed quadratic
forcing bound. Their configurations
and tolerances are recorded in `details`; see [methodology](methodology.md).
They do not cover every table entry above.

The larger audit commands have distinct check semantics. For example, SOE
accuracy audits can pass with poor but finite legacy fits, while an implicit
measurement audit can retain correctly labeled nonfinite trajectories. Read each
audit's definition; a successful process is not a universal accuracy certificate.

Remaining Phase A work includes general state-dependent adaptive error calibration, broader
nonlinear/stiff and mixed-scale precision evidence, tolerance-driven SOE fit
selection, reverse-mode/accelerator peak-memory measurements, and a reviewed
clean-source release dossier. The matrix records these gaps rather than treating
implemented combinations as automatically validated.
