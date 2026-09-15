# Fractional Solvers

HPFRACC v0.1 starts with deterministic Caputo initial value problems:

```text
D_C^alpha y(t) = f(t, y(t), params)
y(0) = y0
0 < alpha < 1
```

## Predictor-Corrector Solver

`hp.solvers.PredictorCorrector` configures a fixed-step full-history
predictor-corrector method. `hp.solvers.simulate` returns a `SimulationResult`
with the time grid, latent trajectory, and `SolverInfo` diagnostics.

The time-stepping recurrence is evaluated with a single `jax.lax.scan` over a
preallocated history buffer rather than an unrolled Python loop. The solver is
therefore `jax.jit`-traceable with a bounded autodiff graph whose size does not
grow with the number of time steps, which keeps compilation and reverse-mode
gradients tractable for long trajectories. The arithmetic cost is unchanged: it
remains `O(n**2)` because each step weights the full history.

The dynamics callable should accept:

```python
f(t, state, params, *, rng_key=None, inputs=None)
```

The returned trajectory always has the time axis first. Scalar initial states
produce shape `(time,)`; array initial states preserve arbitrary trailing
dimensions as `(time, ...)`.

## Time Grids and Input Validation

All solvers require a nonempty, one-dimensional grid of finite, strictly
increasing time nodes. A single finite node is allowed. Validation includes
single-node inputs; `NaN` and infinite times are not valid bounds.

`PredictorCorrector` and `ImplicitPredictorCorrector` additionally require a
finite positive `dt` and uniform grid spacings that agree with it. The rounding
tolerance scales with the grid magnitude and dtype, without a unit-scale or
absolute floor that could mask large relative errors on tiny grids. These checks
also apply to the experimental stochastic helper's uniform time grid.

`NonUniformPredictorCorrector` uses the supplied node spacings without a scalar
`dt`. `AdaptivePredictorCorrector` uses only the first and last nodes as bounds,
not as requested output samples. Adaptive time buffers and step-size control use
a common floating dtype derived from the time nodes and initial state. Integer
bounds such as `[0, 1]` are promoted, and higher input precision is retained when
JAX x64 is enabled.

Shape checks run during tracing, but value checks require concrete arrays and
are skipped when time nodes are JAX tracers. Validate grids outside transformed
calls before passing them as dynamic `jax.jit` arguments; compiled simulation
does not provide runtime rejection of invalid grids. JIT examples should return
array fields such as `.latent_state`, not the unregistered `SimulationResult`
dataclass itself.

## Graded-Mesh Precision

On a strongly graded mesh, an early interval can be much narrower than its lag
from the current time. Directly subtracting lag powers then loses significant
digits, and subtracting two such moments can even produce negative corrector
weights. This can occur in float64 as well as float32.

The nonuniform and adaptive solvers evaluate the same product-integration
weights through shared, stable power-kernel moments. Write `A = t_n - t_j`,
`B = t_n - t_{j+1}`, `h = t_{j+1} - t_j`, and `r = h/A`. The interval width is
computed from adjacent nodes, not reconstructed as `A-B`. For kernel exponent
`0 < beta < 1`, the left/right nodal moments are:

```text
L = h A^(beta-1) sum_{k>=0} c_k r^k / ((k+1)(k+2))
R = h A^(beta-1) sum_{k>=0} c_k r^k / (k+2)
c_0 = 1, c_k = c_{k-1} (k-beta)/k
```

For `r <= 1/4`, the implementation retains terms `k=0,...,25`. Since
`0 < c_k <= 1`, the relative truncation remainder in either moment is bounded
by `2 r^26 / (28 (1-r))`, at most approximately `2.12e-17`. This bounds the
series remainder, not total floating-point error. Larger intervals use the
closed forms with `expm1` and the directly computed `B/A`; endpoint and inactive
branches are kept finite for autodiff. Zero-width trial intervals in completed
adaptive scans contribute zero.

This is a stable evaluation of the existing full-history discretization, not
weight clipping, reduced precision, memory truncation, or a different numerical
scheme. The Caputo L1 operator reuses the constant moment with `beta=1-alpha`;
the solvers use `beta=alpha` and retain their gamma normalization.

Tests compare individual weights with 70-digit mpmath references on the actual
rounded input nodes, including both sides of the series switch and near-zero
lower lags. They also check positivity, constant/affine moment identities, and
JIT/order-gradient behavior. The fix cannot recover distinct time nodes already
rounded to the same number or guarantee accuracy for every long-history problem;
extreme lag/width scales can still underflow or overflow intermediate quantities.
Float64 and problem-specific refinement studies remain important.

## Uniform Weight Evaluation

Uniform explicit and implicit solvers now offer `weight_evaluation="stable"` as
an opt-in alternative to the default direct power differences. It combines the
same interval moments described above on unit lag intervals, preserving the
full-history method and separate current-endpoint coefficient. The
[coefficient and paired solver evidence](../validation/uniform-weights.md) shows
the improvement in float32 roundoff; discretization error, coarse-grid positivity
loss and working-precision model/accumulation effects remain separate limits.

## Implicit Residual Criterion

The implicit solver still executes a fixed number of Newton iterations. After
the final update, it reuses the dynamics value needed for the next history entry
to assess the corrector equation `y = C + scale*f(t_n, y)`:

```text
r_i = y_i - C_i - scale*f_i(t_n, y)
q = max_i |r_i| / (newton_atol + newton_rtol*max(|y_i|, |C_i|))
```

A step passes if the relevant quantities are finite and `q <= 1`. The maximum
is over all state components; a failing component cannot be hidden by averaging
it with many components having small residuals. Defaults are `newton_rtol=1e-6`
and `newton_atol=1e-8`. Tolerances must be representable at the working precision
to be meaningful.

The check does not alter updates, perform extra Newton iterations, retry a step,
or change the mesh. Diagnostic values are excluded from differentiation while
the trajectory retains its existing gradient paths. Per-step flags and residuals
remain available as arrays inside JIT; see the
[diagnostic API](../api/contract.md#implicit-corrector-diagnostics) for eager
serialization and warning behavior.

This is an algebraic residual check, not a local truncation-error estimate or
a guarantee of proximity to a unique root. Poor conditioning can make a small
residual misleading. Newton damping, condition estimation, convergence-driven
termination, and automatic failure recovery remain future work.

The [nonlinear/stiff audit](../validation/implicit-nonlinear.md) provides broader
case-specific evidence and an important counterexample. For `f=-k*y`, the first
implicit step is `(1-alpha*mu)/(1+mu)`, with `mu=k*dt^alpha/Gamma(alpha+2)` and
`y0=1`. It can be negative on a coarse grid despite a positive continuous solution
and a converged algebraic solve. The audit also separates working-precision
residuals from independent float64 reconstruction; cancellation in uniform
history weights can matter in float32. Neither increasing Newton iterations nor
tightening diagnostic tolerances alone fixes these integration/precision limits.

## Adaptive Termination

By default, adaptive trials retain the step-doubling RMS criterion. The
experimental `error_estimator="subdivision"` option adds four quarter steps,
uses the maximum componentwise C/F/Q discrepancy, and makes bounded-factor
step proposals; see [the prototype](../validation/adaptive-subdivision.md).
Both modes retain the full PECE step. Acceptance requires finite trial
states/dynamics and normalization, representable required interior nodes, and
normalized indicator at most one. Non-finite trials fail fast; recovery remains
future work. Finite over-tolerance trials are rejected and retried at smaller
steps until the attempt budget or small-step threshold is reached.

The small-step failure threshold is `1e-12 * abs(t_end - t0)`. It is not a
forced-acceptance rule: if the estimate still fails at or below this threshold,
the solver reports a step-size failure instead of silently accepting the step.
Shorter steps may be accepted if their indicator passes and the required trial
nodes are representable. The default needs a midpoint; subdivision also needs
distinct quarter points. Unresolvable trial nodes cause a step-size failure.

Completion requires reaching the actual clipped endpoint, not proximity within
an absolute tolerance. This prevents tiny time spans from being reported as
complete after only a fraction of the requested duration. The bounded scan skips
further state updates after completion or failure. Counts and the output mask
describe only the initial node and accepted prefix; a terminating failed trial
counts as rejected but is not appended to that prefix.

The [adaptive diagnostic API](../api/contract.md#adaptive-completion-and-failure-status)
distinguishes completion, budget exhaustion, non-finite failure, and step-size
failure in eager and compiled calls. A success flag certifies this operational
completion criterion, not calibrated global error control or biological realism.
The [independent calibration audit](../validation/adaptive-calibration.md) shows
that the current Richardson-like normalization underestimates even the retained
first-panel error for quadratic forcing. It is a heuristic indicator, not a
calibrated local or global bound. Shared older-history error is invisible to the
trial difference in that example. Smaller absolute errors under tolerance
refinement do not resolve this calibration defect; a general correction remains
open and cannot be justified by a blanket multiplier from one case.

The [startup analysis](../validation/adaptive-startup.md) distinguishes the
`h^(2*alpha)` autonomous startup discrepancy from `h^(alpha+beta)` for time-only
power forcing. It also gives a smooth cubic counterexample where the pair
indicator vanishes. These results motivate additional independent error probes,
not an unqualified solver-wide change of multiplier or exponent.

Integer bounds must be exactly representable in the selected floating time
dtype. A conversion-induced collapse is a precision failure, not a zero-duration
success; the check also handles integer maxima where a saturating reverse cast
could otherwise hide lost precision. The check operates on the supplied JAX
array values and cannot recover precision already lost before that boundary.
State/history buffers use the common initial-state and dynamics dtype, so a
lower-precision dynamics result does not silently overflow a finite initial state.

## Conditional Prescribed-Forcing Bound

The experimental `BoundedForcing` / `AdaptiveForcingQuadrature` path restricts
the right-hand side to prescribed C2 forcing `q(t, params)` with a supplied
global componentwise curvature bound `M >= abs(q'')`. Linear interpolation on
panel `j` has error at most `M*h_j^2/8`. Integrating this majorant against the
positive fractional kernel over **all** accepted panels gives a conditional
nodal interpolation-error envelope. A conservative relative scale implies the
reference-relative tolerance in exact arithmetic. See the
[full proof and calibration evidence](../validation/bounded-forcing.md).

This path shares the nonuniform product-integration weights and freezes the
realized mesh for state/parameter/order gradients. It takes real floating bounds
and states, retains full history, and returns masked fixed-budget output. The
envelope excludes roundoff and forcing-evaluation error; it is not a bound for
arbitrary state-dependent or weak-startup forcing. The general adaptive PECE
indicators above retain their existing semantics.

## Validation

The first validation target is the scalar linear Caputo FDE:

```text
D_C^alpha y(t) = lambda y(t)
y(0) = 1
```

Its reference solution is:

```text
y(t) = E_alpha(lambda t^alpha)
```

where `E_alpha` is approximated by a truncated Mittag-Leffler series in tests
and validation reports.

## Limitations

- Only scalar `0 < alpha < 1` orders are supported.
- The explicit and implicit uniform solvers use fixed positive timesteps;
  nonuniform and adaptive explicit variants are also available.
- The method uses full history, so runtime grows quadratically with the number
  of time samples.
- Implicit integration reports residual convergence/failure but still uses
  fixed-iteration Newton without convergence-driven termination or retries.
- Adaptive output is padded to `max_steps + 2` entries. Inspect `success`,
  `valid_mask`, and failure flags in `solver_info.diagnostics`; these remain
  arrays under JIT. Failed runs can return a retained partial trajectory.
  Gradients freeze the realized mesh and do not differentiate the controller;
  failure-path gradients are not guaranteed.
- Solver-history compression, combined adaptive/implicit integration, dense
  output, and multi-device execution remain future work. Experimental stochastic
  FSDE support is provided separately through `hp.prob.simulate_stochastic`.
