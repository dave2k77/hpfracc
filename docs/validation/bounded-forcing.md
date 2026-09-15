# Conditional Adaptive Bound for Prescribed Forcing

Status: **Experimental, unreleased**, Phase A. Local CPU evidence on 2026-09-15.

`BoundedForcing` and `AdaptiveForcingQuadrature` add an explicitly restricted
adaptive path for `D_C^alpha y = q(t, params)`. The supplied forcing has no
evolving-state argument and must be twice continuously differentiable in time.
The caller provides a global componentwise bound on its second derivative.
This additional information permits a full-history nodal interpolation-error
bound, including error from previously accepted panels.

The [earlier calibration](adaptive-calibration.md) and
[cubic sampling blind spot](adaptive-startup.md) remain limitations of the
general adaptive PECE indicators. This extension does not calibrate those
indicators or change their defaults.

## Mathematical contract

For fixed parameters, inputs and explicit random key, assume a deterministic
prescribed function `q` on `[t0, T]`, scalar `0 < alpha < 1`, and finite
nonnegative componentwise constants `M_i` satisfying

```text
q_i is C2 on [t0,T]
abs(q_i''(s)) <= M_i for every s in [t0,T]
y_i(t) = y0_i + integral[t0,t] (t-s)^(alpha-1) q_i(s) ds / Gamma(alpha)
```

The bound must hold over the whole parameter domain used, including perturbed
parameters in a sensitivity study. It is a caller-supplied assumption, not an
enclosure certified by sampling derivatives. `M` can be scalar or broadcastable
to the trailing state shape. It bounds **curvature**, not forcing amplitude.

On each accepted panel `[t_j,t_{j+1}]` of width `h_j`, let `L_j q` be the
linear interpolant of the two prescribed forcing values. The interpolation
remainder obeys

```text
abs(q_i(s) - L_j q_i(s))
    <= M_i (s-t_j)(t_{j+1}-s)/2
    <= M_i h_j^2/8.
```

The fractional kernel is positive. Therefore the product-integration state
`yhat_n = y0 + I^alpha(Lq)(t_n)` satisfies, in exact arithmetic,

```text
abs(yhat_n_i - y_i(t_n)) <= E_n_i
E_n_i = M_i/8 * sum[j=0,...,n-1] h_j^2 P_nj
P_nj = ((t_n-t_j)^alpha - (t_n-t_{j+1})^alpha) / Gamma(alpha+1).
```

All accepted panels appear in this sum. The implementation reuses the stable
nonuniform solver moments: predictor weights supply positive kernel masses,
and corrector weights integrate the linear forcing interpolant. There is no
state-feedback propagation term because `q` is prescribed. On the same realized
mesh, the states agree with `NonUniformPredictorCorrector` for this wrapped
forcing up to floating-point evaluation differences.

Acceptance requires finite data and, for every component,

```text
E_n_i <= (atol + rtol*abs(yhat_n_i)) / (1+rtol).
```

This scale is conservative with respect to the unknown reference state. Indeed,
`abs(yhat_n_i) <= abs(y_i(t_n)) + E_n_i`; substitution into the acceptance
inequality yields `E_n_i <= atol + rtol*abs(y_i(t_n))`. This is an exact-arithmetic
implication under the stated assumptions, not a total floating-point error
certificate.

The next-step proposal allocates remaining tolerance after the old-history
contribution to the new panel, whose majorant scales as `h^(alpha+2)`.
Every trial recomputes the complete envelope. No common scaling law for the
old-history contribution or changing relative scale is assumed. A decreasing
relative budget can prevent completion after coarse history has been retained;
the solver reports failure rather than accepting an over-budget trial.

## Usage and operational contract

```python
import jax.numpy as jnp
import hpfracc as hp

def forcing(t, params, *, rng_key=None, inputs=None):
    return params["amplitude"] * t**2

# Valid for abs(amplitude) <= 1 throughout the intended parameter domain.
model = hp.solvers.BoundedForcing(forcing, curvature_bound=2.0)
solver = hp.solvers.AdaptiveForcingQuadrature(
    order=0.5, atol=1e-3, rtol=0.0, first_step=1.0, max_steps=128,
)
result = hp.solvers.simulate(
    model=model, solver=solver,
    ts=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
    initial_state=jnp.asarray(0.0, dtype=jnp.float32),
    params={"amplitude": 1.0},
)
d = result.solver_info.diagnostics
assert d["success"]
accepted_states = result.latent_state[:d["n_nodes"]]  # Eager slicing only.
accepted_bounds = d["interpolation_error_bound"][:d["n_nodes"]]
```

- The new solver requires a `BoundedForcing` model. Its callable adapter accepts
  the standard dynamics signature while discarding the evolving state.
- `ts` and initial states must be real float32/float64 arrays. `ts` supplies
  integration bounds, not requested output samples; a finite singleton is allowed.
  Fractional orders remain scalar. Curvature is concrete, frozen configuration;
  unbroadcastable, nonfinite, negative or unrepresentable bounds are rejected.
  Narrowing the stored bound to the working dtype rounds upward where needed.
- `max_steps` is an attempt budget. Time, state and envelope arrays have
  `max_steps+2` entries, with repeated-tail padding and an accepted-prefix mask.
  Check `success`, `valid_mask` and the failure flags before reducing a trajectory.
  Budget exhaustion, nonfinite data and unresolved acceptable steps retain the
  accepted prefix. The minimum-step threshold is `1e-12*abs(T-t0)`; it never
  forces acceptance. A singleton completes with zero attempts if initial data
  are finite.
- Defaults: `rtol=0`, `atol=1e-5`, `max_steps=1024`, initial proposal one percent
  of the span, `safety=0.8`, `min_factor=0.2`, `max_factor=5`.
- Diagnostic tags include `estimator="curvature_bound"`,
  `bound_scope="conditional_nodal_interpolation_error"` and
  `roundoff_included=False`. Eager metadata exports to JSON; compiled wrappers
  return numerical fields and numeric diagnostics, not the whole dataclass.
- Time-grid/order value checks generally require concrete inputs. Validate them
  outside JIT. All integration, weight and controller arithmetic is JAX-native;
  host NumPy is used only for concrete curvature configuration validation and
  representability checks, following existing order/grid validation practice.
- Successful state/parameter/order gradients freeze the realized mesh and all
  controller decisions. Curvature configuration and diagnostic bounds are not
  differentiable outputs. Time-bound and failure-path gradients are not claimed.
- Full-size buffers and full-history trials give quadratic worst-case work in
  the attempt budget, linear forward history storage and a bounded scan graph.
  There is no new runtime, peak-memory or accelerator benchmark claim.

## Independent evidence

`tests/unit/test_bounded_forcing.py` covers quadratic tolerance refinement at
orders 0.3/0.5/0.8 in both precisions, a 70-digit independent reconstruction of
the entire envelope, the cubic case, componentwise relative tolerances, shifted
time origins and tensor states, nonuniform consistency, JIT and frozen-mesh
state/parameter/order finite differences. It also checks accepted-prefix failures,
last-attempt completion, singleton/resolution behavior, invalid configuration,
bounded graph size, a deliberately false curvature assertion and roundoff beyond
a zero interpolation envelope.

The predeclared [protocol](audits/bounded-forcing-2026-09-15/protocol.json)
specifies five cases, three orders and two absolute tolerances per precision:

| Case on `[0,1]` | Curvature | Continuous reference |
| --- | --- | --- |
| `q=1+2t`, `y0=0` | 0 | Analytic fractional monomials. |
| `q=t^2`, `y0=0` | 2 | `2t^(alpha+2)/Gamma(alpha+3)`. |
| `q=t^2-2t^3/3`, `y0=0` | 2 | Analytic quadratic minus cubic integral. |
| `q=sin(2t)`, `y0=0` | 4 | Independent SciPy weighted continuous quadrature. |
| `q=(1+2t,t^2)`, `y0=(1,0.5)` | `(0,2)` | Componentwise analytic integrals, `rtol=1e-3`. |

All other cases use `rtol=0`. Orders are 0.3/0.5/0.8, `atol` is 1e-3/1e-4,
first step is 1, and attempt budget is 512. Oracles use the actual stored order
and accepted nodes. The report independently reconstructs the panel envelope
in host float64 arithmetic; tests separately check that reconstruction at
70-digit precision and cross-check continuous polynomial oracles by quadrature.

Both checked sweeps passed: **60/60 completed configurations** on Windows 11,
Python 3.13.13, JAX/jaxlib 0.10.1, CPU `cpu:0`. Float32 ran with x64 disabled;
float64 ran with x64 enabled. This is dirty-tree development evidence based on
`e19f0c1`, with source and CSV digests in the archived JSON files.

| Selected result | Float32 | Float64 |
| --- | --- | --- |
| Worst reference-scaled error across 30 configurations | 0.663393 | 0.662962 |
| Largest accepted envelope / conservative budget | 0.998958 | 0.998958 |
| Quadratic, alpha 0.5, atol 1e-4: max true error | 6.187706e-5 | 6.179732e-5 |
| Same case: max envelope | 9.706317e-5 | 9.706314e-5 |
| Same case: accepted / rejected steps | 55 / 2 | 55 / 2 |
| Affine: largest true error despite zero interpolation envelope | 1.335093e-6 | 2.220446e-15 |

`--check` requires the full unique configuration set, coherent completed output,
independent envelope agreement, accepted envelope/budget ratio at most `1+2e-6`,
and reference-scaled true error at most one. The separate comparison of true
error to the interpolation envelope allows predeclared absolute roundoff slack
of `3e-6` (float32) or `3e-12` (float64). This slack is an audit allowance, not a
solver error bound or a relaxation of the reference-scaled accuracy check.
Corruption tests reject missing cases, false success and zeroed envelopes.

Artifacts:

- [Float32 CSV](audits/bounded-forcing-2026-09-15/bounded_forcing_float32.csv)
  and [source/environment JSON](audits/bounded-forcing-2026-09-15/bounded_forcing_float32.json).
- [Float64 CSV](audits/bounded-forcing-2026-09-15/bounded_forcing_float64.csv)
  and [source/environment JSON](audits/bounded-forcing-2026-09-15/bounded_forcing_float64.json).

Reproduce after `uv sync --locked --extra dev` (PowerShell):

```powershell
$env:JAX_ENABLE_X64 = '0'
uv run --frozen python -m benchmarks.numerical.bounded_forcing_validation --check
$env:JAX_ENABLE_X64 = '1'
uv run --frozen python -m benchmarks.numerical.bounded_forcing_validation --dtype float64 --check
```

Use `--output-dir` with a new directory when archiving another experiment.
The aggregate includes one additional fixed float32 quadratic bound check;
it does not replace the full sweep.

## Development verification checkpoint

Completed on the same Windows CPU environment, with the bounded-forcing
extension and preceding tooling consolidation present in the working tree:

| Command | Result |
| --- | --- |
| `uv run --frozen python -m pytest` | **699 passed**, one performance test deselected by the normal configuration; exit 0 in 761.19 seconds. |
| `uv run --frozen python -m benchmarks.numerical.validation_summary --check` | **28/28 rows passed**, exit 0. |
| `uv run --frozen python -m benchmarks.numerical.history_validation --check` | **8/8 rows passed**, exit 0 with global x64 enabled; fixed cases still execute in float32. |
| `uv run --frozen ruff check .` | Passed. |
| `uv run --frozen mkdocs build --strict` | Passed. |
| `git diff --check` | Passed. |

The full suite and aggregate ran with global x64 disabled; scoped float64 tests
enable it locally and restore the setting. The two larger audit sweeps above
each exited 0. This is a local development checkpoint on dirty `main`, based on
`e19f0c1`; it is not remote CI evidence or release sign-off. Package version
remains `0.1.0a1`. The earlier
[662-test consolidation checkpoint](../developer/consolidation-2026-09-15.md)
records the preceding state and retains its original counts.

## Limits and next calibration targets

The reported envelope excludes floating-point accumulation, gamma/moment
evaluation, input rounding and forcing-evaluation error. In particular, affine
forcing has zero interpolation error yet nonzero floating-point error; success
can therefore coexist with total error above an extremely small requested
tolerance. Supplying false curvature also invalidates the conditional bound.
The tests make both limitations executable.

State-dependent/nonlinear feedback, forcing without a global C2 bound (including
common weak-startup forcings), stochastic evolution, dense output, per-state
orders and compressed solver histories are outside this contract. A solution
itself can have a fractional startup singularity even when its prescribed
forcing is smooth; the C2 assumption is on `q`, not `y`. Broader adaptive FDE
calibration still needs justified history-error and feedback propagation control.
Phase A remains active and both new public classes remain experimental.
