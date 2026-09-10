# Experimental Subdivision Indicator

Status: Unreleased, experimental, opt-in. Not a calibrated error-bound method.

`AdaptivePredictorCorrector` now accepts `error_estimator="subdivision"`. The
existing `"step_doubling"` mode remains the default. The prototype adds independent
quarter-step information to detect failures such as the
[cubic pair-cancellation example](adaptive-startup.md#a-zero-indicator-cubic-counterexample).
It does not claim to detect every possible cancellation or accumulated-history
error, and it does not turn `rtol`/`atol` into global-error bounds.

## Select explicitly

```python
import hpfracc as hp

solver = hp.solvers.AdaptivePredictorCorrector(
    order=0.5,
    rtol=1e-4,
    atol=1e-6,
    max_steps=512,
    error_estimator="subdivision",
)
```

Unknown estimator names raise `ValueError`. The dynamics signature, result type,
full-history discretization, scalar-order scope, and public output length
`max_steps + 2` are unchanged. The experimental option is not part of the tagged
a1 release.

## Indicator and controller

Every trial starts from the same accepted history and computes:

- `C`: one full PECE step;
- `F`: two sequential half PECE steps;
- `Q`: four sequential quarter PECE steps, using the corrected RHS at each
  temporary quarter node.

The half/quarter histories are temporary. Accepted trajectories retain **C**,
not an extrapolated or quarter-step value, so they remain the nonuniform coarse
PECE trajectory on the realized mesh.

For each state component `i`, the new indicator is:

```text
D_i = max(abs(C_i-F_i), abs(C_i-Q_i), abs(F_i-Q_i))
S_i = atol + rtol*max(abs(C_i), abs(F_i), abs(Q_i))
rho = max_i D_i/S_i.
```

There is no Richardson divisor. The norm is a componentwise maximum across the
state/batch array, unlike the default mode's RMS. Inactive components therefore
do not dilute another component's discrepancy. This is a heuristic discrepancy
indicator, not an independent estimate of the true error.

The bounded-factor proposal deliberately avoids assuming one universal startup
power:

```text
proposal = 0.5  if rho > 1
           2.0  if rho < 0.25
           1.0  otherwise
factor = clip(safety*proposal, min_factor, max_factor).
```

With default safety this means shrinking on rejection, growing only when the
indicator is comfortably small, and modestly shrinking otherwise. The usual
endpoint clipping, attempt budget, and small-step threshold still apply. This
policy is experimental, not an optimal or universally calibrated controller.

## Precision, status, and differentiation

- All quarter targets must be strictly increasing and representable. A mesh that
  can represent a midpoint but not distinct quarter points fails with
  `step_size_underflow` in this mode.
- Every quarter-state and its corrected RHS participates in finite checks.
  Quarter-only nonfinite values terminate without appending the failed trial.
- Extra internal trial slots prevent out-of-bounds writes on the final allowed
  attempt. They are discarded from public arrays and the output mask.
- Completion/failure flags, counts, padding, and `valid_mask` keep their existing
  meaning. Success is operational completion, not guaranteed accuracy.
- The mesh and indicator/controller decisions remain frozen for differentiation.
  Tests check compiled outputs/status, initial-state and order gradient finiteness,
  and parameter-gradient agreement with the nonuniform solver on the realized
  mesh and finite differences.

Diagnostics additionally identify `estimator`, `estimator_stability`,
`error_norm`, and `trial_substeps`. The new mode reports `subdivision`,
`experimental`, `max`, and `4`, respectively; the default reports `step_doubling`,
`provisional`, `rms`, and `2`. `trial_substeps` names the finest subdivision, not
the total work of a trial.

## Measured comparison

Run the independent scalar comparison with:

```bash
uv run --frozen python -m benchmarks.numerical.subdivision_comparison --check --output-dir docs/validation/audits/subdivision-prototype
JAX_ENABLE_X64=1 uv run --frozen python -m benchmarks.numerical.subdivision_comparison --dtype float64 --check --output-dir docs/validation/audits/subdivision-prototype
```

The matrix has four problems, two tolerances, two modes, and two precisions.
All 32 recorded runs completed within 512 attempts. The problems use the same
continuous oracles as the previous audits, plus weak forcing `t^0.25`, whose
solution is `Gamma(1.25)*t^(alpha+0.25)/Gamma(alpha+1.25)`.
Initial steps are 1 for the cubic blind spot and 0.01 for the other cases.
`atol=0.01*rtol`; all intervals are `[0,1]`.

At `rtol=1e-4` in float64, errors at the returned accepted nodes were:

| Case | Default max absolute error | Subdivision max absolute error | Default accepted steps | Subdivision accepted / rejected | Subdivision max reference-scaled error |
| --- | --- | --- | --- | --- | --- |
| Cubic cancellation | 7.164312e-3 | 4.952148e-5 | 1 | 63 / 8 | 2.456370 |
| Weak power forcing | 3.219028e-4 | 1.553992e-4 | 60 | 96 / 15 | 1.929041 |
| Half-order decay | 2.450158e-4 | 5.872097e-5 | 89 | 387 / 4 | 0.664889 |
| Manufactured feedback | 2.878880e-4 | 4.056411e-5 | 86 | 284 / 2 | 1.577596 |

The cubic maximum absolute error is about 145 times smaller in this selected
comparison, and the previously accepted blind-spot trial is rejected. However,
several reference-scaled errors still exceed one. Comparisons use each method's
own accepted mesh and are **not matched-work comparisons or dense-output bounds**.

The extra half/quarter trials and often larger accepted histories cost more work.
No runtime speedup, memory compression, or general efficiency claim is made.
The prototype remains full-history and its worst-case cost is still quadratic
in the attempted-step budget. This is a correctness-oriented safeguard, not
promotion of Phase A performance or error-control claims.

`--check` only enforces complete finite measurements; incomplete runs leave error
fields empty. It does not certify global error/tolerance at most one or universal
improvement over the default.

## Evidence and compatibility

- [Float32 comparisons](audits/subdivision-prototype/subdivision_comparison_float32.csv)
  and [provenance](audits/subdivision-prototype/subdivision_comparison_float32.json).
- [Float64 comparisons](audits/subdivision-prototype/subdivision_comparison_float64.csv)
  and [provenance](audits/subdivision-prototype/subdivision_comparison_float64.json).
- `tests/unit/test_adaptive_subdivision.py`: cancellation rejection, analytic
  accuracy, dtypes/shapes/final slots, nonfinite and resolution failures,
  singleton behavior, componentwise normalization, and JIT/gradient checks.
- `tests/unit/test_subdivision_comparison.py`: measured improvement for the cubic
  example and suppression of incomplete-run accuracy claims.

Two default-mode linear-decay trajectories/grids captured before the source edit
were bitwise identical afterward on the same CPU environment. This is selected
compatibility evidence, not a claim of cross-backend bitwise identity. Existing
regressions continue to exercise the default mode.

Earlier audit bundles remain historical snapshots of pre-prototype source;
their source hashes intentionally differ from this additive implementation.
Do not overwrite those bundles to make fingerprints match. New comparison JSON
identifies its own source, environment, dirty-tree state, and CSV hash; it is not
a release sign-off or a substitute for archiving reviewed source.

## Nonlinear and long-span stress follow-up

The follow-up audit adds three scalar half-order problems:

- `nonlinear`: `D^0.5 y = 2*t^1.5/Gamma(2.5) - (y^2-(1+t^2)^2)`.
- `strong_feedback`: the same equation with feedback coefficient 10.
- `long_decay`: `D^0.5 y = -y`, extended from `[0,1]` to `[0,16]`.

All start at `y(0)=1`. The first two have the exact solution `1+t^2` on
`[0,1]`; the third has `erfcx(sqrt(t))`. These references do not use solver
weights or refined solver trajectories. Continuous Caputo quadrature tests
independently check both the manufactured forcing and decay equation, including
a wrong-RHS negative control. Strong feedback probes a more
restrictive explicit integration regime; it is not a general stiff-solver test.
Long-span decay exercises larger retained histories, not memory compression.

The sweep uses `rtol = 1e-2, 3e-3, 1e-3, 3e-4, 1e-4`, `atol=0.01*rtol`, and
`first_step=0.01`. Run it with:

```bash
JAX_PLATFORMS=cpu uv run --frozen python -m benchmarks.numerical.subdivision_stress --max-steps 2048 --output-dir docs/validation/audits/subdivision-stress/budget2048
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --frozen python -m benchmarks.numerical.subdivision_stress --dtype float64 --max-steps 2048 --output-dir docs/validation/audits/subdivision-stress/budget2048
```

Add `--check` to require the exact case/tolerance/mode sweep without missing or
duplicate rows, every trajectory to complete, and at least one eligible work
pair per case. **Both recorded 2,048-attempt checked commands returned 1**:
the incomplete runs below are retained as evidence, not discarded to pass a gate.
This option does not test an error bound or demand subdivision improvement.

### Work definition and comparison limits

Accepted-step counts alone omit rejected trials and subdivision overhead.
For `A = n_accepted + n_rejected` and attempt budget `B`, the source-level
work proxy records:

| Quantity | Default | Subdivision |
| --- | --- | --- |
| PECE weight-builder evaluations | `3*A` | `7*A` |
| RHS evaluations, including the initial call | `1+5*A` | `1+13*A` |
| Padded history-weight slots | `3*A*(B+2)` | `7*A*(B+4)` |

Each weight-builder constructs predictor/corrector weights over the fixed-size
buffer, including masked slots. The last row counts one slot per builder, not
FLOPs, active history panels, memory bytes, or runtime. Test-only JAX callbacks
verify the counts against executed calls, including rejected trials. Compiler
optimization, RHS complexity, and backend costs are not modeled.

Pairs must share the problem, precision, order, bounds, first step, and attempt
budget, and both must complete. Every pair satisfying
`max(weight_slots)/min(weight_slots) <= 1.2` is exported, independently of its
errors; there is no interpolation, extrapolation, or selection for improvement.
These are **approximately work-matched proxy comparisons**, not equal-runtime
benchmarks. Maximum errors use each method's accepted nodes, not a common dense
grid. Endpoint errors share the exact physical endpoint.

### Recorded results

The WSL CPU run used the locked JAX 0.10.1 environment. At 2,048 attempts,
27 of 30 runs completed in each precision. Subdivision exhausted the budget for
strong feedback at `rtol=1e-4` and long decay at `rtol=3e-4` and `1e-4`.
Every default run completed. Incomplete runs have blank full-span error fields.

Six eligible pairs were found in each precision, covering all three cases.
All had **larger maximum accepted-node errors with subdivision**. The complete
float64 pair matrix is below; error ratios are subdivision divided by default.

| Case | Default rtol | Subdivision rtol | Work mismatch | Max-error ratio | Endpoint-error ratio |
| --- | --- | --- | --- | --- | --- |
| Nonlinear | 3e-4 | 1e-2 | 11.09% | 8.038 | 0.422 |
| Nonlinear | 1e-4 | 3e-3 | 0.26% | 9.849 | 7.816 |
| Strong feedback | 1e-4 | 1e-2 | 1.13% | 122.023 | 50.566 |
| Strong feedback | 1e-4 | 3e-3 | 5.28% | 38.026 | 7.133 |
| Long decay | 3e-4 | 1e-2 | 6.13% | 4.608 | 45.811 |
| Long decay | 1e-4 | 3e-3 | 0.60% | 3.320 | 0.219 |

The float32 matrix has the same tolerance pairs and qualitative max-error result;
strong-feedback trajectories show precision-sensitive mesh/error differences.
Endpoint rankings are mixed in both precisions. Equal-tolerance comparisons can
favor subdivision while roughly equal-work comparisons favor tightening the
default tolerance. Neither observation establishes universal dominance.

The earlier 1,024-attempt float64 pilot completed 23 of 30 runs and found three
pairs, with no eligible strong-feedback pair. That pilot is preserved separately.
Across the pilot and two 2,048-attempt sweeps, 90 measured rows are archived:

- [Pilot CSV](audits/subdivision-stress/subdivision_stress_float64.csv) and
  [provenance/pairs](audits/subdivision-stress/subdivision_stress_float64.json).
- [Float32 CSV](audits/subdivision-stress/budget2048/subdivision_stress_float32.csv)
  and [provenance/pairs](audits/subdivision-stress/budget2048/subdivision_stress_float32.json).
- [Float64 CSV](audits/subdivision-stress/budget2048/subdivision_stress_float64.csv)
  and [provenance/pairs](audits/subdivision-stress/budget2048/subdivision_stress_float64.json).

The archived benchmark fingerprints precede a reporting-only refinement that
rejects missing/duplicate sweep rows. All archived matrices were independently
verified complete; measurement definitions, solver source, and CSV data are
preserved. The subsequent [runtime audit](adaptive-runtime.md) extracts the
same result checks into a shared helper. Historical fingerprints are retained.

The new evidence does **not** support promotion as a generally more efficient
method. The option remains an experimental cancellation safeguard. No solver
implementation or default was changed for this follow-up.

## Before promotion

Keep the option experimental. The selected nonlinear/long-span proxy comparisons
and [synchronized runtime follow-up](adaptive-runtime.md) do not close the broader
nonlinear/stiff or efficiency gates. Extend finite-step calibration, cancellation
tests, and runtime coverage across tolerances and backends before stronger claims.
Stronger compiled failure-prefix and custom-controller
boundary regressions are also useful. Additional sampling cannot supply a
universal bound for unrestricted forcing; define the supported problem class and
controlled quantity before making stronger accuracy claims.
