# Dense Subdivision Runtime Audit

Status: Unreleased development evidence. This follow-up leaves subdivision
experimental and opt-in; no solver numerics or defaults are changed.

## Predeclared question

The [coarse runtime audit](adaptive-runtime.md) lacked near-runtime pairs for
long decay in either precision and nonlinear feedback in float32. This follow-up
fills in the **same tolerance range**, instead of changing the matching threshold
or choosing tolerances according to favorable accuracy.

The [local protocol](audits/subdivision-runtime-dense/protocol.json) was written
before executing the dense sweep, with source hashes and analysis/stopping rules.
It is informed by earlier coarse measurements, not an independently registered
confirmatory study. Its coverage goal is at least one eligible pair for every
case in each precision. Missing coverage must be reported rather than repaired
by changing the grid after seeing this sweep.

## Fixed design

The 13 tolerances, identical for both modes, every case, and both precisions, are:

```text
1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1.5e-3, 1e-3,
7e-4, 5e-4, 3e-4, 2e-4, 1.5e-4, 1e-4
```

All five coarse tolerances are included. The three half-order problems,
continuous references, `y(0)=1`, `first_step=0.01`, `atol=0.01*rtol`, and
2,048-attempt budget are unchanged. There are 78 configurations per precision,
156 overall, with seven timed repetitions each: 1,092 planned samples.

The timing protocol is unchanged: synchronize dynamic inputs, lower/compile,
perform one synchronized warm-up per configuration, then execute seeded shuffled
serial rounds. Time compiled dispatch through whole-output synchronization;
keep host validation and continuous-reference evaluation outside timed regions.
Float32 and float64 run sequentially on WSL CPU. No tests or other numerical
benchmarks run concurrently with timing.

Runtime eligibility still requires compatible completed solves with stable,
finite positive timings and
`max(median_seconds)/min(median_seconds) <= 1.2`. Export **every** eligible pair,
without selecting by error, interpolation, or extrapolation. Maximum errors use
each solver's accepted nodes; endpoint errors share the physical endpoint.

The stopping rule is one full sweep per precision, retaining every failure and
without increasing the attempt budget or adjusting tolerances/thresholds.
Pairs can reuse configurations and are **not independent trials**: pair counts
must not be interpreted as win probabilities or statistical significance.

## Reproduce

Use a new output directory to preserve prior evidence. Run these sequentially:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 uv run --frozen python -m benchmarks.numerical.subdivision_runtime --sweep dense --dtype float32 --max-steps 2048 --repeats 7 --seed 1729 --check --output-dir docs/validation/audits/subdivision-runtime-dense
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --frozen python -m benchmarks.numerical.subdivision_runtime --sweep dense --dtype float64 --max-steps 2048 --repeats 7 --seed 1729 --check --output-dir docs/validation/audits/subdivision-runtime-dense
```

`--sweep dense` selects only the tolerance grid. Seven repetitions and CPU
execution must still be requested explicitly as above; the CLI defaults remain
`--sweep coarse` and five repetitions. Source hashes and the final artifacts must
be compared with the local declaration externally; the runner does not enforce
a protocol file itself.

`--check` requires exact matrix coverage, positive finite times, the requested
sample count, repeatable outputs, and successful endpoint completion for every
solve. It does **not** check runtime-pair coverage or accuracy superiority.
Therefore matrix integrity, timing integrity, solver completion, and pair
coverage are reported separately. A budget-exhausted trajectory retains timing
and failure diagnostics but has no full-span accuracy values.

## Recorded results

The sweep followed the fixed protocol without modifying the grid, matching
threshold, or budgets. Both precision processes finished and wrote their
artifacts; each actual checked CLI exited with status 1 because of retained
budget failures, not missing configurations or invalid timings.

| Gate / measurement | Float32 | Float64 |
| --- | --- | --- |
| Unique configurations, expected / collected | 78 / 78 | 78 / 78 |
| Timed samples, expected / collected | 546 / 546 | 546 / 546 |
| Completed solves | 71 / 78 | 71 / 78 |
| Stable elementwise repeated outputs | 78 / 78 | 78 / 78 |
| Eligible approximately runtime-matched pairs | 33 | 33 |
| Coverage of all three cases | Yes | Yes |
| Frozen configuration and source hashes match | Yes | Yes |

All 156 configurations and 1,092 samples are retained. All default solves
completed. In both precisions, subdivision exhausted the budget for strong
feedback at `rtol=2e-4, 1.5e-4, 1e-4` and long decay at
`rtol=3e-4, 2e-4, 1.5e-4, 1e-4`. These seven failures per precision have blank
full-span error columns and are excluded from accuracy/runtime pairs.

### Every case now has near-runtime coverage

The following table summarizes **all** eligible pairs. Lower/higher refers to
subdivision's error relative to the default mode; no exact error ties occurred.

| Precision | Case | Pairs | Lower max error | Higher max error | Lower endpoint error | Higher endpoint error |
| --- | --- | --- | --- | --- | --- | --- |
| Float32 | Nonlinear | 8 | 0 | 8 | 4 | 4 |
| Float32 | Strong feedback | 16 | 2 | 14 | 4 | 12 |
| Float32 | Long decay | 9 | 0 | 9 | 4 | 5 |
| Float64 | Nonlinear | 9 | 0 | 9 | 5 | 4 |
| Float64 | Strong feedback | 16 | 3 | 13 | 4 | 12 |
| Float64 | Long decay | 8 | 0 | 8 | 4 | 4 |

Subdivision has a larger maximum accepted-node error in every matched nonlinear
and long-decay pair. For long decay, subdivision/default max-error ratios range
from approximately 1.708 to 4.269 across the two precisions.

The stronger-feedback result is mixed: subdivision has lower maximum error in
five of the 66 overall eligible pairs. This extends the coarse audit rather than
supporting an extrapolation of its seven unfavorable comparisons to every case.
Endpoint errors favor subdivision in 25 pairs and the default in 41 pairs.
Different maximum-error and endpoint rankings are possible on the same pair.

These counts are descriptive, not independent win trials. They do not establish
universal dominance, calibrated error control, or promotion of subdivision.
The practical conclusion remains to keep the default unchanged and the extra
subdivision probes explicitly experimental.

### Variability limits the interpretation

Within-configuration maximum/minimum sample-time ratios range from 1.056 to 1.513
in float32 and 1.041 to 1.925 in float64. Seven repetitions with this variability
do not establish tight uncertainty bounds on median runtime or pair membership.
Near-threshold pairs could move in or out of the 20% band in another experiment.

CPU affinity, host background activity, temperature, clock frequency, and power
mode were not controlled. This is local WSL CPU evidence using the locked
JAX/jaxlib 0.10.1 environment, not a cross-hardware or GPU result. The coverage
goal was met in this recorded sweep, but coverage alone does not close the
scientific calibration, stability, or release-readiness gates.

## Artifacts and verification

- [Predeclared protocol](audits/subdivision-runtime-dense/protocol.json).
- [Float32 CSV](audits/subdivision-runtime-dense/subdivision_runtime_float32.csv)
  and [raw samples / provenance / pairs](audits/subdivision-runtime-dense/subdivision_runtime_float32.json).
- [Float64 CSV](audits/subdivision-runtime-dense/subdivision_runtime_float64.csv)
  and [raw samples / provenance / pairs](audits/subdivision-runtime-dense/subdivision_runtime_float64.json).
- [All 66 eligible pairs as CSV](audits/subdivision-runtime-dense/matched_runtime_pairs.csv),
  sorted without selecting by accuracy.
- [External protocol and artifact verification](audits/subdivision-runtime-dense/verification.json),
  binding the protocol digest to the exact JSON/CSV artifacts and recording
  separate integrity, timing, completion, and coverage results.

External verification checks every expected unique run key, sample count,
positive finite timing, repeated-output flag, CSV/JSON measurement agreement,
failed-run error suppression, all recomputed runtime/work pairs, recorded
configuration, and frozen source hashes. Earlier coarse audit bundles are
unchanged. The new code only adds a named sweep and its reporting/coverage tests;
the solver, timing implementation, and continuous-reference definitions are
unchanged.
