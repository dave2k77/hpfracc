# Synchronized Subdivision Runtime Audit

Status: Unreleased development evidence. The subdivision estimator remains
experimental and opt-in. No solver numerics or defaults were changed.

## Question and scope

Does the [stress audit's padded-work proxy](adaptive-subdivision.md#nonlinear-and-long-span-stress-follow-up)
predict actual execution time, and do its accuracy comparisons persist at
similar measured runtimes?

The answer is qualified: the proxy is **not proportional to runtime across the
two modes**. Subdivision is faster at the previously paired proxy-work levels,
but all eligible approximately runtime-matched pairs in this experiment still
have larger maximum accepted-node errors with subdivision. Endpoint accuracy is
not uniformly worse, and the tolerance grid leaves gaps in runtime coverage.
This is not a general efficiency ranking, dense-output bound, or release gate.

The subsequent [predeclared dense sweep](adaptive-runtime-dense.md) extends the
same tolerance range to address these coverage gaps. This page and its artifacts
retain the original coarse experiment; its source fingerprints are historical.

## Reproduce

Run these sequentially, without tests or other numerical workloads in parallel:

```bash
JAX_PLATFORMS=cpu uv run --frozen python -m benchmarks.numerical.subdivision_runtime --repeats 7 --output-dir docs/validation/audits/subdivision-runtime
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --frozen python -m benchmarks.numerical.subdivision_runtime --dtype float64 --repeats 7 --output-dir docs/validation/audits/subdivision-runtime
```

Use a new output directory for a new experiment rather than overwriting archived
evidence. The default attempt budget is 2,048; the default repeat count is five,
explicitly overridden to seven for this record. `--seed` defaults to 1729.

`--check` additionally requires the exact complete case/mode/tolerance matrix,
finite positive compilation/warm-up/sample times, the requested sample count,
repeatable outputs, and every solve to reach its endpoint. It does **not** require
speed improvement, global error below tolerance, or runtime-pair coverage.
The recorded incomplete solves make this strict completion gate fail. Missing
runtime pairs are reported as missing evidence, not an interpolation opportunity.

## Method

The problems and continuous references are identical to the stress audit:
scalar order 0.5, `y(0)=1`, nonlinear manufactured feedback strengths 1 and 10 on
`[0,1]`, and half-order decay on `[0,16]`. The tolerance grid is
`1e-2, 3e-3, 1e-3, 3e-4, 1e-4`, with `atol=0.01*rtol` and `first_step=0.01`.

For each configuration:

1. Construct device-resident initial state and bounds, and synchronize them.
2. Lower and compile an outer JIT wrapper with **dynamic** initial state and
   bounds. This prevents precomputing a closed-over constant trajectory.
3. Record lowering-plus-compilation time separately. It is not a cold-cache
   compilation guarantee and excludes the first execution.
4. Perform one synchronized warm-up execution and record it separately.
5. After every configuration is prepared, run seven serial rounds, shuffling
   configuration order with a local seeded RNG in each round.
6. Measure `perf_counter` from compiled dispatch through `block_until_ready` on
   the **entire returned tree**: trajectory, times, and numerical diagnostics.
   No host transfer, continuous-reference calculation, or reporting is timed.
7. Outside each timed interval, compare all returned leaves to the warm-up output.
   This is exact **elementwise equality**, with equal NaNs accepted; it is not a
   raw-byte, signed-zero, NaN-payload, or cross-platform bitwise guarantee.
8. Validate the actual compiled outputs using the same measurement helper as
   the eager stress report: dtype/shape, finite accepted prefix, initial value,
   endpoint, mesh monotonicity, counts, mask, and explicit failure flags.

The measurement is steady-state **compiled numerical solve plus host dispatch
and synchronization**, not the cost of the eager Python `simulate` interface,
compilation amortization, data transfer, or application-level orchestration.

## Results

Both sweeps ran serially on WSL CPU using the locked JAX/jaxlib 0.10.1 environment.
There are 60 configurations and 420 timed execution samples. Every configuration
had elementwise-repeatable outputs across its seven samples.

In each precision, 27 of 30 solves completed. The same three subdivision cases
as in the eager stress audit exhausted their budgets: strong feedback at `1e-4`,
and long decay at `3e-4` and `1e-4`. All default-mode solves completed.
Failure timings remain in the data, but incomplete solves have no full-span
accuracy values and are excluded from accuracy/runtime pairing.

### Proxy work is not runtime

For the six proxy-work pairs per precision, subdivision/default median runtime
ratios were:

| Case | Default rtol | Subdivision rtol | Float32 runtime ratio | Float64 runtime ratio |
| --- | --- | --- | --- | --- |
| Nonlinear | 3e-4 | 1e-2 | 0.613 | 0.608 |
| Nonlinear | 1e-4 | 3e-3 | 0.712 | 0.679 |
| Strong feedback | 1e-4 | 1e-2 | 0.695 | 0.655 |
| Strong feedback | 1e-4 | 3e-3 | 0.667 | 0.635 |
| Long decay | 3e-4 | 1e-2 | 0.661 | 0.647 |
| Long decay | 1e-4 | 3e-3 | 0.705 | 0.682 |

Thus the earlier roughly equal proxy-work comparisons were not equal-runtime
comparisons. Padded weight-builder slots do not account for all compiler,
controller, allocation, or backend costs. These measurements establish the
mismatch but do not isolate its cause.

### Approximately matched runtime

Export every compatible completed pair satisfying
`max(median_seconds)/min(median_seconds) <= 1.2`, without selecting by error.
No interpolation or extrapolation is used. The complete eligible pair list is:

| Precision | Case | Default rtol | Subdivision rtol | Default ms | Subdivision ms | Max-error ratio | Endpoint-error ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Float32 | Strong feedback | 3e-4 | 1e-2 | 300.87 | 346.44 | 7.429 | 11.207 |
| Float32 | Strong feedback | 3e-4 | 3e-3 | 300.87 | 332.50 | 2.346 | 2.040 |
| Float32 | Strong feedback | 1e-4 | 1e-3 | 498.55 | 569.67 | 3.055 | 3.192 |
| Float64 | Nonlinear | 1e-3 | 1e-2 | 25.88 | 28.99 | 1.152 | 0.029 |
| Float64 | Strong feedback | 3e-4 | 1e-2 | 377.96 | 417.35 | 7.507 | 8.864 |
| Float64 | Strong feedback | 3e-4 | 3e-3 | 377.96 | 404.52 | 2.339 | 1.250 |
| Float64 | Strong feedback | 1e-4 | 1e-3 | 637.09 | 671.86 | 7.234 | 11.382 |

Error ratios are subdivision/default. Maximum errors are evaluated on each
method's own accepted mesh; endpoint errors share the physical endpoint.
Subdivision is slightly slower and has a larger maximum accepted-node error in
all seven pairs, but has a smaller endpoint error in the float64 nonlinear pair.

There is **no eligible long-decay runtime pair** in either precision and no
float32 nonlinear pair at the stated threshold. The coarse tolerance grid and
timing variability limit coverage; the missing cases do not establish runtime
superiority or inferiority. A denser, predeclared tolerance sweep is needed to
cover them without fitting tolerances to favorable errors.

### Variability and limitations

Within-configuration maximum/minimum sample-time ratios ranged from 1.033 to
1.247 in float32 and 1.033 to 1.166 in float64. Raw samples are retained; seven
repetitions on one machine are not a confidence interval or significance test.
Near-threshold pair membership may change in another run.

CPU affinity, host background activity, thermal state, frequency, and power mode
were not controlled. Float32 and float64 were run in separate sequential
processes, not interleaved precision trials. No GPU measurement, peak-memory
measurement, cross-hardware ranking, or general stiff-solver claim is made.
The timing data do not justify promoting subdivision or changing the default.

## Evidence and verification

- [Float32 CSV](audits/subdivision-runtime/subdivision_runtime_float32.csv) and
  [raw samples, pairs, provenance](audits/subdivision-runtime/subdivision_runtime_float32.json).
- [Float64 CSV](audits/subdivision-runtime/subdivision_runtime_float64.csv) and
  [raw samples, pairs, provenance](audits/subdivision-runtime/subdivision_runtime_float64.json).
- `tests/unit/test_subdivision_runtime.py`: whole-output synchronization order,
  dynamic compiled inputs, repeatability, incomplete-run accuracy suppression,
  pairing exclusions, exact sweep coverage, repeat counts, CLI argument checks,
  precision requirements, and CSV/JSON sample/hash round trips.

Source fingerprints and CSV digests were verified against the actual files.
Earlier stress bundles remain unchanged. The runtime follow-up extracts a shared
result-measurement helper but does not change its numerical definitions.

The experiment shell wrapper failed while reporting individual exit codes after
both sweeps had written their artifacts. The saved measurements were independently
validated and replayed through the strict report gate; this is distinguished from
claiming the wrapper succeeded. Numerical timing samples were not synthesized or
rerun to obtain more favorable comparisons.
