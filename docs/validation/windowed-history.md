# Direct Windowed History Execution

Status: Unreleased implementation and local CPU evidence. `short_memory` remains
an explicit approximation; full history remains the default.

Implementation follow-up: the native batched-convolution call was replaced by
an equivalent scan of shifted, weighted signals after a pure-JAX crash
reduction. Both precision sweeps now complete; see the
[native-convolution investigation and scan evidence](native-convolution.md).
The original timing table and failure record below remain historical evidence
for the first implementation.

## Same truncation, genuinely windowed application

The previous implementation zeroed weights outside the configured window and
then applied a dense history matrix. The new implementation computes only the
retained weights and evaluates direct finite-kernel convolution per feature
with a shift-and-accumulate scan. It
supports uniform Caputo, GL, and RL with scalar or broadcastable per-state orders,
and real or complex signals. The API and metadata fields are unchanged.

For GL/RL at sample `n`, the sum is:

```text
dt^(-alpha) * sum_{k=0}^{min(n,W-1)} w_k*x[n-k].
```

For Caputo it is the same window applied to L1 increments, with the usual
`1/(Gamma(2-alpha)*dt^alpha)` normalization and initial output zero. Existing
window conversion/clamping is preserved: integer conversion, then clamping to
at least one and at most the available lag count. A window covering the whole
signal reproduces the full discrete method within floating-point tolerance.

For `N` samples and `M` features, direct convolution requires `O(N*W*M)` work
without constructing an `N*N` history matrix. Weights require `O(W)` storage for
a scalar order or `O(W*M)` for per-state orders. Input/output still require
`O(N*M)` storage: this batch API does not supply a streaming state or remove the
trajectory from memory. Actual backend workspace and reverse-mode allocations
remain implementation-dependent.

## Explicit approximation boundary

The discarded discrete contribution has a signal-dependent absolute bound:

```text
|full[n] - short[n]| <= |normalization| *
                       sum_{k=W}^{n} |weight[k] * signal[n-k]|.
```

Here `signal` is the input for GL/RL and the increments for Caputo, with the
corresponding lag indexing. This bound requires the discarded samples; it is a
validation/reference bound, not a cheap runtime error estimate. It says nothing
about discretization error relative to the continuous derivative and is not a
universal relative-error bound. Cancellation can make relative errors large.

For example, Caputo applied to `x(t)=t` on `t_n=n*dt` returns exactly
`(min(n,W)*dt)^(1-alpha)/Gamma(2-alpha)` up to roundoff. It saturates after the
window fills while the full Caputo derivative continues to grow. Refining `dt`
with fixed `W` shrinks the physical history horizon; it does not generally
converge to full history.

`tests/unit/test_windowed_history.py` checks independent host sums for oscillatory,
transient, and complex signals, the discarded-tail bound, scalar/per-state orders,
the exact linear cutoff and constant Caputo cases, singleton/window edges,
trailing shapes, JIT, input/order finite-difference gradients, and guards against
returning to dense application or building unneeded weights. Existing FFT,
vector-order, and compression tests also pass.

## Local forward-execution measurements

The benchmark compares the new operator against the **old dense-truncated
algorithm**, not against a different approximation. Scalar order is a dynamic
JIT argument. The fixed design uses Caputo order 0.5, samples `256,1024,4096`,
state dimensions `1,4`, windows `8,32`, and seven serial timing rounds with seeded
shuffled configuration order. Lowering/compilation and warm-up are recorded
separately. Timings include compiled dispatch and output synchronization but
exclude host reference/error checks and transfers.

All 24 float32 rows passed numerical agreement and exact repeated-output checks.
Selected four-component, window-8 results on Windows CPU/JAX 0.10.1:

| Samples | Dense median ms | Windowed median ms | Dense XLA temporary bytes | Windowed XLA temporary bytes |
| --- | --- | --- | --- | --- |
| 256 | 0.357 | 0.169 | 264,336 | 4,368 |
| 1,024 | 2.563 | 0.225 | 4,202,640 | 16,656 |
| 4,096 | 34.423 | 0.277 | 67,141,776 | 65,808 |

For four components the reported dense temporaries grow quadratically with
sequence length, while windowed temporaries grow approximately linearly for
fixed window. **These memory figures are XLA compiler buffer accounting, not
sampled peak process RSS or device memory.** For the scalar float32 cases XLA
already fuses the dense reference into linear temporary storage, despite its
quadratic work. A claim that every dense case actually allocates an `N*N` buffer
at runtime would therefore be incorrect.

The full matrix includes both window widths and dimensions. It covers forward
Caputo execution with scalar order, not per-state-order performance, reverse-mode
memory, GPU behavior, or a universal speedup. Host background activity and CPU
frequency/affinity were not controlled; seven local samples are not confidence
intervals. SOE used dense application at this checkpoint; the later
[recursive SOE extension](soe-history.md) has separate evidence. Solver history
is unchanged.

### Float64 runtime blocker

The float64 shuffled seven-repeat sweep terminated with native Windows access
violation `0xC0000005` (exit `-1073741819`) during `jax.block_until_ready` in the
sampling loop. It produced no completed CSV/JSON bundle. The failure reproduced
with Python faulthandler enabled. Subsequent isolated probes of all 12
size/dimension/window configurations, each with both implementations and one
repeat, passed numerical agreement and buffer accounting; a one-round shuffled
diagnostic sweep also completed. The cause has not been isolated.

A further 4,096-sample, four-component, window-32 probe executed each compiled
implementation 100 times in float64. Both the windowed operator and the dense
reference produced equal repeated outputs with successful process exit. The
failure has not been reduced to an individual configuration or implementation.

These probes are not a replacement float64 timing dataset. Float64 mathematical,
gradient, and compatibility tests passed, but the repeated-run native failure
remains a backend/benchmark investigation item. No float64 performance claim or
cross-precision timing comparison is made.

## Reproduce and artifacts

Run without concurrent numerical workloads, using a new output directory:

```bash
uv run --frozen python -m benchmarks.numerical.windowed_history --check --output-dir docs/validation/audits/windowed-new
```

The default dtype is float32. `--dtype float64` requires x64 enabled; the current
scan implementation completes the previously blocked sweep. `--check` requires all 24 rows,
agreement, repeatability, positive finite timing samples, and available compiler
buffer figures; it does not impose a speedup or a peak-memory bound.

- [Float32 CSV](audits/windowed-2026-09-14/windowed_history_float32.csv).
- [Raw samples, configuration, source digests, and provenance](audits/windowed-2026-09-14/windowed_history_float32.json).

This record is dirty-tree development evidence, not a release sign-off.
