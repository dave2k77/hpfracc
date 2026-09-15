# Native Convolution Failure and Windowed Scan Follow-up

Status: Unreleased remediation on top of commit `37aaac6`. Local Windows CPU
evidence; the underlying native-library defect has not been diagnosed internally.

The subsequent [recursive SOE work](soe-history.md) extends operator history
compression beyond this checkpoint. This page preserves the native-convolution
investigation and its measurements.

## Reduction to pure JAX

The [initial windowed benchmark](windowed-history.md#float64-runtime-blocker)
failed with Windows access violation `0xC0000005`. Instrumenting the mixed-shape
run localized failures to batched float64 convolution. Further reduction removed
HPFRACC, fractional weights, gamma normalization, and adaptive/deterministic
solvers entirely.

`benchmarks.numerical.jax_convolution_probe` imports only standard-library tools,
JAX and NumPy. Its child constructs two four-component signals, lengths 256 and
4,096, with constant kernel weights and windows 8 and 32. It retains a small
compiled dense product and a large compiled batched `jnp.convolve`, after eager
dense reference allocations. It warms both and repeats a seeded shuffled pair
sequence. The alternative control evaluates the identical windowed sum with a
scan of shifted signals.

The parent launches a fixed five fresh processes and preserves **all** outputs,
tracebacks and exit codes. Failures can occur during warm-up or subsequent
execution. Smaller probes and some fresh processes pass: reproduction is
allocation/process-sensitive, not a deterministic failure on every call.

| Five-process diagnostic | Completed | Native access violations | Parent exit |
| --- | --- | --- | --- |
| Batched convolution | 3 | 2 | 1 |
| Shift-and-accumulate scan | 5 | 0 | 0 |

These counts describe this local diagnostic, not a statistical reliability
estimate. They establish a failure path outside HPFRACC's fractional numerics;
they do not identify a particular allocator, race, instruction, or upstream
source-code defect. No JAX dependency version or global runtime flags were
changed in the library.

## Numerical remediation

Short-memory now evaluates the same retained discrete sum using `jax.lax.scan`
over lags. Each iteration gathers the causally shifted input, masks the
prehistory to zero, multiplies by the scalar or per-feature weight, and adds to
the output. This keeps `O(N*W*M)` work, retained-window-only weights, and
full-size input/output arrays without a dense history matrix.

The scan is used on every backend rather than a platform-specific fallback. It
also avoids reintroducing batched convolution through an outer `jax.vmap`.
Accumulation order may change floating-point roundoff; the approximation,
window clamping, operator metadata and full-history default are unchanged.
Reverse-mode workspace remains backend/JAX-dependent and is not measured here.

Existing independent truncated sums, discarded-tail bounds, exact cutoff,
complex-signal and input/order gradient tests exercise the replacement. A new
outer-vmap regression explicitly forbids calling `jnp.convolve`, compares to the
dense reference, and checks mixed-precision output promotion. The diagnostic
parent's tests verify that a native child failure does not discard successes
or stop collection of the remaining preselected processes.

## Completed benchmark evidence

The original seven-round, seeded, shuffled benchmark now completes in both
precisions. Each precision contains 24 configurations and 168 synchronized
samples; all measurements agree with the dense-truncated reference and all
repeated outputs are equal. These are new bundles, not replacements for the
initial native-convolution artifacts.

At 4,096 samples and four components:

| Precision | Window | Dense median ms | Scan median ms | Dense XLA temporary bytes | Scan XLA temporary bytes |
| --- | --- | --- | --- | --- | --- |
| Float32 | 8 | 41.016 | 0.182 | 67,141,776 | 82,352 |
| Float32 | 32 | 39.587 | 0.298 | 67,141,776 | 82,480 |
| Float64 | 8 | 52.687 | 0.193 | 134,283,408 | 164,272 |
| Float64 | 32 | 54.381 | 0.365 | 134,283,408 | 164,656 |

The unchanged protocol measures compiled dispatch plus output synchronization,
with host accuracy checks outside timed intervals. Compilation and warm-up are
separate fields. Memory numbers are XLA compiler buffer accounting, not sampled
peak RSS/device memory. The benchmark uses scalar-order Caputo forward paths;
the results do not establish reverse-mode, per-state-order, or accelerator
performance. Host load/frequency/affinity were not controlled, and the two
precision sweeps ran separately. Do not infer cross-precision or cross-run
speed rankings from small timing differences.

The recorded native path is avoided and both full sweeps now pass. This is a
validated local remediation, not proof that all native JAX failures are fixed.

## Final verification

- `uv run python -m pytest`: **508 passed**, 1 performance test deselected,
  in 674.84 s; confirmed process exit 0.
- `uv run python -m benchmarks.numerical.validation_summary --check`: all
  **20 rows passed**, exit 0.
- Both seven-repeat windowed benchmark commands with `--check`: exit 0,
  **48 configurations and 336 timed samples** overall.
- Strict documentation build, Ruff 0.16.7 on the four changed/added Python
  files, and `git diff --check`: passed. Existing repository-wide lint/tooling
  gaps remain as recorded at the earlier checkpoint.

The first development cycle was committed as `37aaac6`; this follow-up remains
local uncommitted work. Recursive SOE execution and its approximation/gradient
validation are the next numerical workstream. The pure-JAX reproducer remains
available for upstream investigation of the native-library failure.

## Reproduce

Use a new artifact directory for a new experiment. The convolution diagnostic
can return 1 when one or more child processes fail; its JSON still contains
every outcome. The parent forces x64 CPU execution only in its child environment.

```bash
uv run python -m benchmarks.numerical.jax_convolution_probe --kernel convolution --output docs/validation/audits/native-new/convolution.json
uv run python -m benchmarks.numerical.jax_convolution_probe --kernel scan --output docs/validation/audits/native-new/scan.json
```

Run the numerical timing sweeps serially without other numerical workloads:

```powershell
$env:JAX_ENABLE_X64 = '0'
uv run --frozen python -m benchmarks.numerical.windowed_history --check --output-dir docs/validation/audits/windowed-scan-new
$env:JAX_ENABLE_X64 = '1'
uv run --frozen python -m benchmarks.numerical.windowed_history --dtype float64 --check --output-dir docs/validation/audits/windowed-scan-new
```

## Artifacts

- [All convolution child outcomes](audits/native-convolution-2026-09-14/convolution.json).
- [All scan child outcomes](audits/native-convolution-2026-09-14/scan.json).
- [Float32 benchmark CSV](audits/windowed-scan-2026-09-14/windowed_history_float32.csv)
  and [raw samples/provenance](audits/windowed-scan-2026-09-14/windowed_history_float32.json).
- [Float64 benchmark CSV](audits/windowed-scan-2026-09-14/windowed_history_float64.csv)
  and [raw samples/provenance](audits/windowed-scan-2026-09-14/windowed_history_float64.json).

The records identify Windows 11, Python 3.13.13, JAX/jaxlib 0.10.1 and CPU
execution. Diagnostic JSON includes a digest of its self-contained source;
benchmark JSON also records the package revision, dirty state, numerical source
digests, CSV digest and complete configuration.
