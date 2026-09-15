# Recursive SOE History and Scaled-Tail Fitting

Status: Unreleased, provisional numerical approximation. Uniform grids and scalar
orders only. The full-history default is unchanged.

## Select the fit explicitly

```python
import jax.numpy as jnp
import hpfracc as hp

dt = 1 / 255
t = jnp.arange(256) * dt
result = hp.ops.caputo(
    t**2, dt=dt, order=0.5, history="soe", soe_poles=12,
    soe_fit="scaled_tail", return_info=True,
)
print(result.operator_info.to_dict())
```

The same `soe_fit` keyword is available for GL and RL. `"legacy"` remains the
default fit policy; `"scaled_tail"` is the new opt-in policy. Both now apply
exponential histories recursively rather than constructing a dense history
matrix. Legacy keeps its original fitting objective, pole range, regularization
and 400 fixed-step projected updates; the updates now use a bounded JAX loop.
Changes in floating-point accumulation are not a cross-version bitwise promise.

`soe_poles` must be a positive integer, and `soe_t_max` must be finite and
positive. The default horizon is the number of input samples times `dt`.
Dimensionless pole construction cancels this physical horizon mathematically,
so it is not an independent accuracy control. Legacy arithmetic at extreme
horizons can still be precision-sensitive. Vector orders and nonuniform SOE are
unsupported; invalid SOE configurations are also checked for singleton inputs.

## Recurrence and fitting targets

### Legacy fit

For fitted coefficients `c_j` and decays `rho_j`, the legacy approximation is:

```text
k_hat[k] = sum_j c_j * rho_j^k
s_j[n] = rho_j*s_j[n-1] + v[n], starting from zero history
output[n] = sum_j c_j*s_j[n].
```

`v` is the input for GL/RL and the signal increments for Caputo. This evaluates
the same fitted discrete convolution as the previous dense application up to
roundoff. It does not improve the approximation quality of the legacy fit.

### Scaled positive-tail fit

The new policy retains `k_hat[0]=1` exactly and approximates only the positive
tail:

```text
k_hat[k] = sum_j c_j*rho_j^(k-1), k >= 1, c_j >= 0
output[n] = v[n] + sum_j c_j*s_j[n-1].
```

For Caputo, the target is the L1 kernel
`b_k=(k+1)^(1-alpha)-k^(1-alpha)`. Positive tail values are evaluated using
`k^(1-alpha)*expm1((1-alpha)*log1p(1/k))`, avoiding subtraction of close powers.

For GL/RL, the target is the **cumulative** GL kernel:

```text
g_0 = 1
g_k = g_(k-1)*(k-alpha)/k
g_k = sum_{j=0}^k w_j
v_0 = x_0; v_n = x_n-x_(n-1), n >= 1
sum_{j=0}^n w_(n-j)*x_j = sum_{j=0}^n g_(n-j)*v_j.
```

This positive-kernel/increment identity avoids fitting a strongly cancelling
signed derivative kernel. The initial value is retained in `v_0`; RL of a
constant is therefore nonzero, whereas Caputo of a constant is zero. The usual
GL or Caputo timestep/gamma normalization is applied after the recurrence.

## Scaled-tail fitting algorithm

With `N` kernel samples and `P` poles, dimensionless decay rates are log-spaced
from `1/(16*N)` to `8`, and `rho=exp(-rate)`. Let
`A[k,j]=rho_j^k`, for `k=0,...,N-2`, and let `b` be the target tail.

The fitter avoids materializing `A`:

- `G=A.T*A` is a finite geometric sum, evaluated with `expm1` and the exact
  `N-1` limit when a rounded pole product equals one.
- `q=A.T*b` is accumulated over lags with bounded state.
- Column norms are `d_j=sqrt(G_jj)`; normalized Gram entries are
  `B_ij=G_ij/(d_i*d_j)` and normalized RHS entries are `q_i/d_i`.
- Starting from zero, 400 projected-gradient iterations use
  `z <- max(z-(B*z-q/d)/L, 0)`, with `L=max_i sum_j |B_ij|`; finally `c=z/d`.

In exact arithmetic the row-sum bound supplies a conservative Lipschitz step
for the convex least-squares objective. Fixed iterations do not certify an
optimal NNLS solution, a tolerance, or monotonically improving fits as poles are
added. The pole and Gram/normalization constants depend only on shape, policy and
dtype. They are evaluated with `jax.ensure_compile_time_eval` so eager and JIT
paths use consistent constants; order-dependent fitting remains differentiable.

## Complexity and limits

For `M` flattened state components:

| Stage | Forward work | Additional forward working state |
| --- | --- | --- |
| Legacy fit | `O(N*P^2 + 400*P^2)` | `O(N*P + P^2)` design/Gram storage |
| Scaled-tail fit | `O(N*P + 400*P^2)` | `O(P^2)` |
| Recursive application | `O(N*P*M)` | `O(P*M)` history |

Input/output remain `O(N*M)` arrays. This is still a batch API: it does not expose
a resumable streaming state, and the fit itself depends on the requested length.
Reverse-mode storage may retain intermediate fitting/history states and is not
covered by these forward bounds. Solver histories remain uncompressed.

Let `epsilon=max_k |k_hat[k]-k[k]|`. In exact arithmetic the unscaled discrete
convolution error obeys:

```text
|approx[n]-reference[n]| <= epsilon * sum_{j=0}^n |v[j]|.
```

Apply the operator normalization to obtain a derivative bound. For scaled-tail
GL/RL, `v` includes the initial value and increments, and `epsilon` concerns the
positive prefix kernel. This is a signal-dependent **absolute** bound, not a
universal relative bound, a cheap automatic error controller, or a bound on the
underlying discretization's error relative to the continuous derivative.

## Diagnostics and tests

`OperatorInfo.diagnostics` identifies the fit policy, recursive application,
target kernel (`l1`, `gl`, or `gl_prefix`), exact-head flag and fit sample count.
It reports actual streamed kernel max-absolute/relative-L2 residuals and a finite
fit flag. Nonfinite eager diagnostic values export as JSON null; numeric fields
can be extracted under JIT. Kernel residuals and `soe_fit_finite=True` do not
certify derivative accuracy. A singleton Caputo input reports empty history.

`tests/unit/test_soe_recursive.py` validates:

- recursive versus direct fitted sums, including legacy compatibility;
- streamed sufficient statistics versus an independent dense normalized fit;
- the original legacy projected-update rule;
- the signal-dependent residual bound, including the signed-GL/prefix identity;
- selected polynomial refinement and the decisive Caputo/RL constant distinction;
- input/order finite-difference gradients at orders 0.3, 0.6 and 0.85 for both
  policies, JIT/eager checks, complex signals, trailing shapes and outer vmap;
- bounded fitting graphs without a length-by-poles design/target array;
- singleton/invalid/unsupported configurations and JSON diagnostics.

The dedicated audit tests check independent references, failed-measurement
suppression and missing/duplicate matrix rejection. These are selected cases;
they do not establish full-history convergence order for fixed SOE fits.

## Accuracy audit

The [pre-execution protocol](audits/soe-2026-09-14/protocol.json) fixes two
operators, both fit policies, orders 0.3/0.5/0.9, sample counts
64/256/1024/4096, and 8/12 poles. Each of 96 configurations per precision acts on
four signals: constant, quadratic, `sin(8*pi*t)`, and a decaying transient switched
on at `t=0.25`, on `[0,1]`. Full discrete references use independent host float64
weights and convolution of the actual input samples; quadratic continuous
references are also recorded. Relative L2 is undefined (null) for the zero
Caputo constant reference, whose max absolute error is zero.

Both final checked accuracy sweeps exited 0: **192 complete finite configurations**.
This includes poor legacy fits. At 4,096 samples and 12 poles in float64, quadratic
relative L2 errors against the full discrete reference were:

| Operator | Order | Legacy | Scaled tail |
| --- | --- | --- | --- |
| Caputo | 0.3 | 1.000 | 5.680e-4 |
| Caputo | 0.5 | 0.9814 | 6.780e-5 |
| GL | 0.5 | 0.2222 | 6.785e-5 |
| GL | 0.9 | 6.956 | 7.268e-5 |

Legacy's L1 fixed step becomes unsuitable at large lengths; some fits return
zero weights despite being finite. GL normal equations and signed-coefficient
cancellation also show precision sensitivity. Recursion faithfully evaluating
such a fit cannot repair it. The new policy is explicitly selected rather than
silently changing the released fit.

Approximation floors remain. With 12 poles and order 0.5 in float64, Caputo's
quadratic max error against the continuous derivative decreases from about
`9.56e-4` at 64 samples to `5.29e-5` at 1,024, then rises to `9.82e-5` at 4,096.
More samples or poles with a fixed iteration budget need not improve error.
Whole-trajectory L2 errors are not pointwise relative bounds, particularly for
RL/GL constant responses with large early-time magnitudes. All signal max-absolute
errors, including unfavorable transient cases, are retained in the artifacts.

## Scaling and the static-constant follow-up

The initial float32 scaling check failed even for compiled dense application
against its eager fitted reference. A slow pole differed by a few float32 ulps,
perturbing the fit; compiled dense versus recursive outputs were already close.
The [recorded follow-up](audits/soe-2026-09-14/static-constants-followup.json)
fixes evaluation of order-independent constants, preserves the initial failed
run, and retains the original tolerances and reference. New long-sequence
eager/JIT regressions cover this behavior.

The final scaling matrix uses scalar-order Caputo, `scaled_tail`, lengths
256/1024/4096, dimensions 1/4 and poles 8/16. Both implementations include the
order-dependent fit and application with dynamic input/order arguments.
Compilation/warm-up are separate; seven seeded shuffled serial rounds measure
compiled execution through synchronization, excluding host checks/transfers.

Both precisions passed all checks: **48 configurations, 336 timed samples**,
with finite repeatable outputs agreeing with the eager dense-fitted reference.
Selected four-component, eight-pole measurements:

| Precision | Samples | Dense fitted median ms | Recursive median ms | Dense XLA temporary bytes | Recursive XLA temporary bytes |
| --- | --- | --- | --- | --- | --- |
| Float32 | 1,024 | 0.935 | 0.314 | 4,202,792 | 17,512 |
| Float32 | 4,096 | 5.642 | 0.531 | 67,141,928 | 66,664 |
| Float64 | 1,024 | 1.564 | 0.448 | 8,405,288 | 34,024 |
| Float64 | 4,096 | 15.142 | 0.744 | 134,283,560 | 132,328 |

These are local Windows CPU measurements with JAX/jaxlib 0.10.1 and Python
3.13.13. XLA figures are compiler buffer accounting, **not peak RSS/device
memory**. Static constants and generated-code storage are not a complete process
memory measurement. Some scalar dense cases are fused into linear temporary
storage by XLA. No universal speedup, accelerator ranking or reverse-mode memory
claim follows; small configurations can be dominated by dispatch and fitting.
CPU affinity/frequency/background load were not controlled.

## Reproduce and evidence

Use a new directory for a new experiment. Timing sweeps should run serially
without competing numerical workloads:

```powershell
$env:JAX_ENABLE_X64 = '0'
uv run --frozen python -m benchmarks.numerical.soe_audit --check --output-dir docs/validation/audits/soe-new
uv run --frozen python -m benchmarks.numerical.soe_audit --mode scaling --check --output-dir docs/validation/audits/soe-new
$env:JAX_ENABLE_X64 = '1'
uv run --frozen python -m benchmarks.numerical.soe_audit --dtype float64 --check --output-dir docs/validation/audits/soe-new
uv run --frozen python -m benchmarks.numerical.soe_audit --mode scaling --dtype float64 --check --output-dir docs/validation/audits/soe-new
```

Accuracy `--check` requires exact coverage and finite measurements, not small
approximation error. Scaling additionally checks repeat counts, reference
agreement and finite positive timings/buffer figures, not speed superiority.

Final consistent-constant evidence:

- [Float32 accuracy CSV](audits/soe-2026-09-14/consistent-poles/soe_accuracy_float32.csv)
  and [JSON/provenance](audits/soe-2026-09-14/consistent-poles/soe_accuracy_float32.json).
- [Float64 accuracy CSV](audits/soe-2026-09-14/consistent-poles/soe_accuracy_float64.csv)
  and [JSON/provenance](audits/soe-2026-09-14/consistent-poles/soe_accuracy_float64.json).
- [Float32 scaling CSV](audits/soe-2026-09-14/consistent-poles/soe_scaling_float32.csv)
  and [raw samples/provenance](audits/soe-2026-09-14/consistent-poles/soe_scaling_float32.json).
- [Float64 scaling CSV](audits/soe-2026-09-14/consistent-poles/soe_scaling_float64.csv)
  and [raw samples/provenance](audits/soe-2026-09-14/consistent-poles/soe_scaling_float64.json).

The root audit directory preserves the earlier accuracy runs and failed float32
scaling run before static-constant hardening. Their source hashes are historical.
Every bundle records source/CSV digests, configuration, dirty-state and runtime
provenance. These are development measurements, not release sign-off or closure
of the remaining Phase A error-control and broader scaling gates.

The legacy-fit docstring was subsequently clarified to describe an exponential
sum rather than implying that GL coefficients are nonnegative. That docstring-only
change affects the kernel file's byte digest, not the numerical implementation
or the preserved measurements.

## Final development checks

- `uv run python -m pytest`: **572 passed**, 1 performance test deselected,
  in 557.76 s; confirmed process exit 0.
- Dedicated SOE implementation/report tests: **64 passed**.
- `uv run python -m benchmarks.numerical.validation_summary --check`: all
  **20 baseline/hardening rows passed**, exit 0. SOE's complete matrices are
  checked by the separate commands above.
- Final accuracy and scaling checked commands: all four exited 0, covering
  **192 accuracy configurations** and **48 scaling configurations / 336 samples**.
- Strict documentation build, Ruff 0.16.7 on SOE files, and whitespace checks:
  passed. Existing repository-wide lint/tooling gaps remain separate work.

This work is local and uncommitted on top of `37aaac6` and the preceding native
convolution remediation. The next numerical priority is stronger nonlinear/stiff
implicit-solver validation. Remaining history-method work includes tolerance-driven
fit selection, broader approximation calibration, reverse-mode/accelerator peak
memory evidence, and future solver-history compression.
