# FFT order-gradient precision correction

Status: Scoped local development evidence; not Phase A or release sign-off.

Independent pre-commit review found that the first FFT order-gradient test
matrix enabled x64 for its nominal float32 cases. Default-dtype weights/padding
promoted forward computations and objectives to float64 even though input orders
and their cotangents were float32. The earlier green matrix did not establish
float32 execution coverage. This is tracked as VAL-004 in the
[bug-resolution tracker](../../../developer/bug-resolution-tracker.md).

## Red and green evidence

- `cpu-red.xml`: after adding effective-dtype assertions, 12 float32 cases failed
  and 12 float64 cases passed. This is historical failure evidence; the test
  implementation was subsequently corrected.
- `cpu-final-green.xml`: all 24 corrected cases passed, zero failures/errors/skips.
- `gpu-final-green.xml`: all 24 corrected cases passed, zero failures/errors/skips.

The corrected fixture scopes x64 off for float32 and on for float64, outside JIT,
then restores the caller's previous setting. Eager/JIT FFT/full forward outputs,
objectives and gradients must have the declared dtype. Finite-difference steps
and numerical tolerances were not loosened; no production numerical code changed
for this correction.

The matrix covers Caputo, GL and RL operators at three selected scalar orders and
one per-state order vector, both precisions, 33 samples and two state components.
It checks finite differences, eager/JIT derivatives and full-history derivative
agreement. It does not establish arbitrary broadcasting/length coverage, an
order-gradient error theorem, or all of Phase A.

The green-run numerical source is represented by the companion environment
manifests. They were collected after the test correction; JUnit files identify
the actual run timestamps. The red artifact records the intentionally failing
assertion but does not contain a complete source snapshot. Do not relabel the
separate historical 713-test normalization bundle as including this test matrix.

## Reproduce

From the repository root with the locked development environment:

```bash
JAX_PLATFORMS=cpu uv run --frozen --extra dev python -m pytest tests/unit/test_fft_order_gradients.py -q
UV_PROJECT_ENVIRONMENT="$HOME/.local/share/hpfracc/venv-cuda13" JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --frozen --extra dev --extra cuda13 python -m pytest tests/unit/test_fft_order_gradients.py -q
```

The suite's own dtype fixture controls x64. Global input labels or cotangent
dtypes alone are insufficient evidence of the effective computation precision.
