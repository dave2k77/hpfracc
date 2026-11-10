# Local GPU normalization follow-up

This is dirty-tree development evidence on top of `6f5d3d3`, not release sign-off,
Phase A closure, or a general GPU/TPU error guarantee. Numerical source and test
SHA-256 digests, actual devices, package versions and runtime controls are in
`cpu-environment.json` and `gpu-environment.json`. Earlier audit bundles are
unchanged.

## Diagnosis and correction

On the RTX 5070 Laptop GPU, float32 general-purpose Gamma evaluation shifted the
uniform corrector scale for order `float32(0.7)` and `dt=0.5` by approximately
`-1.27e-6` relative to an independent host-float64 reference. Newton accurately
solved that perturbed discrete equation: the reported scaled residual was
`0.03296`, but the independent one was `1.79460`, above the fixed limit of one.

The uniform explicit/implicit setup now evaluates the reciprocal Gamma factor
using the supported-interval series described in
[uniform-weight validation](../../uniform-weights.md#working-precision-normalization-follow-up).
It derives the corrector scale from the predictor scale by recurrence. The
independent residual is now `0.0783186`; the reported residual is `0.0247206`.
The Newton iteration count, tolerances and independent pass threshold did not
change. No global x64 override or additional runtime dependency was added.

The coefficient comparison in `gpu-coefficients.json` records nine actual
float32 orders, the old Gamma formulas, the new scales and independent host
references. It measures relative scale errors, not trajectory error bounds.
Both checked aggregate reports contain 28 passing rows; see
`cpu-aggregate.csv` and `gpu-aggregate.csv`.

Full default-suite reruns passed on both backends: **713 passed**, one performance
test deselected, with confirmed process exit 0 in each environment. Pytest reported
642.89 seconds on CPU and 1205.95 seconds on GPU; concurrent local development
runs are not a performance comparison. Raw JUnit results are in `cpu-tests.xml`
and `gpu-tests.xml`; `verification.json` summarizes the checks. Ruff, strict docs,
dependency compatibility and whitespace checks passed. All four documented
fractional-ODE examples also completed with CUDA as the default backend.

The preceding operator fixes are included in this tree: highest-precision dense
history/SOE fitting products, independent discrete-reference tests, and explicit
CPU availability for debug callbacks. The latter requires `JAX_PLATFORMS=cuda,cpu`
rather than disabling the CPU backend entirely. CUDA remains the default device
and its initialization is required.

## Reproduce

Use the repository root, Python 3.13 and CI's uv 0.11.21. The local CPU environment
was synced with `--extra dev`; the separate GPU environment was synced with
`--extra dev --extra cuda13`. Both use the same project `uv.lock` and
unchanged JAX/jaxlib 0.10.1 versions.

```bash
JAX_PLATFORMS=cpu uvx --from 'uv==0.11.21' uv run --frozen --extra dev python -m pytest
JAX_PLATFORMS=cpu uvx --from 'uv==0.11.21' uv run --frozen --extra dev python -m benchmarks.numerical.validation_summary --check

export UV_PROJECT_ENVIRONMENT="$HOME/.local/share/hpfracc/venv-cuda13"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uvx --from 'uv==0.11.21' uv sync --locked --extra dev --extra cuda13 --python 3.13
uvx --from 'uv==0.11.21' uv run --frozen --extra dev --extra cuda13 python -m pytest
uvx --from 'uv==0.11.21' uv run --frozen --extra dev --extra cuda13 python -m benchmarks.numerical.validation_summary --check
```

Do not set global x64 for the full suite: tests enable it locally when needed and
some deliberately exercise float32 defaults. One performance test is deselected
by the normal project configuration. Hardware drivers are not installed or
pinned by the Python lockfile.
