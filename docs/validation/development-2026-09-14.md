# Development checkpoint: 2026-09-14

This is local development evidence, not a release sign-off. The starting tree
was clean on `main`, one commit ahead of the local `origin/main` tracking ref:
`c6c40784b90d244b3ec4ad49178c8bc323a2b274`.

## Starting baseline

Environment recorded at `2026-09-14T15:09:57.676499+00:00`: Windows 11
(`Windows-11-10.0.26200-SP0`), Python 3.13.13, JAX 0.10.1, CPU device `cpu:0`,
package metadata `0.1.0a1`. JAX x64 was disabled by default; scientific tests
enable it locally where needed. Commands used the repository's uv environment.

| Command | Observed result |
| --- | --- |
| `uv run python -m pytest` | Pytest reported 465 passed, 1 performance test deselected, in 555.74 s. The shell tool timed out at 600 s after the summary; successful process exit was not confirmed. |
| `uv run python -m benchmarks.numerical.validation_summary --check` | Exit 0; all 20 reported rows passed. |
| `uv run mkdocs build --strict` | Exit 0; documentation built. |
| `uv run ruff check .` | Failed to spawn: Ruff is absent from the project environment. |
| `uv run --with ruff ruff check . --statistics` | Temporary Ruff 0.16.7: 25 findings (16 E402, 6 E501, 2 F401, 1 UP035). |

The E402 findings concern test import/precision setup. F401 concerns private
kernel re-exports in `ops/__init__.py`; UP035 concerns `typing.py`. E501 affects
probabilistic calibration and four tests. This inventory does not close the lint
or lockfile/CI alignment gates.

## Adaptive evidence decision

The existing scalar evidence includes nonlinear/strong-feedback/long-span stress
tests, source-level work accounting, synchronized runtime comparisons, and a
predeclared dense tolerance sweep. See [dense runtime evidence](adaptive-runtime-dense.md).
Repeated scalar runtime sweeps are not the next missing scientific check.

The next bounded work item is coupled nonlinear manufactured solutions with
different component scales and fractional orders, plus compiled retained-prefix
failure checks. The following measurement design is fixed before running it:

- Two component-scale cases: `(1, 1)` and `(1e-3, 1e3)`.
- Scalar orders `0.25, 0.5, 0.8`; bounds `[0, 1]`; feedback strength `1`.
- Both existing estimator modes; `rtol=1e-2, 1e-3`, `atol=0.01*rtol`.
- `first_step=0.01`, attempt budget `1024`; float32 and float64 separately.
- Exact dimensionless solution `(1+t^2, 0.5+t^3)` with a coupled nonlinear
  restoring term that vanishes on that solution. Check the forcing against an
  independent continuous Caputo integral.
- Record componentwise absolute, dimensionless, and reference-tolerance-scaled
  errors at accepted nodes and the endpoint. Retain failures with no full-span
  accuracy values. These are not common-grid or equal-work comparisons.
- Check frozen-mesh initial-state, parameter, and order sensitivities on a small
  successful coupled case, and JIT-readable failure prefixes separately.

Keep subdivision experimental regardless of whether these selected cases pass.
The stopping point is a documented extension of the evidence boundary, not a
new controller or a claim of global tolerance calibration. After this bounded
audit, proceed to genuinely windowed short-memory operator execution.
