# Numerical development progress: 2026-09-14

Status: Historical snapshot of the first development cycle on top of
`c6c40784b90d244b3ec4ad49178c8bc323a2b274`, subsequently committed as `37aaac6`.
Phase A remains active.

The later [native-convolution follow-up](../validation/native-convolution.md)
reduces the float64 failure to pure JAX and avoids that path with a windowed lag
scan. Both precision benchmarks now complete and the full suite passes 508 tests.
The baseline and open-work list below preserve the earlier checkpoint.

The [starting checkpoint](../validation/development-2026-09-14.md) records the
initial environment, baseline results, and pre-execution coupled-audit design.

## Delivered

1. **Coupled adaptive evidence:** 48 completed configurations across component
   scales, fractional orders, tolerances, both estimators, and float32/float64.
   Independent continuous forcing checks, frozen-mesh sensitivities and compiled
   retained-prefix failure checks extend the scalar evidence. Subdivision remains
   experimental; see the [audit](../validation/adaptive-coupled.md).
2. **Direct short-memory execution:** retained-window-only weights and direct
   convolution replace dense short-memory application. Independent discrete sums,
   discarded-tail bounds, exact Caputo cutoff identities, complex signals, per-state
   orders, and gradients validate the change. See the
   [contract and measurements](../validation/windowed-history.md).
3. **Recorded evidence:** CSV/JSON configuration, source digests, dirty-state and
   runtime provenance for the coupled sweeps and 24-row float32 windowed benchmark.
   Existing historical audit bundles are preserved.

## Final numerical verification

On Windows CPU, Python 3.13.13 and locked JAX/jaxlib 0.10.1:

| Check | Result |
| --- | --- |
| `uv run python -m pytest` | **506 passed**, 1 performance test deselected, in 522.48 s; confirmed process exit 0. |
| `uv run python -m benchmarks.numerical.validation_summary --check` | All **20 rows passed**, exit 0 after the implementation change. |
| Coupled audit, float32 and float64, `--check` | Both exited 0; **48 completed runs** overall. |
| Windowed benchmark, float32, `--check` | Exit 0; **24 rows**, each with seven repeatable timed executions and numerical agreement. |
| `uv run mkdocs build --strict` | Passed. |
| Ruff 0.16.7 on all seven changed/added Python files | Passed, using `uv run --with ruff ruff check ...`. |
| `git diff --check` | Passed. |

The final full-suite exit resolves the initial baseline's post-summary shell
timeout. It does not replace a release-candidate evidence run. Ruff is still
absent from the declared dev environment; the 25 pre-existing repository lint
findings remain outside this work item.

## Open follow-ups

1. **Isolate the Windows native float64 benchmark failure.** The mixed-configuration
   repeated sweep reproducibly exited with `0xC0000005` during JAX output
   synchronization. Isolated configurations, the single-round diagnostic sweep,
   and 100-repeat single-configuration probes of both implementations passed.
   Preserve these distinctions when reducing the reproducer or comparing a
   second backend/environment. There is no complete float64 timing bundle.
2. **Recursive SOE execution and fitting validation.** SOE still applies approximate
   weights densely. Windowed short-memory completion does not close this gate or
   the uncompressed solver-history work.
3. **Broader memory and numerical evidence.** Per-state/reverse-mode/accelerator
   scaling and sampled peak memory are still open; compiler buffer accounting is
   not a peak-memory measurement. Adaptive calibration and stronger nonlinear/stiff
   implicit validation remain open scientific gates.
4. **Reproducible tooling and release preparation.** Declare lint tooling, align CI
   with uv/lockfile, resolve baseline lint/import-convention drift, and eventually
   capture a reviewed clean candidate. Package version remains `0.1.0a1` with these
   changes recorded under Unreleased.

No commit, push, or release action was performed in this work session.
