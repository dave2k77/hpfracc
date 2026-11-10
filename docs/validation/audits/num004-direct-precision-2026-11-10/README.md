# NUM-004 direct-weight precision repair evidence

Status: Selected-gate repair and local full CPU/CUDA suites verified on Python
3.13.13. Independent review and remote delivery are recorded separately; this
is not Phase A closure or release sign-off.

The [frozen scientific policy](../../scientific-acceptance.md) remains
`implicit-scientific-v1`: four required-success cases and two negative controls
per backend/dtype/evaluation variant. This bundle records new executions after
the deterministic direct-weight arithmetic repair. It does not replace or
reinterpret the immutable
[pre-repair bundle](../scientific-acceptance-2026-11-10/README.md), whose two
float32 direct gate failures remain historical evidence.

## Actual outcomes

| Backend | Dtype | Weights | CLI exit | Weak-decay independent scaled residual |
| --- | --- | --- | --- | --- |
| cpu | float32 | direct | 0 | 0.0163316 |
| cpu | float32 | stable | 0 | 0.052358 |
| cpu | float64 | direct | 0 | 4.99552e-06 |
| cpu | float64 | stable | 0 | 1.2035e-05 |
| gpu | float32 | direct | 0 | 0.0174327 |
| gpu | float32 | stable | 0 | 0.0164322 |
| gpu | float64 | direct | 0 | 5.54052e-06 |
| gpu | float64 | stable | 0 | 5.26731e-06 |

All eight reports pass the unchanged selected policy. Across 48 measured
configurations, all 32 required-success trajectories have `accepted=True` and
`policy_passed=True`. All 16 negative controls have `accepted=False` and
`policy_passed=True`; none is counted as a scientifically accepted trajectory.
The independent residual threshold remains one, and continuous-error budgets,
manifest selection, Newton iterations and tolerances are unchanged.

The float32 direct weak-decay residuals are `0.016331607987060983` on CPU and
`0.01743271498279139` on CUDA. The corresponding pre-repair residuals were
`12.453928382961097` and `8.497625302550595`, respectively. The unchanged stable
path's weak-decay residuals are `0.052358017489466745` and
`0.01643224778294578`; they are reproduced unchanged from the prior bundle.
Table cells use six significant figures; JSON retains the recorded precision.

## Arithmetic and compatibility boundary

The repair changes `_pece_weights` in the uniform deterministic solver, not the
Diethelm full-history discretization or default `weight_evaluation="direct"`
selector. The predictor uses cancellation-safe `expm1`/`log1p` power differences;
correctors use positive binomial series in the actual working dtype. Boundary
degrees 28/56 and 14/28 interior even terms have exact-arithmetic truncation
bounds below float32/float64 epsilon. These are not floating-point accumulation
or solver-error guarantees. See [uniform weights](../../uniform-weights.md).

**Legacy direct floating-point roundoff is not preserved.** Normalization,
Newton updates, diagnostic/acceptance thresholds and history policy are
unchanged. The explicitly selected stable interval-moment implementation and
the separate stochastic drift/noise kernel are untouched; the deterministic
repair is not evidence for a stochastic-kernel change.

## Artifacts and replay checks

- [executions.json](executions.json) records the eight actual commands and zero
  process exits. Each companion `scientific_acceptance_*.json` records the
  manifest/digest, measured rows and decisions, 61 source-input hashes,
  backend/devices, effective dtype/x64 mode, versions, dirty source provenance,
  and CSV/raw-artifact digests. CSV files retain measured values and separate
  acceptance/policy flags.
- `*_case_*.npz` preserves all 48 raw trajectories, times, inputs and per-step
  Newton diagnostics, including the nonfinite controls. Load with
  `numpy.load(..., allow_pickle=False)`; failed trajectories are not repaired or
  replaced with fabricated accuracy values.
- [verification.json](verification.json) records host reconstruction of all 48
  configurations, independent continuous errors/residuals, decision comparison,
  authoritative protocol/CSV/NPZ digests and 61 source hashes per report.
  [verify.py](verify.py) is the host replay checker, run from the repository root.
  Use `PYTHONPATH=. uv run --frozen python docs/validation/audits/num004-direct-precision-2026-11-10/verify.py`
  so the checkout's benchmark modules are importable.
  This is artifact/source/measurement integrity, not universal scientific
  acceptance or cryptographic execution attestation.
- [before-checkpoint.json](before-checkpoint.json) identifies the pre-repair
  checkpoint and [before-sources.tar.gz](before-sources.tar.gz), a 62-input source
  snapshot. Its 61 numerical/report inputs matched the prior reports. This
  preserves the original arithmetic separately from the repaired executions.
- [red-regressions.json](red-regressions.json), `cpu-red.xml.gz` and
  `cuda-red.xml.gz` retain two expected pre-repair regression failures on each
  backend, with raw/compressed digests and the test-source hash. These are
  failing TDD evidence, not accepted numerical outcomes.
- `tests/unit/test_docs_consistency.py` preserves eight old-bundle table guards,
  adds eight table guards for this bundle and checks the tracker's two paired
  current residual claims against these JSON rows and this evidence attribution.
  Historical residuals are explicitly identified as historical, not matched as
  current measurements.
- [development-regressions.json](development-regressions.json) retains the
  additional policy/comparison red evidence, initial three-test green runs, and
  the 187-test existing regression selection on both CPU and CUDA. The frozen
  initial test source matches its recorded digest. The expanded precision,
  moment, mask, nonlinear-gradient and selector checks pass 14 tests on each
  backend; [expanded-regressions.json](expanded-regressions.json) records their
  XML/source hashes. These selections are not substituted for a full suite.
- [suite-source-inputs.json](suite-source-inputs.json) captures 116 actual
  numerical, benchmark and test source inputs before both full-suite launches,
  not an invented Git-index identity. Both full CPU and CUDA processes exited
  zero with **850 passed, one performance test deselected per backend**; raw XML
  and source
  verification are in [full-suite-verification.json](full-suite-verification.json).
  Later documentation-only edits are checked separately, not presented as
  collection-time source or Git-index attestations.
- [additional-gates.json](additional-gates.json) records zero exits on CPU and
  CUDA for the existing aggregate (28 checks), hardening (12 checks) and history
  (8 checks) gates. These do not establish general scientific acceptance.
- [wheel-build.json](wheel-build.json) and
  [installed-wheel-smoke.json](installed-wheel-smoke.json) record a built wheel,
  an isolated hash-locked runtime-only environment, lazy imports, JIT and
  float32 order gradients. Independent selected-case residual reconstruction
  passes from the installed package. This is a CPU smoke, not a multi-platform
  wheel/release validation matrix.

The reports identify local dirty-tree execution, Python 3.13.13 and JAX/jaxlib
0.10.1. They are not a clean-candidate or release dossier. The local full CPU/CUDA
suites and isolated-wheel smoke above are verified. Independent review and
remote multi-version CI must be checked separately; earlier suites/reviews do
not verify this repair.
General nonlinear/stiff float32 accuracy, longer horizons and adaptive error
calibration remain separate gates. The coarse stiff-decay positivity/inaccuracy
control is still rejected, and Phase A remains open.

## Corrected stiff observation — outside the frozen matrix

[stiff-residual-float64-reference.json](stiff-residual-float64-reference.json)
is the authoritative scoped observation for CPU float32 half-order decay at
strength 30, 257 nodes, eight Newton iterations and `rtol=1e-5`, `atol=1e-7`.
It casts reference states to float64 before reconstructing the independent
equation. The unchanged stable path reports Newton convergence but has independent
maximum scaled residual `1.1998344389138271` in both the pre- and post-repair
checkpoints. Direct evaluation changes from the legacy `673.7955854875966` to
`0.20526358291781596`. This is an algebraic-residual observation outside the
frozen six-case manifest, **not continuous-accuracy acceptance** or a general
stiffness guarantee. The stable-path limitation remains open; it is not caused
by the direct-arithmetic repair.

[stiff-probe-history.json](stiff-probe-history.json) marks
`stiff-residual-observation.json` and its original probe as superseded: that
exploratory caller passed float32 states, leaving the half-decay host RHS
multiplication in float32. The original exploratory files are preserved but
must not be cited as the corrected independent float64-reference result.
The frozen-policy collectors already cast states to float64; all eight reports
and their selected acceptance decisions above are unchanged and unaffected.

## Separate metadata observation

[metadata-export-observation.json](metadata-export-observation.json) and
[metadata-export-probe.py](metadata-export-probe.py) reproduce API-001: both
uniform deterministic solvers retain a strong scalar-array order in
`SolverInfo.to_dict()`, so plain JSON export raises `TypeError`. Python-float
orders export successfully. Both checkpoints have this behavior; the metadata
implementation is unchanged. Selector/metadata equivalence tests do not claim
to repair serialization. See the [tracker](../../../developer/bug-resolution-tracker.md).

## Reproduce without overwriting evidence

Use a new output directory and run each dtype/evaluation combination:

```bash
JAX_PLATFORMS=cpu uv run --frozen --extra dev python -m benchmarks.numerical.scientific_acceptance --dtype float32 --weight-evaluation direct --check --output-dir path/to/new-evidence
UV_PROJECT_ENVIRONMENT="$HOME/.local/share/hpfracc/venv-cuda13" JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --frozen --extra dev --extra cuda13 python -m benchmarks.numerical.scientific_acceptance --dtype float32 --weight-evaluation direct --check --output-dir path/to/new-evidence
```

Substitute `float64` and/or `stable` for the other variants. Omitting
`--weight-evaluation` selects direct, never an automatic stable fallback. The
command scopes x64 per declared dtype; no hidden global float64 promotion is
part of the repair. Exact recorded commands are in `executions.json`.
