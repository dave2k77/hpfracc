# API-001: host export of supported scalar-array solver orders

Status: verified host-export increment; not Phase A closure or release approval.

Baseline: `7a11d2d62546470119618a75d15640e69e940a8d`. The baseline's remote
CI run [42971612268](https://github.com/dave2k77/hpfracc/actions/runs/42971612268)
is now independently read back as successful: Python 3.11–3.13, lint, docs and
the separate performance job all pass. That is NUM-005 evidence, not validation
of this later metadata change.

The two initial CPU regressions reproduce `TypeError: Object of type ArrayImpl
is not JSON serializable` for explicit and implicit uniform solvers configured
with a strong float32 scalar order. The scalar order remains an array in
`SolverInfo`; `dataclasses.asdict` copies it without JSON normalization.

The intended correction is confined to host-side `SolverInfo.to_dict()`.
Numerical construction, JIT/gradient paths, working precision, solver-order
support, diagnostics and scientific acceptance thresholds must not change.
Whole-result JIT returns and arbitrary user-supplied metadata serialization
remain separate from this concrete scalar-order export repair.

The RED test source, serializer source and raw compressed failure XML are
preserved before the correction. Subsequent verification is recorded separately.
The serializer snapshot is gzip-compressed to preserve its original blank EOF
line without weakening whitespace checks; its decompressed digest is verified.
The existing NUM-005 bundle retains its recorded source identity; a later
metadata change must not be retroactively attached to those measurements.

## Verification progression

The original two CPU regressions pass after the host correction. The expanded
150-test selection passes on CPU and CUDA; `targeted-{cpu,cuda}.xml` and
`targeted-test-source-before-style.py.txt` retain that intermediate checkpoint.
The final source removes two overlong decorator lines, annotates the already
supported scalar-array constructor arguments consistently with neighboring
tests, and corrects the older selector test's stale serialization comment.

Two initial full-suite launches (`proc_0c37ede29a3d`, `proc_ae25f0fda7e2`) were
explicitly cancelled before completion for that comment cleanup; they are not
passing or failing test runs. Their source archive/manifest is retained as
`source-snapshot-before-comment-cleanup.*`. The restarted runs use the final
`source-snapshot.*`, with 119 source inputs and 900 selected tests (one
performance case deselected). Both final processes now exit zero with 900 passes;
the CUDA exit was confirmed after delayed teardown, not inferred from its summary.
`executions.json` retains completed/cancelled process identities and commands.

Source review passes at tree `c199fe1390f73ce1d075d3720c8d4e2cd37398e6` without
blocking findings. `independent-review.json` preserves that scoped verdict, which
explicitly leaves the parent-managed full suites for separate verification.
The non-blocking RED-hash suggestion is applied in the archive verifier; later
outcome records and documentary updates do not change reviewed runtime/tests.

`verify.py --capture` creates the source capture once. After completion,
`uv run --frozen --extra dev python docs/validation/audits/api001-metadata-export-2026-11-11/verify.py`
checks exact source/archive hashes, preserved RED regression AST, comment-only
selector-test changes, the isolated runtime export-method change, all enumerated
full-suite identities and the earlier NUM-005 remote CI readback. It does not
claim new continuous numerical acceptance or a general metadata serializer.
