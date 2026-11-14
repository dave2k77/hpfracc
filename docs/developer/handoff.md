# Development hand-off — 2026-11-14

Resume on `fix/phase-a-gpu-hardening`. This is a navigation and continuation
checkpoint, not a new scientific acceptance record.

## Where work stopped

The latest completed increment is the predicate-only `CAP-AUTH-TYPE-01` / VAL-017
repair and its static cap-after-qualifier trace. Read the
[repair checkpoint](../validation/audits/a2-preparation-2026-11-11/cap-after-qualifier-trace-2026-11-13/identity-repair-2026-11-13/README.md),
then its `outcome.json`, `final-manifest.json` and `completion-readback.json`.
The final manifest SHA-256 is
`d609c3a5a8797a2a13b8285d6956683e47ac375c79b160481626efd54d5267a3`.

The independent review verifies 230 scoped checks; parent readback verifies the
720-file review inventory and separately replays 69 independent controls.
All three valid canonical traces and their 38/27/28 emitted bodies are unchanged.
The corrected private helper is inside the checkpoint's
`correction-evidence.tar.gz`; its identity and member inventory are recorded by
`correction-result.json` and `correction-preservation.json`. The numerical library,
main static checker, scientific criteria and resource caps did not change.
The original rejected helper, failed verdict and counterexamples remain failures.

The local closure passed repository lint, 23 documentation tests, strict docs
and whitespace checks. Full numerical CPU/CUDA suites were not rerun for that
closure. Inherited archived checker/test lint findings remain disclosed; a green
repository lint command does not inspect every archived source.

## Next session

1. Check the live branch, working tree and remote before editing. Read the
   [Phase A programme](phase-a-completion.md),
   [issue tracker](bug-resolution-tracker.md) and
   [A2 draft protocol](a2-compression-budget-protocol.md).
2. Plan conditional resource accounting for the eight blocked FIT-REAL profiles.
   Use the three reviewed static traces as bounded examples, not a proof of
   whole-packet sufficiency. Derive actual branch-specific artifact and index
   transaction obligations; do not use mutually exclusive all-complete states
   or arbitrary ancillary allowances as necessary bounds.
3. Present a bounded next-work proposal for approval before new implementation
   or cap changes. Preserve missing fitter/worker/AD observations as missing;
   do not fabricate values to complete a ledger. A2 acceptance, protocol freeze
   and broader Phase A/release gates remain open.

## Delivery and evidence precautions

The owner separately authorized committing and pushing this checkpoint and
updating this hand-off on 2026-11-14. Obtain the containing revision from Git;
this file deliberately does not embed its own commit SHA or predeclare push/CI
success. Verify the remote branch and inspect CI for that exact revision.
Older `commit/push=false` fields describe their original local checkpoints and
are preserved, not rewritten as delivery receipts.

The [privacy-remediation package](../validation/audits/a2-preparation-2026-11-11/privacy-remediation-2026-11-13/README.md)
identifies the approved sanitized derivative. Its private original and personal
delivery receipts must remain outside the repository. The derivative omits eight
receipt members and is not a complete replay closure for every historical script.
Do not run archived collection/closure scripts blindly: they can write old
locations or require the pre-commit HEAD/index. After delivery, validate preserved
artifact bytes and declared scope rather than trying to recreate historical Git
state or interpreting an expected old-HEAD mismatch as source corruption.

Scratch directories are disposable. Resume from the repository's preserved
archives and manifests, not from absolute scratch paths in historical receipts.
