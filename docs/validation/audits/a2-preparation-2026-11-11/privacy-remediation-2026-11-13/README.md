# Fixed-trace evidence privacy remediation

Status: independent privacy/integrity review passed and parent readback verified.
The publication hold is resolved only for the identified sanitized derivative;
the original remains private. Final QA/disposition is recorded in `closure.json`.

## Authorized boundary

[authorization.json](authorization.json) records owner approval for this packaging
increment only. No numerical source, fixed-trace checker, scientific criterion,
resource limit, A2 accounting scope or protocol freeze is changed. No commit,
push or release is authorized. The previous exhausted fixed-trace correction
loop is not restarted. A blocking independent finding requires a stop and a
new owner decision.

## Original and derivative are different artifacts

The original rejected-review archive remains unchanged in owner-only private
storage, located by `../single-trace-privacy-hold.json`. Its original SHA-256 is
`264cfcc67365499f19705b4aef5c2ba0163e5bc0e3cb23d53ae263ff590a02c2`.
Its old repository location remains intentionally absent.

The new archive is
`sanitized-bundle/single-trace-independent-review-failed-evidence.sanitized.tar.gz`.
Its SHA-256 is
`26b0c4b4ccf59e415c9631471e01713ab8b27feda9351b077f831c557b981ff2`.
The new `sanitized-bundle/inventory.json` is the authoritative member inventory
for this derivative. Its SHA-256 is
`2164616b90e50bb961c3e45a9e1e15d6fc1b242a0c5c97ec53c258b32ea3e525`.

Exactly eight receipt members from the hold record are omitted. All other 449
member bodies are byte-identical to the private original's corresponding bodies;
all 457 original members are partitioned once. Tar header metadata is normalized
(uid/gid/mtime zero, empty user/group names, regular mode 644), not presented as
original metadata. No archive is extracted to the filesystem.

**Historical manifests inside this derivative are preserved, not rewritten.**
They may still reference omitted receipt paths and their hashes, and they may
contain earlier, now-qualified privacy/completion statements. They are evidence
of the original failed review, not authoritative completeness statements for the
sanitized copy. In particular, the old `private_telegram_receipts_excluded=true`
claim was contradicted by VAL-014 and is not restored by this remediation.
The derivative is intentionally **not** a complete input closure for every
historical capture/replay script. Do not execute those original replay scripts
expecting the omitted private receipts to exist. This does not alter the failed
review verdict or the separately preserved later passing fixed-trace review.

## Reproduce member-integrity verification

From the repository root, run the project-locked command below (line wrapping is
for display only):

```bash
uv run --frozen --no-sync python -B docs/validation/audits/a2-preparation-2026-11-11/privacy-remediation-2026-11-13/sanitize_evidence.py verify --bundle docs/validation/audits/a2-preparation-2026-11-11/privacy-remediation-2026-11-13/sanitized-bundle --inventory-sha256 2164616b90e50bb961c3e45a9e1e15d6fc1b242a0c5c97ec53c258b32ea3e525
```

The helper and bundle can also be copied to an isolated directory and the same
`verify` subcommand run with the external inventory pin. It does not require the
private original. This checks retained bytes and the recorded partition, not an
independent reconstruction of the original or a universal privacy guarantee.
Original-to-derivative equality is separately checked against the quarantined
original by the parent and independent reviewer.

The helper is a bounded, owner-controlled local packaging utility, not a
concurrent hostile-filesystem service. It rejects existing symlink paths and
unsafe, duplicate or nonregular tar entries; it never extracts tar paths. Its
local read/expansion bounds are packaging safeguards, not amendments to the
frozen A2 numerical/evidence resource limits. The externally trusted inventory
hash is required: replacing the inventory and the supplied trust pin together
is not authenticated verification.

## Checks and preservation

- The initial collected test failed because the new module was absent. This was
  module-absence RED, not an exercised semantic rejection.
- A separate behavioral RED demonstrated acceptance of an extra bundle file;
  the explicit bundle file allowlist repairs that packaging gap.
- The final packaging selection passes 23 checks. Additional defensive tests
  written after the first implementation are not claimed as test-first evidence.
- Ten independently disabled guard combinations are rejected by the test suite
  in copied-source mutation runs. These are synthetic packaging controls, not
  numerical simulations. Raw commands, before-snapshots, XML and actual exits are
  preserved in the test/mutation archives and their member indices.
- No private receipt is included in these captures. The capture allowlist is
  helper, tests and project test configuration only; delivery receipts are stored
  separately in private storage and never copied into the evidence tree.
- Privacy inspection is bounded to the explicit evidence inventory and recursive
  archive bodies, checking structured identifier fields, the observed private
  chat-identifier token and selected credential patterns. It is not universal
  secret detection or clearance for future changed files.
- `baseline-inputs.json` pins the 189 existing scoped inputs and raw Git-stage
  identity before remediation. Only the three enumerated status files may change
  after independent review; the original hold/delta and all prior evidence stay
  unchanged. `pre-remediation-status.tar.gz` preserves the old mutable statuses.

Source/finite-proof acceptance, this packaging review, broader A2 accounting,
scientific acceptance and remote delivery remain separate gates.

## Independent review and bounded disposition

`independent-review-verdict.json` preserves the passing scoped verdict;
`independent-review-preservation.json` binds its 198 indexed proof files and the
199-member `independent-privacy-review.tar.gz`. The reviewer independently passes
23 packaging checks, reproduces portable verification of the actual bundle and
rejects 16 CLI tamper controls. Its 100-file public-closure privacy scan has no
findings. The parent verifies every indexed file and archive member and scans
the review proof itself before accepting the verdict. Harness setup/schema
failures remain preserved separately from the successful reruns.

The derivative is approved only within these declared privacy/integrity limits.
`single-trace-outcome.json` retains its false blanket-publication and unqualified
increment-completion flags, while recording derivative-specific readiness. The
allowed publication/status delta changes no numerical or fixed-trace source,
tests, packet, scientific criterion, resource limit or historical evidence.
Further A2 accounting, implementation, freeze and commit/push remain separate
decisions; this review does not authorize them.
