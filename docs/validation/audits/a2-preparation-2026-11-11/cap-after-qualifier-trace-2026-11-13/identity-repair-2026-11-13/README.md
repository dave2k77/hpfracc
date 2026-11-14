# Bounded fixed-authority identity repair

Status: **scoped correction independently approved and parent readback passed**.
Final local closure/QA is recorded in `outcome.json` and `completion-readback.json`;
this is not broader A2 or scientific acceptance.

This package is the single authorized correction of `CAP-AUTH-TYPE-01` / VAL-017.
It does not replace the parent package's failed verdict, blocked records,
rejected source or counterexamples. Those remain immutable historical evidence.

`authorization.json` limits the source change to a derivative of `frozen_case.py`
and its type-preserving identity predicates, with separate new regression/proof
tests. The main `a2_trace.py`, existing tests, authorities, fixture/schema/protocol
bytes, resource caps and numerical sources stay unchanged.

`preflight.json` records re-verification of the 30-file blocked manifest, all
1,706 indexed author files and all 532 indexed failed-review files. The
933-file baseline leaves only three named progress documents mutable; the other
930 files remain protected. The 87 explicit copied inputs include the original
counterexamples, test selections and valid trace references.

Required gates:

- Genuine rejection-test RED before editing the helper, then GREEN for the
  type-only mutations through authentication, construction and verification.
- Coverage of the affected case, witness and scalar identity comparisons, not
  just hard-coded rejection of the three original examples.
- All existing 126 selected checks, plus new targeted regressions and meaningful
  predicate mutation controls, with captured copied inputs and actual exits.
- Complete canonical trace and emitted-body equality for all three valid static
  cases, including their unchanged ledgers and explicit unknown dispositions.
- Parent evidence readback and fresh independent review of the corrected bytes.

## Corrected candidate and preserved self-checks

Only `frozen_case.py` changed: its case, witness and working-scalar comparisons
now use the contract's canonical JSON bytes rather than Python structural
equality. The corrected helper is
`f18113b7d05469e7cd657f8ef522114dcc6f6a175c3c60a32f55c21c44370c43`;
the main `a2_trace.py` remains
`819fbbafd5b42e9f1bc0c5be320e56146876ff7047a71073def9b9141fba92fa`.

`correction-result.json` retains the author report. The same 27 identity tests
run RED then GREEN: 26 RED cases lacked the expected refusal, including all nine
original counterexample/entrypoint combinations; one sibling verifier case raised
a TypeError instead. Those failure modes are not conflated. All 126 existing
selections and eight posthoc defensive checks pass. Isolated case, bundle and
scalar predicate reversions produce meaningful test failures; a full original
helper reversion reproduces the earlier failures.

`correction-preservation.json` binds 1,632 indexed files and 1,633 archived
members, with 42 captured runs checked by the parent. All 40 original regression
input occurrences other than the authorized helper remain byte-identical.
Parent fresh copied-input replays reject all three original authority variants
with ValueError and preserve the complete canonical traces and every emitted
body for the three cases (38, 27 and 28 bodies). Their ledgers, initial indexes
and journals remain unchanged. The positive golden container uses pretty JSON;
the promised equality is its complete canonical trace and exact emitted bodies,
not identical outer-container whitespace. Initial proof-helper/diagnostic
failures remain preserved rather than relabeled successful executions.

The parent replay proof is in `parent-replay-evidence.tar.gz` and
`parent-replay.json`. All 930 protected baseline files remain unchanged. The
corrected helper and new proof helpers lint clean; 106 inherited archived
checker/test findings and the earlier capture-helper E501 remain disclosed.
Bounded privacy scans found no unresolved pattern findings; local paths and
hostnames remain, and no universal privacy clearance is claimed.

## Independent approval and bounded closure

`independent-review-verdict.json` records a completed passing fresh re-review.
The parent verifies all 720 indexed proof files and 721 preserved members of
`independent-review-evidence.tar.gz`, including raw process/import/XML bindings.
The reviewer reproduces 230 passing checks: 27 typed-identity regressions,
126 earlier selections, eight author defensive checks and 69 additional
independent controls. The parent separately reruns those 69 controls with the
reviewed helper and confirms a zero child exit. Repeated executions are not
additional scientific cases.

The independent controls exercise further nested JSON type aliases, signed zero,
nonalias changes, nulls, extra fields and canonical encoding. Dictionary-order
identity is distinguished from the separately declared artifact publication
schedule; arbitrary reordered schedules are not newly admitted. The complete
three valid traces, every emitted body, ledgers, initial indexes and journals
remain unchanged. The original failed helper/verdict and all rejected attempts
remain immutable historical evidence.

This closes `CAP-AUTH-TYPE-01` / VAL-017 only for the reviewed local JSON identity
predicate and completes the authorized static cap-after-qualifier increment.
The final status delta is limited to the four named progress documents; all
protected source, tests, fixture/schema/protocol bytes and earlier evidence
remain unchanged. Current repository QA and final privacy/readback evidence are
separate from the independent verdict. Inherited lint and bounded privacy limits
remain as disclosed above.

No additional source repair after a failed re-review is automatic. No broader
case, numerical implementation, measured worker/fitter/AD observation, cap change,
protocol freeze, A2 acceptance, commit/push, release or phase closure is authorized.
