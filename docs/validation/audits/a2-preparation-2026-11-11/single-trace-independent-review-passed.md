# Independent corrected fixed-trace re-review — scoped PASS

- **No security concerns or logic errors found in the approved scope.**
- Exactly ONE `STATIC/HYPOTHETICAL selected-REF-PASS-f64-GL` witness: N=3,
  two provided synthetic candidates and all 22 exact slots. No repair performed.
- `implementation_authorized: false`, `protocol_freeze_authorized: false`,
  `a2_acceptance_passed: false`, `broader_accounting_proved: false`.
  Parent tasks and the universal/resource-admission blocker remain open.

## Routing finding closed, original rejection preserved

The sole source delta is the 78-byte guard line in the existing X/0/1 loop:
`need(binding['binding_id'] == 'host-' + suffix, 'binding path-to-ID')`.
This enforces the frozen `records/binding-{binding_id}.json` relation rather
than adding an ID-uniqueness requirement or broadening the route allowlist.
All original tests, authority/schema/protocol and positive trace bytes match.

The independent executable starts with preserved positive JSON, mutates each
existing shared/candidate binding to a duplicate **and** fresh wrong ID, and
independently re-signs bindings, standalone/inline slots, index, export, manifest
and byte snapshots. All six full-schema/link/payload/ledger controls pass the
exact rejected checker and its one-line-deletion mutant. Each is rejected by the
corrected checker with exactly `binding path-to-ID`. No hash/schema masking.
The direct preserved original minimal counterexample (SHA
`ca98c968ab4a46ecc4e07baf9071308606fc569aee8138f2184a87ce36e7117e`) also now rejects for that exact relation.
Removing only the guard yields byte-identical rejected source. Thus the line is
necessary/sufficient for this reported fixed-route gap, not a universal verifier
or operational security proof. Original rejected verdict remains rejected.

## Actual executions and count boundaries

| Execution | Actual child exit | Parsed outcome |
|---|---:|---|
| Corrected original + six route parameters | 0 | 26 passed, 0 errors |
| Exact rejected source, same selection | 1 | 20 passed, 6 DID NOT RAISE, 0 errors |
| Isolated corrected source/data | 0 | Same 26 passed |
| Prior independent negative selection, corrected source | 0 | 23 passed |
| Portable positive materialization | 0 | Preserved/rebuilt bytes and ledger equal |
| Missing preserved trace control | 1 | Required-artifact FileNotFoundError |
| Independent authoritative positive proof | 0 | 36 records / 22 exact slots / 38 physical artifacts |
| Independent resigned route + guard-deletion proof | 0 | Six expected route rejections, old/mutant acceptance |
| Direct preserved minimal-counterexample proof | 0 | Exact route rejection |
| Historical receipt/snapshot checks | 0 | Corrector RED/GREEN/isolated/missing and parent readbacks agree |
| Scoped new-helper/test lint E4/E7/E9/F | 0 | Passed, not whole-repository lint |
| Protected pins before / after | 0 / 0 | Exact declared immutable state equal |

These are repeated verification selections/control families, **not summed into
an inflated test, case or numerical-suite total**. The six route controls are
mutations of one case, not six scientific cases. Commands, source snapshots,
raw stdout/stderr, actual child exits and pytest XML are retained per execution.

## Finite positive proof and byte accounting

The inspected, hash-verified prior independent proof was copied and freshly
executed against actual frozen fixture bodies. It independently constructs the
request, validates all schemas/routes/shapes, links/inline copies/completion,
qualification and fixed winner, phase arithmetic and publication order, derives
all 22 exact payloads with stdlib Fraction, and physically writes every scheduled
body, replacing the final index and recounting actual filesystem sizes.
It does not import the audited helper or run a numerical worker/fitter/JAX/AD/FD.
The resulting materialized bodies were independently compared byte-for-byte to
the preserved trace. All 38 emitted bodies and all ledger terms are unchanged.

| Quantity | Verified bytes |
|---|---:|
| Cumulative produced writes | 70427 |
| Final live files | 62970 |
| Peak old/new coexistence | 65229 |
| Index transaction | 19070 |
| Old index | 7457 |
| Final staging/index | 9676 |
| Manifest own serialized body | 4648 |
| stdout / stderr | 45 / 45 |

There are 39 charged writes, 40 events, two index generations, maximum
1204 JSON nodes and depth 9 in the 36 emitted JSON records. The canonical trace
container is 76188 bytes, SHA
`4ce9d81072637ddbde8bb3b8fd0ed3ce6941543261c57162c916a9ee9e4b4e61`; it wraps journal/ledger and is
**not one worker's export bytes**. This proves one finite hypothetical schedule,
not all-case/resource admission, runtime watchdogs, operation/bit counts or safety.

## Frozen evidence and capture qualifications

- Verified manifest SHA `05fd9b428c379e7d6b59db004b8fcc04bc6cec45e76aab2298ab68ff425749c3`, HEAD, full Git index and
  newline stage-entry pins; all 170 protected Root-union files, including the
  overlapping 125 numeric / 9 packet / 44 history inventories.
- Verified all 210 original-author files (5,388,615 bytes), 452 rejected-review
  proof files (15,627,201 bytes), and 235 corrector entries (6,356,655 bytes:
  232 regular files plus three readlink-UTF8 representations).
- Read back all 220 / 457 / 238 frozen archive members. Three archived links
  remain explicitly mapped regular UTF-8 data. Two old mutable status bodies
  are bound by the frozen failed archive, not today's status docs.
- Corrector current runs execute their recorded pre-run snapshots. Historical
  six RED failures, all GREEN results, actual exits, raw hashes, exact identities,
  source pins and order were checked; fresh reproduction agrees.
- The two initial guard representation mistakes remain setup failures, not
  numerical/trace REDs or protected drift. Their first receipts are transcriptions
  and post-failure source, not pre-run/raw attestation. Later captured reproductions
  are separate. The original author helper's snapshot/live-execution limitation
  remains disclosed; no historical OS attestation is invented.
- Process/PID/command/clock/ownership/counter fields within the trace remain
  inherited **STATIC/HYPOTHETICAL**, not observations of these review processes.
  The repeated-one packet digest remains illustrative, distinct from actual
  frozen packet SHA `2d572491100e76897bc4c298f1a75387110e6942a027da9eeae14b06aadec481`.
- Portable Python open-hook replay is not an OS sandbox/security attestation.
  No entire Git-status/directory-membership immutability claim was made; allowed
  parent status/new evidence writes remain allowed.

## Delivery

Only `/home/davianc/.hermes/cache/scratch/a2-routing-independent-review-Lnvtx4VN` was written. No protected source/data/test/contract, repository,
profile, credential or skill was changed. No implementation, freeze, A2 or
broader-accounting approval follows. Parent must read back/archive this verdict,
its proof index and final pins before publishing the scoped result. Remaining
correction cycles: zero; any later acceptance failure requires preserve/stop.

See `assessment.json` for exact proof paths and counts. `proof-index.json` and
`final-pins.json` are sealed after these assessments to avoid self-hashing.
