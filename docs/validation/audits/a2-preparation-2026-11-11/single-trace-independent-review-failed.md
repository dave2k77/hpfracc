# Independent fixed-trace review: REJECTED

Exactly one fixed STATIC/HYPOTHETICAL `REF-PASS-f64-GL` trace was reviewed.
No author/root/index changes, repairs, bulk-case expansion, cap change, numerical
worker, JAX/fitter/application/AD execution, staging, commit or push occurred.
All implementation/protocol-freeze/A2/broader-accounting authority flags remain false.
Parent owns task/status closure; none is marked closed here.

## Blocking fixed-target finding

`verify_trace` accepts `records/binding-host-0.json` after changing only its
`binding_id` from `host-0` to `host-1` and rebuilding the outer links and ledger.
The frozen `x-artifact-routing` requires `records/binding-{binding_id}.json`;
`schema-control-contract.md:233-240` explicitly requires those exact mappings.
The accepted artifact therefore has the wrong route and duplicates the retained
other binding's ID. Source `a2_trace.py:516-524` checks lexical safety and the root
schema rather than the route identity; `338-365` omits this binding ID relation.

Minimal reproduction: `t = a.make_trace(lambda r: r["records/binding-host-0.json"].update(binding_id="host-1")); a.verify_trace(t) returns success.`

The negative retains correct source/mode/occurrence/payload identities. All outer
manifest/index/export/binding hashes and the recomputed byte ledger pass separate
checks. The standalone reproduction records acceptance, then **actually exits 1**
on the route assertion. The reviewer test run has **22 passed / 1 failed** (actual
child exit 1; failing test: `test_binding_filename_must_match_its_declared_binding_id`).
This is a fixed-target gap, not a request to implement a general operational parser.
No repair or alternate model was attempted.

## Independently verified positive evidence

- All **20 supplied focused tests pass** in the copied sandbox (actual child exit 0).
- All seven archived RED/GREEN snapshot pairs replay with original exact selections,
  identities and exits: RED counts 1/4/3/4/4/2/2, final GREEN 20. First RED is
  collected module absence, not reached semantic assertions; the other 19 are
  actual missing-rejection failures. The controlled missing `protocol.json`
  produces one collected failure and actual child exit 1.
- Frozen input authority equals the actual root fixtures and selected static witness.
  The illustrative repeated-`1` digest is **not** the actual packet digest
  `2d572491100e76897bc4c298f1a75387110e6942a027da9eeae14b06aadec481`.
- All **36 JSON records**, **22 exact payloads**, standalone/inline copies,
  input/alpha/dt/source/occurrence/mode/shape/returned-descriptor relationships,
  digest DAG, initialized/final inventories and conditional phase/publication/
  completion relationships hold for the preserved positive artifact.
- Decode the supplied raw float64 input as `[1,1,1]`, alpha `1/2`, dt `1/4`.
  Independently recur GL weights `[1,-1/2,-1/8]`, prefix `[1,1/2,3/8]` and
  direct output `[2,1,3/4]`. Recompute every candidate's fitted weights, residual,
  matrix/output, component error/bound and mean-square analytic input gradient.
  `bad1` has bound `[0,1/4,1/4]`, exceeding tau `3/16`; `a-good2` has zero
  bounds and is the sole qualifying minimum supplied rank. This uses stdlib
  Fraction on already-provided N=3 payloads, not a library numerical oracle.
- Materialize all 39 writes and the one atomic replacement from the stored journal
  into a new reviewer directory; stat-recount live bytes after every event.
  **Cumulative 70427; final live
  62970; peak 65229;
  old/new index transaction 19070 bytes.**
  The manifest's own 4648 bytes, both
  45-byte raw logs, 7457-byte initial index, 9676-byte final staging index, process,
  export and output-control are included. Atomic replacement retains the staging
  inode and writes no second copy; the old write remains cumulative, not final live.
  Snapshots are independently checked as pre-write values, not final totals.
- Reviewer controls reject eight real omitted charges with every total recomputed,
  selected-with-unknown (either occurrence), wrong winner, same-source wrong mode/
  descriptor, wrong dt/source/occurrence/case, re-bound false exact gradient, reversed
  phase, journal omission and reordering. These are the same fixed case, not new cases.
- Positive record maximum is 1204 nodes / depth
  9. These finite measurements do not establish any
  general cap sufficiency or universal/all-case byte bound.

## Evidence and scope limits

All hypothetical command/PID/time/ownership/counter fields remain representability
examples; no worker, security/no-follow mechanism, watchdog, operation/bit budget,
FIT-REAL or AD/FD result is proved. One static schedule is not a runtime/cap theorem.
The capture helper archives before spawning, but runs live HERE sources instead of
archived copies; exact portable replay supports consistency, not historical OS
attestation or a guarantee against between-snapshot/run edits.

Exploratory invocation/observation ID-collision acceptances are preserved but not
scored as separate defects without assuming an additional uniqueness requirement.

Final guards reverified **210 author files /
5388615 bytes**, the frozen manifest, 170 protected repository-union
files (125 numerical, 9 packet, 44 history groups with overlap), HEAD and exact Git
index/stage pins. Authorized parent status/new archive writes are not misclassified
as changes to protected bytes. Only this reviewer's unique scratch tree was written.

## Proof handles

- `a2_independent_proof.py`, `independent-positive-proof.json`,
  `independent-record-table.json`, `independent-exact-arithmetic.json`,
  `materialization-timeline.json`, `materialized/`.
- `test_a2_independent_negatives.py`, `targeted-negative-tests/raw.xml`,
  `targeted-negative-tests/process.json`, `targeted-fixtures/binding-route-mismatch.json`.
- `a2_minimal_reproduction.py`, `minimal-reproduction-receipt.json`,
  `minimal-reproduction/` raw stdout/stderr and actual child-exit receipt.
- `tdd-independent-replay.json`, `replay-red-01..07/`, `replay-green-01..07/`,
  `missing-input-control/`: raw XML, actual exits, exact new pre-run source captures.
- `final-protected-pins.json`, `proof-index.json`, `proof-index-verification.json`,
  `assessment.json`, `final-pins.json`.

Proof-index SHA-256: `ea1cf140c393c5426e328833bff087ba52c37b0fcf297fd9ae83dd75ffe3d4d9`. It excludes its own file, index verification,
assessment JSON/Markdown and final receipt to avoid self-reference. Their exact
hashes/sizes are separately bound by `final-pins.json` and the tool readback.
