# Final-cycle schema/control contract (preparation only)

**Corrected DRAFT, pending independent FINAL review and owner approval.** All
witnesses in `schema-control-fixtures.json` are STATIC/HYPOTHETICAL
REPRESENTABILITY data, not worker/process/fitter observations. Their hypothetical
external packet digest is not this packet's digest. They demonstrate encoding and
relationships, not execution, fitting, numerical accuracy or security enforcement.
All certification and authorization flags remain false. This supplement is
normative together with the closed schemas and canonical predicates.

## Expansion, ownership and precedence

1. Deep-clone the case's reference-level request. Apply only mutations labeled
   `stage=references` in listed order, requiring every replace target to exist.
   Here `request/candidate_refs` really names the supplied reference list.
2. Dereference input, caps and each candidate occurrence independently, never
   sharing a mutable candidate object between occurrences or library entries.
   Generate required request metadata explicitly. Candidate-array refs resolve to
   cloned raw descriptors; future unobserved fits have null arrays, never fabricated
   values. Remove reference keys from the expanded request.
3. Apply `stage=expanded` replacements in listed order to existing paths only.
   `candidate:i` addresses the i-th supplied occurrence, NOT an ID-keyed map.
   Then apply `stage=evidence` replacements to a predeclared hypothetical evidence
   tree whose time cells exist but start null; the missing-row control additionally
   declares that cell unavailable. It is a fault recipe, not an observed array.
4. Derive the exact preinventory from the resulting supplied occurrences. Empty
   supplied lists have only shared slots and zero assessments. Duplicate supplied
   occurrences retain repeated IDs AND distinct `candidate_index` values throughout
   raw capture, slot inventory, assessments and the failure index. They are bounded
   invalid input, not coerced to unique IDs or canonical successful requests.
   For invalid canonical records `input_bundle.request=null`; preserve the bounded
   expanded raw request bytes and ordered occurrence list instead. For an exact
   canonical capture the raw hex copy may be empty ONLY when canonical(request)
   reconstructs the entire received capture and its raw_request_sha256; otherwise
   retain the bounded complete raw hex and use request=null if necessary. Never
   label a normalized or truncated copy as the full received input. A resource
   control's explicit cap override is applied after expanded replacements and
   before capture; injected checkpoints are not cap measurements. Do not discard
   a second duplicate or retain stale pre-mutation candidate slots.
5. Global syntax/types/tau/duplicate errors dominate request disposition, retaining
   every observable reason. Otherwise unsupported requests are indeterminate.
   Missing/corrupt required producer evidence makes an assessment unknown EVEN
   WHEN authenticated local configuration already demonstrates incompatibility;
   retain both reasons. A fully authenticated known-ineligible candidate is a
   completed assessment, not an unknown. Unknown dominates selection under the
   existing exhaustive policy. No observation is inferred from a static expected
   outcome. Combined controls exercise both duplicate ID+content and local
   incompatibility+missing producer.

## Typed dispositions and shapes

The schema's `x-disposition-table` is mandatory for slots and check records.
A slot has one primary `reason` and a nonempty ordered, duplicate-free `reasons`
list, with `reason == reasons[0]`; `none` is exclusive. Order uses the frozen
reason enumeration, except that the causal unavailable reason precedes secondary
locally observable reasons. Retain all reasons in assessments/results/index rows.

- Complete: exactly one raw/exact payload, binding digest present, reason none;
  complete shape equals expected shape. A partial slot has exactly one payload
  and binding, a proper nonempty time-leading prefix with unchanged trailing rank,
  and explicit missing suffix via the expected/completed shapes in the index.
- Unknown/not_evaluated: no payload/binding/artifact; an explicit unavailable or
  not_evaluated reason. Unsupported admitted requests retain true shapes and every
  slot/check disposition, without attempting unsupported arithmetic.
- Not applicable: no payload/binding/artifact; a typed known-ineligible reason,
  `unselected_candidate`, `stage_not_used`, `unused_quantity` or
  `singleton_fd_adjacency` as appropriate. `not_evaluated` is not a substitute for
  a known-ineligible reason. Already captured authenticated raw supplied data lives
  in the input bundle and need not be fabricated as a numerical observation.
- Comparison and binding checks retain pass/fail/unknown/not_evaluated/
  not_applicable, all observable reasons and their declared operand/observation
  IDs, including unfinished and unselected checks. Only the explicitly declared
  singleton adjacency has null operands. A pass has reason none; a fail has a
  real mismatch reason. Missing operands cannot pass or become a bound rejection.

Transport shape dimensions are integers 0..1024, rank at most two. This bound is
metadata admission, NOT numerical support: the numerical reference remains
N=1..17, M<=3. N=18's [18] and [18,18] expectations remain exactly representable
with null payloads. Do not truncate, drop or relabel them invalid. Input byte and
shape-product limits still apply. The rank-one (N,) and rank-two (N,1) forms are
distinct; explicit N=1 and N=3 rank-two controls cover both dtypes and aliases.

## Candidate-free full-history observations

`observation` has disjoint candidate and full-reference configuration/producer
variants. The full variant requires candidate_id/candidate_index null,
configuration.history=full, kernel=gl_signed, and the actual source-bound
`hpfracc.ops.fractional.grunwald_letnikov` or `riemann_liouville` invocation with
history=full. Source is the pinned `src/hpfracc/ops/fractional.py`, not `_fit_soe`.
No poles, fit method, fit iterations, t_max or rate grid are invented. The record
binds actual mode, operator alias, alpha/dt/input and raw full output. All 64
future FIT-REAL full observations (32 cases times two modes) and their exact-D
comparisons have explicit bindings. This variant does not assert they were run.

For unknown/not_evaluated/not_applicable observations the actual producer and
binding may be null; the declared configuration still identifies the variant.
No complete observation may use a null producer/binding or invent runtime
context. Future supplied fitter-call specifications use the separate closed
candidate_spec/producer_call_spec variant, with null fit arrays and no actual
runtime producer. They become actual candidate records only after real capture.

## Acyclic invocation/producer binding

`canonical(x)` is exactly
`json.dumps(x, sort_keys=True, separators=(",",":"), ensure_ascii=True,
allow_nan=False).encode("utf-8")`, with no newline/BOM. All values must first pass
closed canonical types and semantic predicates. The producer binding digest is
SHA-256 of canonical bytes of the **entire `producer_binding` payload record**.
It has no digest field of its own. Descriptor digests hash raw decoded bytes (or
canonical exact-array payload alone), never a slot/observation wrapper containing
its own binding digest. Candidate ID-free content identity remains unchanged;
volatile invocation/process data is separate and never added to that identity.

Binding order is: external frozen packet/source -> immutable input bundle ->
immutable launch receipt -> raw returned payload descriptors -> producer binding
-> slot wrappers -> observations -> immutable final process -> final failure
index -> later supervisor-export receipt -> external run manifest. A launch
receipt is written after launch but before the barrier releases numerical work;
it contains run/process/case/request/input/command/PID/monotonic-launch identity,
not observations or their digests. Final process and binding both link the same
launch receipt; a binding never hashes final process output. Independent host
exact work uses a truthful host launch receipt and `host_exact` binding context.
An unlaunched request cannot have a completed numerical observation.

Every binding includes run_id, process_id, invocation_id, observation_id,
case_id/request_id, input_bundle_sha256, packet_manifest_sha256, policy, source
list, exact configuration/producer, mode/branch, candidate occurrence or null,
actual input/alpha/dt/h/direction descriptors, and named returned payload
descriptors. Null h/direction means unused, not an invented scalar. The binding's
input/alpha/dt descriptors must agree with the invocation's actual slots and
configuration; each returned descriptor must agree with that observation's raw
or exact payload, including effective dtype/rank. Compare all these links to
input capture, launch receipt and final process, not just the digest string.
Selected input/ad-base branches require same-mode base identity; order branches
bind their actual perturbed order, not base coefficients. Static swapped-receipt,
case, branch, source and array controls recompute outer digests before semantic
rejection. These hashes establish consistency, not execution authentication.

## Phase receipts, watchdog and export

Every `phase_timing` is observed with ordered start/end monotonic ns and exact
difference, unavailable with null times and an explicit unavailable reason, or
not_applicable/not_evaluated with null times and the appropriate reason. No fake
zero stands for an unavailable interval. Monotonic fields use the same
supervisor clock and bounded integer transport; out-of-envelope measurements
become unavailable/indeterminate, not clipped.

The final immutable process records preflight, launch-through-confirmed-exit
watchdog interval, terminate/grace interval and kill-through-confirmed-exit
interval, including explicit exit confirmation and barrier status. Preflight
ends before launch. Grace begins at terminate and ends at confirmed exit or kill,
with an intended 1s supervisor wait, not a scheduling guarantee. Normal exit has
no grace/kill phases. Failed launch has no watchdog interval. Unconfirmed exit
has unavailable duration/exit code and cannot be marked complete. At watchdog=0
the barrier remains held (never released); terminate/kill and confirmed exit
are still required. No numerical work is authorized across the held barrier.

A later immutable `supervisor_export` references the finalized failure-index
bytes/digest and final process digest and records the just-completed index-write
interval. The index does NOT refer to that later receipt, and the process does
NOT contain its own write duration/digest or an index-write duration that has not
yet occurred. Receipt-writing time is not recursively measured. Unavailable
export timing is explicit. Preflight, grace, kill, index write and receipt export
are NOT concealed in a 60-second total guarantee; 60s is only the watchdog
trigger proposal. No such process/control was executed in preparation.

## Output security and byte-accounting lifecycle

These are missing-invariant repairs for a future implementation, not a claim
that a currently implemented worker was exploitable. A future request must use
a freshly created, exclusively privately owned root (0700, owner euid), opened
as a directory handle, and fresh privately owned `arrays`, `records`, `tmp` and
`logs` subdirectories. Never reuse/adopt an existing request root. Preserve the
lexical artifact grammar, but lexical safety alone is insufficient. Create
relative to trusted directory handles with no-follow and exclusive creation;
reject symlinks at every component, absolute/dot/dot-dot/backslash/drive paths,
nonregular files, wrong ownership, hardlinks (nlink != 1), changed inode/device,
and containment failures. Verify before and after creation/readback with
handle-relative lstat/fstat and owner/device/inode/type checks; read only through
no-follow handles. Do not resolve/follow an unsafe object to inspect its target.
A private root plus no-follow rules is required, not an unchecked `resolve()`.
Existing unsafe targets cause bounded `unsafe_output_target`/indeterminate and
no overwrite, traversal, adoption or cleanup of unowned files. A bounded receipt
can be kept in the supervisor's separately fresh owned failure root; its bytes
are charged to the same request budget. If even that cannot be created safely,
report the bounded external failure rather than claim persisted evidence.

Array artifacts are immutable: each slot may commit ONCE, whether complete or a
partial prefix. Never overwrite/upgrade a partial artifact. Use owned exclusive
staging and no-replace publication plus verified readback. Failure-index
replacement alone is atomic replace of an owned regular target with a checked
owned temporary file, with both old+new coexisting bytes charged before write.
No replacing symlinks/directories or foreign targets. Artifacts and final process/
export receipts are immutable; index generations are not silently counted as
free. A prior index digest is not required inside its successor.

Charge every owned byte from creation through cleanup: input, immutable arrays,
records, logs, temporary/partial/coexisting index files, root-failure receipts,
and retained superseded bytes. The 16 MiB ceiling is the maximum simultaneously
retained charged bytes; keep a cumulative bytes-written counter too, separately
bounded by 16 MiB, so deletion or index rewrites cannot create unlimited output.
Reserve before creating/writing, include headers/whitespace/newlines, release
live bytes only after confirmed removal of owned scratch, and never subtract
from cumulative writes. Never delete completed evidence, hide/unlink open files,
use external mirrors, or ignore unowned files in an adopted root. Unowned files
are forbidden, not free storage. Index updates must be scheduled within the
remaining cumulative budget; when no further safe update fits, stop work before
it becomes unreportable and use the pre-reserved final update. No unlimited
rewrites are promised.

The result evidence_bytes and index bytes_accounting fields are snapshots just
BEFORE serializing that record. The pending record/transaction is reserved and
charged before write; these snapshots are not falsely labeled final totals and
do not require a record to predict its own length. Final run-manifest entries
give actual immutable artifact byte lengths, excluding the manifest's own hash;
the manifest's actual length is additionally charged by the supervisor. Neither
an omitted self-hash nor a snapshot exempts any produced byte from accounting.

Before launch, render the entire input bundle and worst-case failure-index
metadata, including ALL quantity rows, comparisons, fit-binding checks,
occurrences, phase/process references and bounded receipts. Count also the
maximum coexistence of old+new index and the later export receipt. The 256 KiB
input and 512 KiB failure reservation cover their whole proposed transaction,
not each unlimited generation separately. If that reservation or total/cumulative
16 MiB cannot cover the complete transaction, preserve a bounded
`reservation_exceeded`/indeterminate preflight receipt; no arithmetic and no
omission of checks to squeeze through. Concrete size templates are illustrations,
not proof that every admitted case can launch. Reserving bytes does not prove
peak memory, transient arithmetic feasibility or runtime scheduling.

## Artifact routing and preparation limits

`record-schemas.json` root admits every emitted JSON record, including standalone
`record_type=slot`, launch receipt, producer binding, process and supervisor export.
`x-artifact-routing` additionally freezes exact safe path-to-definition mappings.
Input/failure records also contain all comparison and binding dispositions before
launch. stdout/stderr are bounded raw byte streams routed as logs, not JSON
records; the run manifest is a closed artifact list outside its own hash scope.
Only explicitly listed routes may be emitted. Static witness containers are not
run artifacts and are not passed to the run-record root as a whole.

No product validator, selector, worker, CLI or tests are implemented here. Scratch
stdlib checks only exercise finite hypothetical encodings, closed required keys/
types, semantic links, DAG and one-field mutants. Their passed checks are neither
independent FINAL review nor owner approval. Universal representability under
finite reservations is expressly not claimed: reservation failure must remain
bounded and indeterminate. All future real fitting/application/AD, GPU/full jobs,
portable operational replay, scientific acceptance and broad Phase A gates remain
open.

## Final-cycle change/check ledger

| Failed finding | Static correction |
| --- | --- |
| MATH-REREVIEW-01 / V-B3-01 | Disjoint candidate-free full configuration/producer, null IDs/indices, pinned actual full-history entrypoints; all 64 future full bindings retained without observed arrays. |
| MATH-REREVIEW-02 | One mandatory typed status/reason/payload table, real known-ineligible reasons, all observable causes retained. |
| MATH-REREVIEW-03 / V-B4-02 | Transport shape metadata through dimension 1024; unsupported [18]/[18,18] are representable, not supported numerical requests. |
| V-B1-03 | Explicit references/expanded/evidence mutation stages, existing-target-only replacement, independent occurrence clones and post-mutation inventory. Empty controls lose stale candidate rows; duplicate occurrences remain distinct invalid raw input. |
| V-B4-04 | Closed launch/process/phase/later-export records; preflight, watchdog, grace, kill-to-exit, barrier and index-write timing without self-digest/write cycles. |
| V-B3-05 | Standalone record_type=slot root plus closed routing for every JSON artifact and explicit raw-log/temp exceptions. |
| V-B3-06 | Canonical typed producer-binding payload, exact UTF-8 recipe and independent volatile invocation/process DAG; semantic swapped-receipt/case/branch/source/array controls. |
| V-S01 | Fresh private ownership, handle-relative no-follow creation/readback, containment/regular-file/single-link checks, immutable arrays versus atomic owned index replacement, bounded unsafe-target refusal and charged live/cumulative lifecycle. This repairs a missing invariant, not an implemented exploit. |
| Related suggestions | Eight rank-two M=1 controls, eight combined-failure controls, synchronized ad-base joint-cotangent inventory, all unfinished comparison/fit-binding dispositions initialized and retained. |

Fresh parsed totals: 560 cases, 840 raw-array descriptors, 160 candidate specs
(12 future real specs still have no observed fit arrays), 24,336 quantity slots,
1,312 working comparisons, 1,056 fit-binding checks, maximum 954 slots per case,
10 unchanged parser controls, 4 unchanged identity controls and 6 unchanged
shortcut controls. All original 544 case IDs, request signals/budgets/configs,
expected stop gates and 832 original raw descriptors remain; no failure was
hidden. Only broken mutation staging/empty-list inventories were corrected,
plus the 16 scoped controls. Threshold/FD policy bytes are unchanged.

The static witness file has 17 linked hypothetical bundles, 38 additional typed
status witnesses, 64 future full-observation bindings and 12 semantic tamper
recipes. Bounded stdlib self-checks exercised all 17 root variants and every
declared status/termination path, closed required keys/types, 37 structural
negative controls, all 12 non-no-op semantic controls with outer digests
recomputed, exact small synthetic bound/qualification components and the binding
DAG. The exact 1,699-byte mathematical section remains byte-identical, SHA-256
`c6b59cae6d948222fdb9c5c0a6287491426864964207f45b747501b1970e1f9e`.
These are preparation self-checks, not independent FINAL approval.

**Residual resource limitation, not a pass:** complete concrete templates retain
all fields/checks. Maximum input capture is 183,673 bytes; maximum single
complete-descriptor index is 446,741 bytes; maximum initial+new-index transaction
including an 8,192-byte receipt allowance is 792,318 bytes. The concrete rendered
receipt set is at most 2,796 bytes. Conservative complete-index metadata can have
36,687 JSON nodes (counting keys), beyond 32,768; maximum depth is 7. Eight
float32 tau=1 FIT-REAL templates therefore require bounded preflight
reservation_exceeded/indeterminate under this conservative transaction rule,
not numerical work or omitted check rows. No cap is enlarged, no signal/budget
or stop gate is retuned, and no universal admission/feasibility claim is made.
Resolving broader empirical attainability or resource feasibility is not granted
by this final preparation repair. Preserve-and-stop still applies.
