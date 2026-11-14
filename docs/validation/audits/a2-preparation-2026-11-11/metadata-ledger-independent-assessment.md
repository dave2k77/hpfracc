# Independent assessment: resumed ledger proof FAILED; stop correct

**Overall passed: false. Accounting complete: false. Cap revision justified: false.**
No budget is proposed/applied. This adjudicates the bounded author attempt, not
packet consistency, numerical correctness, implementation approval or protocol freeze.
Parent owns closure and status updates.

## Verified inputs and actual outcomes

`input-verification.json` and `protected-inputs-after.json` verify all **162 protected
repository files, 210 author files and 218 advisory files**, HEAD and complete Git
index/stage-entry digests against the immutable assessment manifest:

`329b3305c6d6096bb780ea01a5afff60f45c1f318e123026093083b9c8cdf005`.

The nine packet files and 125 numerical source pins remain unchanged. The two
parent-owned status documents are outside protected-byte equality; this reviewer
wrote neither. Both archives were read member-by-member, without extraction or
execution of archive contents, and match all frozen child files:

| Evidence | Verified files / indexed bytes | SHA-256 |
|---|---:|---|
| Author index | 180 / 13,364,848 | `73f58082a53adfa30c0b15a4637b01e635d0077919adf0123446ccb4efd6211a` |
| Advisory index | 201 / 6,830,424 | `1413bd3c4a24e17e9831f23e20dbec2361f7048f472800cc0726583766ab7f64` |
| Author archive, all 210 members | 805,909 archive bytes | `bdfe85992d26b39b3a010c9da63af02b2177f75346afade45d435bfd12804551` |
| Advisory archive, all 218 members | 567,658 archive bytes | `6dc971501724142996ca155d7f4bb836a904280d85852f8b701010cb172ed1a3` |

Verified sources were copied unchanged into fresh scratch after import/output-target
inspection. Relocation alone changes their `__file__`-relative output root. Every
replay has pre-run source snapshots, raw streams and its **child's actual exit**;
the capture wrapper's exit 0 is not the gate outcome.

| Fresh copied replay | Actual outcome |
|---|---|
| `author-copy/semantic-replay/` | exit **1**, same 131 unresolved obligations; output JSON exactly matches original |
| `author-copy/mutants-replay/` | exit **1**, **11 tests, 5 failures, 0 errors** |
| `author-copy/structural-red-replay/` | exit **1**, 4 failures / 4 tests |
| `author-copy/structural-green-replay/` | exit **0**, 4 passes / 4 tests, necessary structural checks only |
| `independent-negatives/` | exit **1**, five reviewer-designed blocker probes reproduced |
| `advisory-copy/lower-replay/` | exit **0**, lower-proof output exactly matches original |

## Coverage is correctly counted, not semantically complete

`coverage-and-raw-record-proof.json` independently parses all three coverage tables:
**1,392 unique profiles covering 560 cases**, including 512 winner-pair labels and
352 supplementary fault labels across 32 FIT-REAL cases; the other 528 cases have
one profile each. All recorded structural exercises succeed. The representative
really has **505 symbolic records, 372 standalone payload terms and 1,389 links**.
An independent node recurrence matches all 505 per-record counts; `result.json`
has 51,560 nodes, maximum depth 9 across the representative.

These are conditional annotated model records, **not schema-valid run exports or
observed production arrays**. Source construction always emits indeterminate
results and unknown assessments. Winner labels do not prove selection/no-qualifier
semantics. **Complete valid exports demonstrated: zero.** Even the 356 coverage
rows with zero findings from the partial semantic checker are not complete-export
proofs. Parsed inventory counts are 24,336 quantity slots, 1,312 working comparisons,
1,056 fit-binding checks and 4,480 exact required comparison names. All 12 future
fit specifications retain null fit-array references; all 64 full observations
remain unobserved.

## Controlling failures and root causes

Independent reduction of the original annotated records, without invoking the
author's `unresolved()` implementation, reproduces:

- **42 unclosed same-mode identities:** 20 alpha, 20 dt, two input. Unrelated
  symbolic descriptors lack equality constraints; this does **not** show actual
  future values are unequal.
- **18 perturbed-order configuration linkages:** configuration retains base alpha
  without a checked equality to actual perturbed alpha.
- **69 unproved passes:** 37 comparisons and 32 binding checks. The model sets
  pass from complete operands or observation presence (`a2_model.py:365,373`),
  not the named finite/shape/tolerance or raw-byte-identity predicates.
- One unconfirmed normal completion and one emitted-evidence schedule with
  unavailable/unsafe ownership controls rather than a valid conditional context.

A symbolic pass can be legitimate **if explicitly constrained by its required
predicate**; a winner tag or present link is no substitute. These 69 statuses are
not 69 observed numerical failures. Reviewer negatives demonstrate that changing
a comparison reference from shape `[17,3]` to an existing `[17]` operand, and
rebinding an eager identity check to an existing JIT observation, both retain pass
and are accepted by `audit()`. A deliberately invalid selected result with every
assessment unknown and exit unconfirmed also passes the local structural
interpreter/audit. This is an invalid-success **negative fixture**, not an actual
author-generated successful export or a defect in the normative schema.

All five original survivors reproduce: uncharged log, manifest, old-index and
temporary-byte summaries, plus admitted unsupported D shape `[18]` truncated to
`[17]`. The original four accounting mutants alter reported totals; they expose
absence of an independent equation check, not necessarily an omission in the
pristine affine formula. A stronger reviewer negative removes the actual
`stdout_raw` coefficient and recomputes bounds while retaining its component
name: the audit still accepts and reports ledger closure. The omitted possible
log contributes up to 16,384 bytes; no actual future log length is asserted.

`symbolic_constraints_closed` is merely a guard-count comparison. Consequently,
the reproduced four-RED-to-four-GREEN transition is necessary structural progress,
not completed accounting. Missing semantic predicates, valid final dispositions,
maximal compatible causes/context bounds and the complete ledger/refusal proof
remain author-method blockers.

## Advisory schedule and resource adjudication

**INITIAL I0 / required QUALIFIER Iq / FINAL If batching is a defensible reading**
of protocol lines 376–394, schema-control contract lines 188–229 and packet-notes
lines 136–175. Atomic array commitment does not require a complete index rewrite
after every individual array. Preserve the explicit candidate-0 qualifier fault
checkpoint. Batching remains conditional on immutable publication, final reservation,
truthful post-kill authentication/accounting of every retained artifact and no
completed-file loss; a stale checkpoint never means completion. This is a design
interpretation, not an implemented crash/recovery proof or approval of the author.

The author's **372 array commits / 374 generations / 136,772,468-byte cumulative
lower endpoint** belongs to its chosen every-array draft schedule. It is **not**
actual overflow of a valid export or proof of intrinsic 16 MiB impossibility.

The copied advisory lower-bound derivation reproduces **614,416–616,852 bytes**
for mandatory old/new index coexistence in eight float32 tau1 cases. These
intentionally smaller-than-valid length skeletons are not exports. Their weaker
bounds corroborate, rather than replace, historical **614,748–617,184** bounds.
Batching does not remove this **524,288-byte index-transaction admission obstruction**.
The advisory's 41,598–46,044 completed-base node lower bounds apply to its chosen
all-inventory result placement, not every schema-permitted placement; nodes are
limited **per record**, never summed over files.

Conservative rational-length uppers exceeding 16 MiB do not prove actual overflow
or simultaneously attainable maxima. No complete constrained LIVE/cumulative
inequality is established for all required branches, including payload copies,
input, context/bindings, receipts, manifest's own length, every index generation,
logs and partial/staging/coexisting bytes. A finite schedule or upper envelope is
not that proof.

**Caps stay unchanged**: index transaction 524,288 bytes, nodes 32,768 per record,
evidence 16,777,216 bytes for **both LIVE and cumulative writes**. The authorized
revision required sufficient completed accounting; no sufficient replacement
values follow from this failed attempt. This does not make the existing caps
adequate for every mandatory positive case. Raising index/node caps alone would
not repair semantic failures; necessity to change the 16 MiB/numerical limits is
not established.

## Stop, limits and one method lesson

**Stopping was correct.** The controlling process failed; its pre-run model and
obligation sources match final frozen bytes. Captured later author processes are
finalization and index preservation only. That supports observed stop compliance,
not a global historical process audit. No post-stop model repair was performed here.

No new scope expansion, schema/representation redesign, or physical fitter/JAX/AD
execution necessity for the **conditional metadata proof** is established by
these blockers. This bounded finding is not a global proof of absence of other
obstacles. No numerical, operational-validator, selector, worker, full-suite or
GPU execution was performed; no scientific, implementation, freeze, A2 or Phase A
approval follows.

**METHOD lesson:** before bulk profiles, close one end-to-end conditional semantic
tracer and independently recomputed ledger equation; make arithmetic omissions and
invalid-success controls fail before treating structural GREEN as progress toward
completion. This is not an automatic recommendation for another cycle. Parent can
close this attempt as stopped/failed; any further work requires separate bounded
authority and an explicit acceptance target/stop rule.

Machine-readable adjudication: `assessment.json`. Actual handles, raw replay exits,
limits and digests: `raw-proof.json`; complete declared review inventory:
`evidence-index.json` with its separately captured verification. All new files are
inside this unique scratch directory; protected originals were not edited.
