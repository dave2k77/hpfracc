# Phase A completion work programme

Status: Active work programme; not a phase-closure or release record.

The [roadmap](roadmap.md#phase-a-gates), engineering contract and
[release checklist](release-checklist.md) remain authoritative. This ledger makes
remaining gates explicit; it does not reduce them or approve deferrals. No release
version, tag, publication or remote write is authorized by this document.

## Baseline and verified first increment

- At programme start, local `main` and freshly fetched `origin/main` agreed at
  `6f5d3d3`; reviewed environment/GPU hardening and the first coverage increment
  are now committed as `ec75960` on `fix/phase-a-gpu-hardening` and pushed.
- The separate
  [GPU normalization bundle](../validation/audits/gpu-normalization-2026-11-10/README.md)
  records the pre-programme baseline: 713 default tests on each backend, both
  28-row aggregates, examples, versions and source hashes. It is evidence for
  that recorded dirty tree, not automatically for later changes.
- The first coverage increment adds `tests/unit/test_fft_order_gradients.py`:
  Caputo/GL/RL, selected scalar and per-state orders, float32/float64, eager and
  compiled derivatives, finite differences and full-history comparisons.
  Its initial 24 cases passed separately on CPU and CUDA, but independent review
  found that the nominal float32 cases promoted forward computation to float64.
  Those initial passes do not establish float32 coverage. The corrected matrix
  scopes precision per dtype and asserts forward/objective as well as gradient
  dtypes; all 24 corrected cases passed on each backend. The
  [precision-correction bundle](../validation/audits/fft-order-gradients-2026-11-10/README.md)
  preserves both the red regression and corrected evidence (tracker VAL-004).
- Replacement full suites passed 737 tests on each CPU/CUDA backend with one
  performance case deselected and confirmed zero process exits. Raw JUnit and
  source verification are in the same precision-correction bundle. Remote CI
  and the remaining Phase A gates are separate; this is not phase closure.

## A1: Numerical correctness and error control — open

Required work:

1. Define a justified adaptive calibration/control target and its assumptions.
   Address the known weak-startup and cubic cancellation counterexamples;
   additional probes or decreasing errors alone do not establish calibration.
2. Extend beyond prescribed C2 forcing to a declared weak-forcing/state-feedback
   scope, or seek explicit review of any narrower completion claim. Account for
   retained old-history error and separate roundoff from discretization error.
3. Extend independent nonlinear/stiff and mixed-scale precision evidence beyond
   the current manufactured and half-order cases.

Acceptance:

- Freeze analytic/reference cases, precision, component scales, tolerance/work
  sweeps and pass criteria before collecting the completion evidence.
- Verify errors against independent continuous/discrete references, including
  failure paths, not only controller or Newton status. Preserve failing cases.
- Check state, parameter and order derivatives plus eager/JIT agreement on every
  newly claimed path. State adaptive frozen-mesh semantics explicitly.
- Do not promote a heuristic indicator or change a default silently.

First numerical work package: separate scientific acceptance from measurement
integrity. Existing audit `--check` commands intentionally test complete,
coherent measurements, not necessarily accurate trajectories. Preserve their
semantics and historical artifacts. Add a separately named acceptance policy
that rejects finite, Newton-converged but independently inaccurate results,
missing/duplicate required cases and nonfinite required-success trajectories.
Expected-failure cases must remain separate from required-success cases.

The first selected implicit policy is now implemented in
`benchmarks.numerical.scientific_acceptance`; its
[frozen protocol and limits](../validation/scientific-acceptance.md) define the
six-case manifest per precision/evaluation/backend. The
[pre-repair bundle](../validation/audits/scientific-acceptance-2026-11-10/README.md)
preserves the direct float32 independent-residual failures on both backends.
NUM-004 repairs cancellation in the same direct Diethelm coefficients; the
default direct selector, full-history method, normalization, Newton updates and
tolerances are unchanged, but legacy direct floating-point roundoff is not
preserved. The explicit stable interval-moment route and separate stochastic
kernel are untouched. Eight
[new reports](../validation/audits/num004-direct-precision-2026-11-10/README.md)
pass the unchanged policy: all 32 required-success cases are accepted, while
all 16 negative controls remain scientifically rejected with passing policy
flags. This verifies the selected-gate repair, not arbitrary nonlinear/stiff
float32 accuracy, adaptive error calibration or A1 closure. Local Python 3.13.13
full CPU/CUDA suites each pass 850 tests with one performance case deselected
and confirmed zero process exits. Independent final review, remote multi-version
CI and release gates remain separate from this local evidence.

The subsequent NUM-005 investigation independently isolates CPU float32 history
accumulation as the dominant cause of a stable-weight stiff-case residual
failure outside the frozen manifest. Its reduction-only candidate passes the
unchanged four focused regressions on CPU/CUDA and all eight frozen-policy
reports, with failures and a bit-matching component decomposition retained in
the [new bundle](../validation/audits/num005-stable-precision-2026-11-10/README.md).
Twenty-five additional compatibility cases pass on each backend. Full CPU/CUDA
suites each pass 879 tests with confirmed zero exits and 118 source inputs
unchanged. VAL-007 retains both passing CUDA pairs and captures different
optimized reduction emitters for identical f32-only StableHLO/configuration.
Emitter selection cause remains unresolved; review and delivery are separate
records. No A1 gate or general stiffness/reproducibility claim is closed here.
The independently reviewed NUM-005 repair is delivered as `7a11d2d`; its exact
revision now passes remote Python 3.11–3.13 CI, lint, strict docs and the separate
performance job. Later changes require their own verification.

API-001 corrects host-side `SolverInfo.to_dict()` export for supported strong
scalar-array orders without changing numerical execution. Its original two
failures are preserved; the correction passes the targeted selection and full
900-test suites on CPU/CUDA with confirmed zero exits and 119 captured source
inputs unchanged. Source review passes; review scope, later evidence and delivery
are recorded separately. This does not resolve the per-state solver-order scope
decision or any broader A1 gate.

For any claimed tolerance-controlled case, test accepted-node errors against
the stated componentwise reference scale, not just decreasing errors. Require
an independently justified error-control contract before extending a bound to
nonlinear feedback. General feedback control requires a trajectory reconstruction,
an enclosed Volterra defect and a valid propagation bound; Jacobian sampling or
additional trial subdivisions alone do not supply those assumptions.

The next [private defect-control investigation](../validation/audits/a1-defect-design-2026-11-11/README.md)
preserves checked reconstruction/propagation derivations, weak-forcing controls
and a working constant-candidate feedback tracer with 15 passing private tests.
Independent review distinguishes an archiveable scoped artifact from A1
acceptance: the latter remains false. No solver/controller, verified working-
precision certificate, narrower completion scope or per-state decision is
introduced by this investigation.

The next [supplied-trajectory increment](../validation/audits/a1-trajectory-defect-2026-11-11/README.md)
implements full-history PL/affine-power defect evaluation and signed continuum-
cell model ranges in a private host-only benchmark module. It preserves the
PECE nodal mismatch, weak-power and coupled/mixed-scale inputs, old-history
negative controls and explicit cell-work exhaustion. After the separate Gamma,
positive-panel and underflow repairs (VAL-008/009/010), 81 focused cases pass in
each CPU/CUDA process; the diagnostics themselves run on the host in float64.
The Gamma-only checkpoint's two 964-test full-suite passes did not clear its
failed independent review; that source and both failures remain preserved.
An independent fixed cell actually under-encloses in ordinary floating arithmetic,
so all verification/certification flags stay false. Source snapshots, RED/GREEN
failures, independent mathematical audit and parent replay are retained. Full
suites and code review are separate records, not implied by these focused passes.
No controller/default/public API, per-state scope or phase boundary changes.

The final repair-cycle review of source
`92ff3dceb217f6097bae8ccde6cf159d8daa8d14` is **rejected** (VAL-011): local
interpolation still cancels a representable tail, internal interpolation
underflow bypasses the outer trap, and the active basis conversion loses an
independently resolvable terminal gap. Parent replay reproduces all six additional
independent RED cases across both APIs. The 981-case CPU default suite passes
with actual exit zero and matching source/identities, but does not contain those
six assertions and cannot clear the review. At this rejection checkpoint the
current CUDA full-suite completion was not yet verified. Verdict, exact inputs,
pinned code and raw failures are archived; numerical implementation is paused
for an owner decision after the bounded repair loop. No artifact approval,
commit/push, acceptance-threshold relaxation, narrower scope or Phase A closure
is inferred. A1 remains open, and the current helper must not be used as an
approved validation oracle.

Subsequent full-suite completion readback confirms 981 passes and one performance
deselection in each CPU/CUDA process, with actual exit zero, exact frozen test
identities and 125 unchanged source inputs. These completed measurements remain
rejected-checkpoint evidence alongside the three blockers and six independent
RED failures. They do not reopen the paused repair loop or imply source approval,
numerical certification, a scope decision or A1 completion.

The owner subsequently authorized the
[arithmetic-contract-first repair plan](../validation/audits/a1-trajectory-defect-2026-11-11/arithmetic-contract.md)
and one new bounded implementation/review cycle. Independent technical review of
the contract passed before numerical changes. A new default regression file
preserves the independently designed 36-case matrix; parent RED reproduces
13 failures and 23 passes with unchanged comparisons. Ten are inaccurate-return
cases, three are masked intermediate-underflow policy failures, not gross final
accuracy failures. The shared repair and scoped source review pass; whole-project
CPU/CUDA suites each complete with 1,017 passes, one deselection and actual exit
zero. Completion-linkage correction and separate independent artifact re-review
pass. Prior
rejected source, full-suite successes and all RED evidence remain historical
records, not retroactive
approval. This authorization does not alter the per-state solver-order question,
the accepted mathematical scope or any A1/A2/A3/A4 completion gate.

The bounded implementation now passes 117 focused checks per CPU/CUDA-configured
process on frozen reference `bf3d4fa06026f4f2efedcf9d899c75ebca53cdbd`; its
126-input snapshot and 1,017 selected default identities are recorded. The final
independent review passed the approved range-checking contract and explicitly
classified the extra exact-input cancellation probe as SCI-003: a substantive
false-zero limitation, not underflow. Its condition number is about `2e20`;
checked fixed-order arithmetic does not control that cancellation. Computed zero
feedback/defect/cell widths are not proof of exact zeros or small exact residuals.
Independent conditioning/evaluation-error control remains required for scientific
acceptance. Both full CPU/CUDA process exits are confirmed, with all 126 inputs
unchanged and exact selected XML identities matched. Neither those outcomes nor
scoped code approval supply numerical certification, an arbitrary-input validation
oracle, a scope deferral or A1 completion. The initial artifact audit failed on a
stale historical-source label and missing completion-readback cross-links
(VAL-012); its original evidence is preserved separately. Documentation correction
and metadata-only linkage repair do not change the reviewed numerical source.
The shared evidence validator and repository-relative capture CLI pass 125
controlled integrity/portability tests, separately from the 1,017 default cases;
the failed original audit and RED/mutation evidence remain preserved. Corrected
artifact review passes with 313 indexed raw proof files and 43 independent
copied-record CLI controls. This closes only this increment's artifact-integrity
gate, not A1 or a broader Phase A boundary. Git delivery and remote CI remain
separate commit-bound observations.

## A2: History approximation and memory — open

Preparation authorized, 2026-11-11: the owner requested getting ready to start
A2. The [preparation contract](a2-compression-budget-protocol.md) is a draft for
independent mathematical and validation/resource review, not an implementation
approval or a scoped alternative to A2 completion. It proposes a private,
CPU-first GL/RL exact-reference budget tracer; work caps, case identities and
working-observation/FD policies must be reviewed and frozen before numerical
work. No selector, public API, fitter/default, dependency or numerical code has
changed. Broader acceptance requirements below remain unchanged.

The initial mathematical and validation/resource reviews both reject protocol
freeze: the exact finite inequality passes, but failure precedence, candidate
identity, exact fixtures/comparison mappings and bounded incomplete-evidence
semantics need correction (VAL-013). Their original failed verdicts and pinned
draft are preserved in the A2 preparation audit. The first corrected packet also
fails independent re-review: decision semantics and finite mathematics pass, but
full-reference/unsupported/invalid-request exports, receipt linkage, timing and
filesystem-ownership contracts remain inconsistent or incomplete (VAL-013).
Both revised failed verdicts, 94 indexed proof artifacts and the rejected packet
are preserved in a 124-entry archive before the second and final bounded
preparation correction. The 352-case parent replay is finite synthetic planning,
not production selector, fitting or AD evidence. All 125 numerical inputs and
index entries remain unchanged. Final independent review and explicit
implementation approval remain separate gates; the protocol is not yet frozen.

Final independent review passes repaired contract consistency but rejects freeze
readiness (`R-FINAL-RESOURCE-01`). The eight required float32 tau=1 cases cannot
launch under the present duplicated-index/atomic-coexistence policy: even
deliberately undercounted pairs of index skeletons require 614,748–617,184 bytes,
above the 524,288-byte reservation before process interiors or receipts. The
792,318-byte all-complete template is conservative; its 36,687-node excess is
not an unavoidable node lower bound because nonselected branches cannot all
complete. The byte obstruction nevertheless survives that correction. This is
an evidence-serialization/admission issue, not numerical or hardware failure.
All 51 independent proof artifacts and exact reviewed packet/status bytes are
preserved in a 66-entry final-review archive. Both correction cycles are used;
work is paused for an explicit owner decision. No cap, case/check, numerical
tolerance or acceptance requirement is changed, and implementation is not
authorized.

Owner authorization, 2026-11-12: the relayed reply “Okay, go ahead with your
recommendation.” approves one bounded preparation-only metadata byte/node budget
revision after valid export accounting and independent re-review. The scope is
recorded in `metadata-budget-revision-authorization.json` under the preparation
audit. Every case/check, scientific input, numerical tolerance, finite-math
definition and nonmetadata resource limit remains protected. This reopens only
the evidence-budget investigation; it does not approve protocol freeze,
numerical implementation, solver-order scope, deferrals or Phase B. Eight new
necessary byte-admission controls fail on the unchanged 512 KiB proposal before
revision (actual exit 1); these are static metadata controls, not fitter tests.

That authorized attempt stopped before any packet edit: its projections leave
372 conditional artifact/link obligations unclosed and omit required parts of
the full byte ledger. Independent assessment confirms an accounting-method
failure, not a new packet or numerical defect. No replacement budget is justified
or applied; all nine packet files and 125 numerical inputs remain unchanged.
Index-transaction bytes, per-record JSON nodes and the separate 16 MiB whole-
request ledger must be proved separately. A complete labelled hypothetical or
symbolic proof does not require actual JAX/fitter execution or a format redesign.
The failed attempt and independent assessment are preserved; further accounting
work is paused for an owner decision, with implementation and freeze unapproved.

Continuation authorization, 2026-11-12: “okay, let's pick up from where we left
off.” resumes one bounded linked-ledger completion and metadata-only revision
cycle, recorded in `metadata-ledger-continuation-authorization.json`. The previous
failed accounting, review and unchanged-cap outcome remain historical evidence,
not passing results. Complete conditional graph/link, per-record node,
index-transaction and live/cumulative whole-request accounting before choosing
replacement limits; then obtain fresh independent review. The 16 MiB overall
limit, all scientific requirements and numerical inputs remain protected.
Numerical implementation, protocol freeze and the other owner decisions remain
unapproved. Fresh baseline checks reproduce eight necessary admission failures;
23 documentation tests, Ruff, strict docs and whitespace checks pass separately.

The resumed author attempt also stops without a budget proposal. Its expanded
505-record symbolic representative still has 131 checked semantic obligations;
five of eleven negative controls survive (uncharged log/manifest/old-index/temp
bytes and unsupported-shape truncation). Four structural checks turn GREEN but
prove only necessary graph presence, not valid exports or sufficient accounting.
All 1,392 profile exercises are draft coverage, not successful numerical or
complete-export results. Author and advisory evidence are preserved separately
in 210- and 218-member archives. The advisory permits a bounded initial/required
qualifier-checkpoint/final index schedule; the author's every-array rewrite
estimate is not an unavoidable 16 MiB obstruction. Fresh independent assessment
reproduces the 131 unresolved conditions and all five surviving controls, and
confirms the failed-ledger stop. Its stronger negatives also accept wrong-shape
and cross-mode operands, invalid selected-with-unknown results, and removal of an
actual affine log term with bounds recomputed. All 102 review-indexed artifacts
and 124 archive members are verified and preserved. No budget revision is
delivered; limits remain unchanged because sufficient accounting is unproved,
not because current limits satisfy every mandatory case. The bounded attempt is
closed as failed and paused; no further writer cycle or cap change follows.

Single-trace authorization, 2026-11-12: “alright, go with your recommendation.”
approves replacing bulk expansion with one fixed conditional trace and an
independently checked byte equation first. `single-trace-authorization.json`
records the boundary: the existing labelled STATIC `REF-PASS-f64-GL` witness,
three samples, two supplied synthetic candidates and all 22 exact slots. No cap
change, broader case expansion, worker, fitter or freeze is authorized.
Twenty focused self-checks pass after seven captured vertical RED/GREEN steps;
an isolated code/data replay also passes twenty, and missing input is refused.
Builder equations and independent byte-state replay agree at 70,427 cumulative
bytes, 65,229 peak live bytes and a 19,070-byte chosen index transaction. These
are concrete sizes of this one hypothetical trace, not full-packet bounds.
Runtime timing, ownership and counters remain explicitly hypothetical, not
execution/security/arithmetic-budget evidence. Initial independent review rejects
the fixed checker for `FIXED-TRACE-ROUTING-01`: a re-signed binding record may
declare another binding ID while keeping its old filename. The positive exact
payloads and byte ledger are independently reproduced, but twenty passing
self-checks do not clear this missing routing predicate. Reviewer targeted tests
are 22 passed / 1 failed, actual exit 1; the parent separately reproduces the
invalid acceptance and required-route assertion failure. The rejected author and
review evidence are preserved in verified 220- and 457-member archives.
`single-trace-routing-correction-plan.json` bounds one third-context correction
inside this same target: enforce the already frozen path-to-ID equality, retain
the re-signed regression and obtain fresh review. This changes no contract or
scope and is not a new owner authorization. If the correction or fresh review
cannot close the target, stop for an owner decision; no second correction or
bulk restart follows this plan. Broader accounting and the known eight-case
admission obstruction remain open, with every packet/scientific requirement
unchanged.

The one routing correction now adds exactly one guard to the existing bindings;
the original twenty tests and all input/schema/protocol bytes are unchanged.
Six re-signed duplicate/fresh-ID controls fail behaviorally on the rejected
source, then all 26 tests pass on the corrected source and isolated copies. The
parent independently replays 26 tests and the portable positive artifact; every
one of its 38 emitted byte strings remains identical. All 235 correction-index
entries and 238 archive members are verified, with three symlinks archived as
labelled readlink data, not recreated links. Fresh independent re-review passes
with no security concerns or logic errors in this fixed scope. It independently
re-signs all six routing controls, confirms rejection for the exact path-to-ID
predicate, removes the sole guard to reproduce old acceptance, and reproduces
the unchanged positive exact payloads and physical byte-state ledger. The parent
verifies all 373 review-indexed entries (14,989,426 bytes), final seal/binding
receipts and all 390 archived members. Original rejection and capture limits
remain preserved. `single-trace-independent-review-passed.json` and
`single-trace-passed-review-preservation.json` bind this result; final disposition
is in `single-trace-outcome.json`.

This passes only the one-case source/finite-proof gate. It neither proves broader
accounting nor clears A2/Phase A, changes any cap, approves numerical
implementation or freezes the protocol. All hypothetical execution/security/
counter qualifications remain. No correction cycles remain, and this result
authorizes neither bulk expansion nor the next numerical phase.

Publication hold recorded 2026-11-12, `FIXED-TRACE-ARTIFACT-PRIVACY-01`: the final packaging
scan finds eight native Telegram receipt copies in the initial rejected-review
archive, outside the fixed trace's input closure. Initial capture copied all
top-level Python/JSON files; historical replay later copied that broader per-run
manifest despite the outer author index excluding private receipt files.
Hash-consistent replay/archival did not enforce the privacy boundary. Nothing was
committed or pushed, and the investigation prints no identifier values.
The original 457-member archive is byte-identically preserved in owner-only
private storage outside the repository and its untracked repository copy is
removed; it is not silently replaced or rewritten. Old archive hashes/paths must
be read with the explicit quarantine-location exception in
`single-trace-privacy-hold.json`. The other four current trace archives contain
no matching identifier fields in the bounded structured-JSON scan; this is not a
universal privacy guarantee. Source/finite-proof passes remain valid, but the
completion notice was premature and publication/artifact completion is on hold.
With zero correction cycles left, stop for an owner decision on a labelled
sanitized derivative and separate privacy/integrity review. No further correction
or broader phase is started.

Owner-authorized privacy remediation, 2026-11-13: the
[labelled sanitized derivative](../validation/audits/a2-preparation-2026-11-11/privacy-remediation-2026-11-13/README.md)
omits exactly the eight receipt members and retains all other 449 member bodies
unchanged. The private original, historical failed verdicts and original
manifests remain unchanged; the derivative does not promise complete historical
replay closure. Independent privacy/integrity review passes with no blocking
findings, 23 fresh packaging checks and 16 rejected CLI tamper controls. The
parent verifies all 198 indexed review files and 199 archived members, and
rechecks their privacy before updating this status. The publication hold is
resolved **only for this explicitly identified sanitized derivative**, not for
the unsanitized original or a blanket A2 evidence/release claim. Final QA and the
bounded status delta are recorded in that remediation package. No numerical
source, fixed-trace checker, scientific requirement, resource limit or prior
evidence is changed. Broader A2 accounting/admission remains blocked; no further
accounting increment, numerical implementation, protocol freeze, commit or push
is authorized by this remediation.

Owner-authorized next accounting increment, 2026-11-13: “Yes, proceed as planned.”
approves one additional static case, recorded in
`nonqualifier-trace-2026-11-13/authorization.json` under the preparation audit.
The selected existing witness is `nonqualify-REF-FAIL-f64-GL`: three samples,
one supplied synthetic candidate and all 13 required exact slots. It contrasts
the reviewed selected-case trace with a completed no-qualifier outcome. The
selected-case artifact and its unchanged 26 tests remain regression controls.
Close exact payload/qualification/linkage checks and the full conditional byte
ledger, then obtain fresh independent review. This is preparation only: no new
fit/worker/AD observations, bulk profiles, resource-limit revision, schema change,
protocol freeze, numerical implementation or commit/push is authorized. Preserve
any blocking review and stop rather than reopening previous exhausted loops.

Partial author checkpoint: the original 26 tests pass at baseline and on the
initial extended checker; the new positive no-qualifier case also passes its
first RED/GREEN cycle. A re-signed missing-bound control then exposes a `TypeError`
before semantic slot validation (VAL-015). The minimal validation-order patch is
unexecuted: `07-missing-bound-green` was stopped by a one-shot tool-approval gate,
with no child process launched. The blocked action was not retried or rerouted.
The parent verifies 132 indexed files and preserves all 133 archive members in
the [partial checkpoint](../validation/audits/a2-preparation-2026-11-11/nonqualifier-trace-2026-11-13/README.md).
Independent physical-byte replay, the remaining semantic controls, portable
replay and independent review are unfinished. This is an execution-approval pause,
not completed accounting or a failed independent verdict. No current-source
GREEN, cap revision or broader acceptance is claimed.

Approved execution resumption: the owner replied “approved.” The original
unlaunched attempt remains historical. A newly captured copied-input run tests
the exact preserved ordering patch and passes both focused tests with actual
exit zero and unchanged input hashes; `approved-resumption.json` binds its
13-member evidence archive. The same bounded increment resumes; broader checks
and independent review are still required. This is not a new repair-cycle or
scope authorization.

The additional no-qualifier trace now passes scoped independent review. The
reviewer reproduces 59 selected tests, rejects 33 independent posthoc controls,
reconstructs all 13 exact slots and verifies the full physical byte ledger and
the original selected case's canonical trace/all 38 bodies. The parent verifies
329 indexed proof files, archives all 330 members and independently replays the
exact/physical checks and all 33 controls. The reviewer completion message's
HTTP 400 is a delivery failure, not a failed mathematical check: the saved verdict
and proof were recovered and verified. Earlier incomplete reviews and approval
pauses remain historical, not relabeled successful executions.

For this one hypothetical schedule, final retained bytes are 40,245, peak live
bytes 41,328, cumulative writes 45,148 and the chosen index transaction 13,098;
old and temporary index charges are 4,903 and 6,258 respectively. Maximum parsed
record size is 720 nodes/depth 9. These are finite trace sizes, not general
admission, working-memory, runtime or operation-count guarantees. The scoped
VAL-015 ordering repair is verified, with inherited archived-source lint findings
still disclosed. Final QA and status-delta evidence live in the additional-trace
package. All numerical sources, original packet/schema/fixtures, scientific
criteria, caps and earlier evidence remain unchanged. The eight FIT-REAL
admission blockers and broader A2 accounting remain open; no numerical
implementation, protocol freeze, commit/push or phase closure follows.

The owner has separately authorized the existing `CAP-after-qualifier-f64-GL`
static case as the next bounded increment. Its frozen fault checkpoint retains
candidate 0's known qualifier and candidate 1's nine unknown quantities; all 22
inventory dispositions remain explicit. Parent exact reconstruction verifies the
13 available quantities and zero bounds at tau zero without calculating the
unavailable candidate's quantities. The exhaustive policy requires indeterminate
selection with a null selected ID. This is an injected hypothetical checkpoint,
not measured primitive-cap exhaustion or an actual killed worker. The test-first
checker extension, conditional byte ledger and fresh independent review are
pending in `cap-after-qualifier-trace-2026-11-13`. Both reviewed cases and all
prior evidence are protected regressions. No cap, schema, numerical-source or
scientific-scope changes are authorized, and a blocking independent finding must
be preserved and returned to the owner rather than repaired automatically.

The cap-after-qualifier author self-checks and parent copied-input replay now
pass. All 59 previous selections run against the new checker with unchanged
tests and both prior canonical traces/all 38 and 27 bodies intact. The new-case
selection passes 66 tests and a separate no-qualifier artifact equality check.
The parent verifies 1,706 indexed files, 1,707 archive members and 56 raw run
records, and replays all 13 known quantities and 28 physical bodies. The finite
schedule retains 51,898 bytes, peaks at 55,421 and writes 59,388 cumulatively;
its index transaction is 18,276 bytes. Ten guard mutants are killed, while one
inventory-guard attempt remains inconclusive because another guard rejects the
same input. The scoped counter-placeholder guard is recorded as VAL-016.
Fresh independent review remains pending; this is not approval of that guard,
the new trace, broader resource sufficiency or any numerical implementation.

Independent review has now completed and **rejected** this added trace's
fixed-authority authentication (`CAP-AUTH-TYPE-01`, VAL-017). Python structural
equality accepts boolean/integer and float/integer substitutions in the
supposedly exact case/scalar identity. Three preserved controls pass both the
authenticator and trace checker despite changed canonical authority bytes; the
parent reproduces them with actual detection exit 2. The original trace's known
mathematics, byte ledger and earlier artifact regressions still reproduce, but
cannot waive this contract violation. The inventory-mutant coverage question
was separately resolved by a new isolated control, not by removing a guard.
All 532 reviewer-indexed files and 533 archive members are preserved. No source
repair follows automatically: this increment is blocked pending a separately
approved narrow repair and fresh review. Numerical sources, criteria, caps and
the broader A2 boundary remain unchanged.

The owner subsequently approved one bounded VAL-017 correction: type-preserving
fixed-authority comparisons, separate counterexample regressions, unchanged
legitimate artifacts and fresh independent review. Only a derivative of the
private `frozen_case.py` may change; the main static checker, original tests,
fixtures, criteria, caps and numerical source remain frozen. All three valid
canonical artifacts and their emitted bodies must stay byte-identical. The
rejected checkpoint is preserved under its failed outcome. Correction is in
progress in `cap-after-qualifier-trace-2026-11-13/identity-repair-2026-11-13`;
no repair success, re-review approval or wider A2 acceptance is claimed yet.

The corrected helper now passes author self-verification and parent readback:
27 typed-identity tests, all 126 existing selections and eight posthoc defensive
checks pass. Case/bundle/scalar predicate reversions are caught. The parent
verifies 1,632 indexed files, 1,633 archive members and 42 captured runs, then
fresh-replays the original counterexamples and all three complete valid
artifacts. Their canonical traces and all 38/27/28 bodies are unchanged; the
main checker and 930 protected baseline files remain frozen. Only the helper's
canonical-JSON identity comparison changed. At this author/parent checkpoint,
independent re-review was still required; the earlier failed verdict and blocked
records stay historical failures.

**Current bounded disposition:** fresh independent re-review and parent readback
now pass. The reviewer reproduces 230 checks (27 identity, 126 earlier, eight
defensive and 69 new independent controls); the parent verifies all 720 indexed
review files/721 archive members and independently reruns the 69 controls.
The three complete canonical traces and all 38/27/28 bodies remain unchanged.
VAL-017 is closed for the predicate-only local JSON identity repair, and the
static cap-after-qualifier increment, including its scoped VAL-016 counter guard,
is complete. Final QA/status-delta evidence is in the nested identity-repair
package. This does not clear the eight FIT-REAL admission blockers, wider A2
accounting/acceptance, numerical implementation, protocol freeze or phase closure.

Required work:

1. Add tolerance-driven SOE fitting/selection or an explicit scoped alternative
   with a recorded failure when a requested tolerance cannot be attained.
2. Validate approximation bounds against full history within each declared
   signal/kernel scope. Fixed poles or a small fitted residual are not a
   universal relative-error guarantee.
3. Measure CPU/CUDA forward and reverse-mode execution, compilation/runtime and
   peak memory across length/window/pole/state-size sweeps. Separate fitting
   cost, application cost and reverse-mode saved state.

Acceptance:

- Use independent fitted-sum/kernel checks and gradient validation.
- Record actual devices, dtypes, synchronization, repeats and failed/resource-
  limited cases. Compiler buffer accounting alone is not measured peak memory.
- Do not infer solver-history compression or vector SOE support from operator
  results. Neither is implemented in the current supported combination matrix.

Selection design constraints: start with absolute discrete-compression budgets,
record every candidate bound and failure, and choose the smallest supplied
candidate that actually meets its bound. Do not assume error is monotone in pole
count or silently fall back to full history. Hold the discrete configuration
fixed during differentiation while recomputing order-dependent fitted weights.
These are proposed requirements, not a new public selection API or a certificate
of discretization/roundoff error. SOE convergence evidence requires fits that
improve with refinement, not just honest failure at an unattainable budget.

Resource protocols must distinguish fitting, recursive application and gradients.
Use isolated workers and synchronized samples; preserve raw memory observations
as well as compiler accounting. Treat GPU memory polling as a lower-bound
observation unless a reliable allocator high-water mark or trace is available.

## A3: Supported combinations and validation coverage — partially addressed

Required work:

1. Audit analytic, convergence, gradient and JIT evidence for every claimed
   operator/solver/history path, not every imaginable cross-product.
2. Reconcile per-state order and nonuniform-grid expectations with actual APIs.
   The current boundary is per-state uniform operators and scalar-order solvers;
   changing that boundary or approving deferrals requires an explicit decision.
3. Review alpha-gradient stability claims against evidence and public tiers;
   selected validation does not automatically promote an entire API to stable.

Acceptance:

- Keep the [capability matrix](../validation/capabilities.md) accurate and link
  each claim to tests or an archived protocol/report.
- Review all explicit scope deferrals. No deferral is approved in this ledger.
- Preserve visible limitations: traced value validation, whole-result JIT return,
  padded adaptive outputs, uncompressed solver history and unsupported combinations.

Further coverage targets include history-path JVP/VJP consistency, nonuniform
solver order gradients, padding/broadcasting-sensitive FFT cases and
scheme-specific convergence. Fixed-window and fixed-fit approximations need
their own truncated/approximated targets; full-history convergence cannot simply
be assigned to them.

## A4: Reviewed candidate and reproducibility — open

Required work:

1. Independently review the complete intended change set and address findings.
2. Establish reviewed, identifiable source and a release/candidate scope.
3. Execute the documented CPU Python matrix and selected GPU checks; capture
   numerical reports, separate performance results and benchmark context.
4. Build distributions and verify the wheel in isolation outside the checkout.
5. Verify actual remote CI for that reviewed revision; a configured workflow or
   an older green run is insufficient.
6. Choose candidate version metadata separately from a release/tag decision;
   synchronize metadata only after that choice, without inventing a release date.

Acceptance:

- Follow a copied candidate checklist; leave the reusable template unchecked.
- Preserve source identity, configuration, runtime context and raw outputs.
- Obtain permission before commits, pushes, tags or publication. Commits/pushes
  for CI and a release/tag/publication decision are separate authorizations.
- Mark Phase A complete only after the roadmap exit criteria and reviewed
  deferrals, if any, have genuinely been satisfied. A subset candidate release
  is not Phase A completion.

## Execution order

1. Resolve scope and repository-workflow decisions; preserve the verified baseline.
2. Design the A1 adaptive correction and A2 selection/error contracts before adding
   public APIs; implement in independently tested increments.
3. Extend the validation/precision/gradient matrix and collect memory/scaling
   evidence against those fixed claims.
4. Independent review, clean-candidate checks, distribution and remote CI evidence.
5. Review the closure dossier; only then update the phase status.
