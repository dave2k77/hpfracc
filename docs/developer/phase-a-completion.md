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
