# A1: supplied-trajectory Volterra-defect diagnostic

Status: scoped source/math review **passed** for the corrected private range-
checking implementation. Current CPU/CUDA full suites each have 1,017 passes,
one deselection and confirmed exit zero. Completion-linkage correction is
implemented and exercised; corrected independent artifact review **passed**.
This is not cancellation safety, a validation oracle for arbitrary signed
reductions, verified arithmetic, scientific residual acceptance or A1 completion.
See the substantive false-zero limitation below and SCI-003 in the bug tracker.

The preceding source `92ff3dceb217f6097bae8ccde6cf159d8daa8d14` was rejected
for three arithmetic defects, reproduced as six independent RED cases across
both APIs. The owner then authorized the
[contract-first plan and one new bounded cycle](arithmetic-contract.md); both
technical contract and corrected-source reviews passed separately. Neither
changes the preceding rejection into approval. Artifact clearance does not attest
commit, push or remote CI; use Git and commit-bound CI for those delivery claims.
No scope deferral, numerical certificate or Phase A closure is implied. The
[preceding design](../a1-defect-design-2026-11-11/README.md) and its
original manifests, failed fixture and acceptance-false review remain unchanged.

`final-code-review-rejected.json` and `final-code-review-rejected.tar.gz` preserve
the verdict, exact inputs, pinned source and six RED tests. Parent execution
independently reproduces all six failures (exit one); its raw XML is retained
byte-for-byte in `final-blocking-parent.xml.gz`. All 33 indexed review files are
hash/size checked. A passing default suite does not clear this rejection.

Historical rejected-92ff readback: both default CPU/CUDA full processes pass 981
tests with one performance deselection and confirmed actual exit zero. Both XML
identity sets match the frozen collection and all 125 declared source inputs
matched at that checkpoint. `rejected-checkpoint-results.json` and the original
readbacks/executions in `pre-arithmetic-checkpoint.tar.gz` retain those outcomes
separately from the failed code
review. The six additional independent RED cases are outside that default
selection; both full-suite successes remain evidence for rejected source, not
approval. The rejection-time pause and then-unanswered decision are historical
facts in that result record; later plan authorization does not change them.

Those rejected-source failures were local interpolation cancellation, interpolation
underflow inside `np.interp` that bypasses the outer error trap, and loss of an
independently resolvable right-end gap in the active basis conversion. The
old implementation and its failed guards remain preserved, not retrospectively
described as successful validation.

## Scoped reviewed candidate and remaining verification

The new four-file numerical/test/fixture reference is
`bf3d4fa06026f4f2efedcf9d899c75ebca53cdbd`, with 126 frozen inputs and 1,017
selected default tests. The fresh implementation changes only the benchmark
helper relative to the parent's RED run: checked independent-distance PL
reconstruction, an independent active-panel endpoint gap, and inspectable
float64 products/accumulations in both matrix and forcing paths. The 117 focused
tests pass in both CPU/CUDA-configured processes, actual exits zero; this remains
host diagnostic arithmetic. `arithmetic-repair-green.tar.gz` retains exact
implementation and raw GREEN evidence. `arithmetic-code-review.json` and its
archive record the separate scoped source/math pass, with all 38 indexed raw
review files hash/size checked. The two new 1,017-case full-suite processes are
recorded in `execution-plan.json`. Both actual process exits are independently
confirmed in `process-completion.json`; raw XML identity sets match the selected
collection and all 126 source inputs remain unchanged. These actual observations
do not by themselves clear the separate artifact-integrity finding below.
The initial documentation/evidence audit **failed**, separately from the source
pass: it found a stale current-source label for the historical 92ff checkpoint
and incomplete readback-to-XML/source/process linkage. Its 41 indexed raw proof
files and original frozen inputs remain in `document-integrity-audit-failed.tar.gz`
with `document-integrity-audit-preservation.json`; these are not candidate test
failures. The label is corrected below. Shared standard-library completion checks
now bind XML hashes/counts/exact identities, every input hash, canonical metadata
types and source/process/exit identities before finalization. Fresh actual
readbacks include preserved exit bytes and process-observation hashes. Record
consistency is not observation authentication or scientific acceptance.

`completion-integrity-repair.tar.gz` and its index preserve the initial strict
four-failure metadata RED, expanded original-script replay, corrected 124-case
GREEN, all 19 guard-removal mutation results/raw XML and the parent's additional
portability RED. The portable capture CLI is now included in this directory;
the final 125 controlled metadata/portability tests pass. These controls are
outside the default numerical selection and are not added to its 1,017 count.
The original no-op XML-mutation test and its correction remain separate records.
`corrected-artifact-review.json` and its archive record the separate passed
artifact review (VAL-012): 313 indexed proof files, 125 controlled tests and 43
independent copied-record CLI controls. The original 85-file reviewed manifest
is preserved in `reviewed-artifact-manifest.json`. Subsequent status-only edits,
new review artifacts and refreshed delivery records are identified separately;
they are not retroactively assigned that original reviewed digest.

An additional parent conditioning probe is a substantive limitation: constant
represented states `[1e20,1,-1e20]` and a feedback row of
ones produce zero computed defect and a zero cell range, versus component defect
about `-1.1283791670955126` from exact-input high-precision arithmetic. This is
fixed-order cancellation, not underflow; it is outside the 117 focused cases.
The reviewer classified it as nonblocking **only under the approved contract**:
the sum is exactly one while its absolute product sum is about `2e20`, giving
condition number about `2e20` and condition-number times binary64 unit roundoff
about 22,204.46. Losing one is consistent with ordinary floating addition, not
an underflow event. Permuting the unchanged components restores the contribution.
This classification does not rely on false certification flags and does not make
the result scientifically acceptable. Checked range arithmetic is not cancellation-
safe: a computed zero coefficient, zero defect or zero-width cell is not proof
of an exact algebraic zero or small exact residual. Independent conditioning and
evaluation-error control are required before scientific acceptance. A cancellation-
robust redesign is outside this bounded cycle and requires separate authorization.

## Implemented increment

`benchmarks/numerical/_volterra_defect.py` reconstructs the continuous
piecewise-linear trajectory through the actual stored states. For scalar
`0 < alpha < 1` and `f(t,Y)=A Y + sum c_beta*(t-t0)**beta`, `beta >= 0`, it evaluates
`Y-y0-I**alpha(f(.,Y))` using local interpolation and positive PL panel-basis
integrals, not cancellation-prone global slope jumps. It retains every old panel,
the original kernel terminal and forcing clock, correct matrix orientation and
the PECE nodal Volterra mismatch. It does not integrate an interpolant of stored
RHS values while pretending to interpolate stored states. The
[reviewed arithmetic contract](arithmetic-contract.md) describes its independent
active-panel gap evaluation and range-checking boundary. The original
[panel-basis derivation](panel-basis.md) is preserved unchanged as historical
real-arithmetic evidence; its subtractive `1-f` conversion is not the corrected
floating evaluation recipe.

The cell helper splits the verification mesh at every stored knot and evaluates
positive-basis weight extrema and signed forcing-power ranges. It retains the
active left weight's interior maximum, not only endpoint samples. Its
`model_lower`/`model_upper` arrays
are floating diagnostics of an exact-arithmetic range construction, NOT verified
numerical enclosures. `max_work` caps cell-term-component evaluations; exhaustion
is explicit and never truncates history. This is not a measured memory/time cap.
Inputs must be finite real float32/float64 arrays, state shape `(time,state_dim)`;
orders remain scalar. All diagnostic RHS arithmetic widens represented inputs to
host float64 first. Float32-input checks are not claims of float32 diagnostic
arithmetic. The two PECE compatibility cases additionally exercise actual solver
outputs with independently scoped float32/float64 JAX settings. The selected
time-power underflow cases raise explicitly, even when a rearranged final
expression could be representable. This conservative failure is not computation
of the rescued value or an approved scope deferral. The corrected reconstruction
does not call `np.interp`; checked products precede accumulation in both APIs.
These expression-level safeguards do not detect every internal C-library event
or bound special-function/summation/cancellation errors. No global precision
promotion is used, and SCI-003 remains a substantive limitation.

Both structured results carry provenance and always return
`arithmetic_verified=False` and `total_numerical_certified=False`. There is no
acceptance decision, feedback propagation, nonlinear domain proof, stiff-case
certificate, JAX tracing or gradient contract for these host helpers. No library
runtime, public export, dependency, lockfile, solver default or tolerance changed.

## Evidence and preserved failures

Technical contract clearance and corrected-source approval are separate passes.
`arithmetic-contract-review.tar.gz` preserves the exact reviewed draft and raw
proof/probe records. `arithmetic-test-design.tar.gz` retains both independent
test-design runs; the final 36-case baseline has 13 failures and 23 passes.
Ten failures concern inaccurate canonical/signed results; three concern masked
intermediate-underflow policy, not gross final-value accuracy. The default-suite
port in `tests/unit/test_volterra_arithmetic.py` independently reproduces this
RED selection with unchanged comparison logic/tolerances and 126 source inputs
captured before/after. `arithmetic-regression-red.tar.gz` preserves its raw
evidence and AST-checked non-numerical port adaptations. The RED evidence itself
implies neither corrected-source approval nor a new full-suite outcome; the
separate source-review verdict and actual full-suite records govern those claims.

- Five vertical features have separately captured pre-run test sources, actual
  RED XML and GREEN XML: reconstruction, weak-power forcing, input/arithmetic
  guards, continuum-cell diagnostics and explicit work exhaustion.
  `development-evidence.tar.gz` preserves those bytes; `development-index.json`
  records hashes/counts. Later passing compatibility assertions are additional
  coverage, not retrospectively described as RED-tested features.
- The initial compatibility run has 53 passes and two test-authoring failures:
  this JAX version has `jax.enable_x64`, not `jax.experimental.enable_x64`.
  The failed XML is retained. The repaired pre-VAL-008 targeted selections pass
  55 cases on each CPU/CUDA process. The diagnostic arithmetic is host-only on
  both; this is not a GPU diagnostic-kernel validation claim.
- Six independently frozen case inputs are copied byte-for-byte in
  `frozen-fixtures.json`. The original failed old-panel discriminator is retained
  inside the independent audit, not erased. Panelwise 80-digit quadrature is
  independent of the implementation's closed-form basis weights.
- `independent-math-audit.tar.gz` preserves the mathematical audit and its raw
  outputs, not an implementation review. It contains 38 quadrature targets,
  336 verification cells and 3,024 sampled corroborations. Samples and
  high-precision agreement do not prove a numerical enclosure. Parent replay in
  a separate copy reproduces five output/fixture files byte-for-byte; see
  `parent-math-replay.json`.
- The frozen half-order quadratic PL trajectory has nodal defect below `1e-15`
  but interior defect above `0.09`. Its newest-only and restarted history
  formulas disagree with the complete original-history defect. Nodal sampling
  or shrinking only the latest panel cannot replace continuum/full-history work.
- A fixed counterexample is retained as a limitation regression: `Y=0`, `A=0`,
  `q=1`, `alpha=1/2`, on `[nextafter(1,0),1]`. The ordinary float64 cell misses
  the true endpoint residual by approximately `1.53e-17`. This is not repaired
  by arbitrary padding or hidden precision promotion. It blocks certification,
  not the explicitly unverified diagnostic.

A later extreme-argument probe finds VAL-008: subtracting large log-gammas
returns a forcing ratio of one instead of approximately `1e-150`. Nine new
regressions fail; the shared reciprocal-Pochhammer repair fixes both pointwise
and cell paths. That Gamma-only checkpoint passes 64 targeted cases per CPU/CUDA
process, without changing arithmetic precision or certification flags.
`gamma-repair-evidence.tar.gz` and its index preserve this sixth RED/GREEN cycle.
The first full-suite launches are killed for source replacement, not counted as
passes/failures; `superseded-runs.json` and the before-gamma source/collection/
static-scan/focused XML retain their identities.

`initial-review.json` preserves the independently failed code review of tree
`6e7a156f025e08b31a314c5bda59183d6a96bf32`, including its blocking Gamma-ratio
finding; `initial-code-review.tar.gz` retains the raw checks. Its authoritative
verdict is `initial-review-verdict.json`; an explicitly labelled unexecuted
pre-discovery draft inside that archive is not a review result. Parent replay
loads trusted old/corrected immutable-tree bytes matched to the two source
archives and reproduces the error and repair in both paths; see
`parent-gamma-replay.json`. Corrected-tree review is a separate record, not an
amendment changing the original failure into a pass.

The Gamma-only tree `8dcbfafd1a5373088e75437551692a85fd7f7b77` passes both
964-test full suites with confirmed zero exits and matching 125-input/test
identities, but its independent review still fails on VAL-009/VAL-010: steep-mesh
hinges erase a nonzero state/history, and an intermediate time power silently
underflows before large finite coefficient scaling. This is preserved in
`gamma-review-failed.json` and `gamma-code-review-failed.tar.gz`, together with
the explicitly historical `*-before-panel-*` / `*-before-panel-repair*` records.
Full unit-test passes do not clear those defects or the failed review.

The historical panel-repair increment used positive panel-basis integrals/local
interpolation and selected expression-level underflow traps in both point/cell
paths. Five regressions go RED, then the
69-case selection goes GREEN; additional compatibility coverage brings the
focused count to 81 per CPU/CUDA process. `panel-repair-evidence.tar.gz` and its
index retain that seventh cycle and a rejected preimplementation near-one
special-function probe. That historical panel-repair source was frozen at tree
`92ff3dceb217f6097bae8ccde6cf159d8daa8d14` and rejected by final source review;
its 981-case full-suite successes do not clear the rejection. It is distinct
from both the earlier 964-test checkpoint and the current reviewed bf3 candidate.

`source-snapshot.json` and its archive freeze the declared numerical/test input
set before the full suites. This scope is tracked `src`, `tests`, `benchmarks`,
project/lock inputs and the new frozen fixture, not every documentation file.
`collection.txt` records exact selected test identities. Full-suite process exits,
XML identity checks, final documentation checks and implementation review are
recorded separately when complete; targeted passes alone do not establish them.
`baseline-ci-readback.json` records successful remote CI for the preceding
`74d8f08` checkpoint, not remote validation of this new implementation.

## Reproduction and remaining gates

The evidence tools are standard-library scripts shipped with this audit:

```bash
uv run --frozen --extra dev python docs/validation/audits/a1-trajectory-defect-2026-11-11/verify.py
uv run --frozen --extra dev python docs/validation/audits/a1-trajectory-defect-2026-11-11/verify.py --current-source
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --extra dev python -m pytest docs/validation/audits/a1-trajectory-defect-2026-11-11/test_completion_integrity.py -q -p no:cacheprovider
```

Default replay validates preserved source/exit bytes and explicit completion
observations without claiming current local source or ephemeral exit-file checks.
`--current-source` additionally requires the actual local inputs and original
local exit records. The output identifies the mode. `capture_completion.py`
regenerates backend readbacks/final records only after independent actual process
completion; it never infers completion from a test summary or creates process
observations. The controlled tests use copied, explicitly labelled metadata and
the repository capture CLI, not an unshipped scratch-directory dependency.

Run the focused selection from the repository root:

```bash
JAX_PLATFORMS=cpu uv run --frozen --extra dev python -m pytest tests/unit/test_volterra_defect.py tests/unit/test_volterra_arithmetic.py
UV_PROJECT_ENVIRONMENT="$HOME/.local/share/hpfracc/venv-cuda13" JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --frozen --extra dev --extra cuda13 python -m pytest tests/unit/test_volterra_defect.py tests/unit/test_volterra_arithmetic.py
```

Extract the independent audit into a fresh scratch directory to rerun its three
probes with locked uv. Do not regenerate frozen fixtures as part of a check.

A1 acceptance remains false. Verified gamma/power/coefficient/input/RHS/output
arithmetic, componentwise feedback propagation and tail budgets, nonlinear tube
closure, broader stiff domains, a frozen tolerance/work/reference matrix and
reviewed opt-in controller/JAX/gradient integration remain open. The per-state
solver-order scope question is separate and unanswered. A2/A3/A4 are not closed.
