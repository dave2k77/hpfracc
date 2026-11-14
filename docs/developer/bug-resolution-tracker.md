# Bug-resolution and learning tracker

Status: Living engineering record; not a release or Phase A completion certificate.

Use this record while investigating and changing HPFRACC. Keep symptoms,
scientific impact, evidence, cause, intervention and verification separate. A
passing regression closes only its declared scope. Preserve failed measurements
and historical evidence instead of rewriting them after a fix.

## Entry policy

- Assign a stable ID and classify the entry as a defect, environment issue,
  validation gap or research limitation. Missing future functionality is not
  automatically a defect.
- Record discovery context, affected paths, observable failure and impact before
  editing. Label untested hypotheses as hypotheses.
- Identify the root cause using a reproducer or independent reference; describe
  the fix and why it addresses that cause.
- Record commands, actual outcomes, source/environment context, regression tests
  and remaining limits. Distinguish local verification from remote CI and release
  sign-off. Add commit/PR references when available.
- Finish with a prevention lesson that is actionable for later development.
  Update an entry when new evidence changes its status; do not erase prior facts.
- Keep credentials, private inputs and local permission/configuration files out
  of this tracker and out of commits.

## Index

| ID | Classification | Issue | Status |
| --- | --- | --- | --- |
| ENV-001 | Environment | Local Python/tooling differed from CPU CI validation context | Resolved locally |
| ENV-002 | Environment / reproducibility | CUDA plugins were outside the project lock and removable by sync | Resolved for the declared CUDA extra |
| NUM-001 | Numerical defect | Reduced-precision GPU dense-history contractions | Fixed; scoped regressions pass |
| NUM-002 | Numerical defect | Reduced-precision legacy SOE fitting contractions | Fixed; scoped regressions pass |
| VAL-001 | Validation gap | FFT/full comparison conflated two independent rounding errors | Fixed for the selected regression |
| ENV-003 | Environment | CUDA-only backend disabled CPU debug callbacks | Resolved by backend configuration |
| NUM-003 | Numerical defect | Float32 Gamma normalization biased an implicit corrector equation | Fixed; independent residual gate passes |
| NUM-004 | Numerical precision defect | Direct float32 history-weight cancellation failed an independent residual gate | Fixed for the selected scope; full local CPU/CUDA suites pass |
| NUM-005 | Numerical precision defect | CPU float32 history accumulation can hide an independent residual failure | Selected-case repair passes full CPU/CUDA suites; review/delivery separate |
| API-001 | Metadata contract defect | Strong array orders prevent plain JSON export of SolverInfo | Host-scope fix; full CPU/CUDA verification passes, review recorded separately |
| VAL-002 | Validation gap | No dedicated FFT order-gradient finite-difference matrix | Selected coverage verified after VAL-004 correction |
| DOC-001 | Documentation | Nonuniform solver missing from API stability table | Fixed |
| SCI-001 | Research limitation | General adaptive tolerance indicator is not a calibrated error bound | Open |
| SCI-002 | Research / validation gap | History selection and measured reverse/peak-memory evidence incomplete | Open |
| SCI-003 | Research / arithmetic limitation | Cancellation-dominated signed reductions can return false-zero private diagnostics | Open; nonblocking only for the scoped range-checking contract, not scientific residual acceptance |
| VAL-003 | Validation semantics | Measurement-integrity checks are insufficient scientific acceptance | Implemented for selected implicit cases; broader gates open |
| VAL-004 | Validation defect | x64-enabled fixture masked nominal float32 FFT execution | Fixed; corrected CPU/CUDA matrices pass |
| VAL-005 | Validation defect caught before commit | Numeric value equality weakened configuration identity | Fixed; strict canonical types and regressions |
| VAL-006 | Evidence documentation | Current citations retained superseded CUDA measurements | Corrected; ten report-to-documentation checks |
| VAL-007 | Evidence bookkeeping / reproducibility scope | Exact isolated/full-suite CUDA residual equality was assumed | Bookkeeping corrected; emitter variation reproduced, selection cause unresolved |
| VAL-008 | Private validation defect | Large finite forcing powers cancelled the defect oracle's gamma ratio | Shared ratio repair; focused CPU/CUDA checks pass, full verification separate |
| VAL-009 | Private validation defect | Global slope jumps erased a graded PL trajectory and history | Selected repair reviewed; later arithmetic findings tracked in VAL-011 |
| VAL-010 | Private validation defect | Intermediate time-power underflow hid a representable scaled contribution | Selected power path rejects; sibling guard repairs reviewed in VAL-011 |
| VAL-011 | Private validation defect | Local interpolation and active basis arithmetic erase representable contributions | Scoped source review and 1,017-case CPU/CUDA suites pass; separate artifact review passed, SCI-003 remains open |
| VAL-012 | Delivery evidence integrity | Stale historical source label and incomplete completion-readback binding | Corrected; 125 integrity/portability controls and independent artifact re-review pass |
| VAL-013 | Preparation specification | A2 failure precedence, identity, fixtures and resource evidence not freeze-ready | Three named static traces reviewed; broader accounting/admission gates remain open |
| VAL-014 | Artifact privacy | Historical replay reintroduced private delivery receipts into an evidence archive | Sanitized derivative independently approved; unchanged original remains private, no blanket publication clearance |
| VAL-015 | Private static-checker validation | Missing exact bound is dereferenced before semantic slot validation | Scoped ordering repair and additional trace independently reviewed; broader A2 gates open |
| VAL-016 | Private static-checker metadata | Fixed unmeasured counter placeholders could be changed without refusal | Scoped guard verified within the completed static increment; broader A2 gates open |
| VAL-017 | Private static-checker identity | Python structural equality accepts type-altered fixed authority | Scoped JSON-identity correction independently approved and parent-verified |
| ENV-004 | Environment / delivery | GitHub write authentication was unavailable | Resolved; authenticated push verified |

## ENV-001 — Local development environment and CI context

Discovery: 2026-11-10, local environment audit.

- Symptom: the first aggregate invocation failed with missing `mpmath`; the local
  project environment used Python 3.14, while CPU CI is configured for 3.11–3.13.
  System uv was 0.11.20, versus CI's 0.11.21.
- Impact: validation could stop before exercising the numerical code, and a
  locally green run would not represent the declared CI interpreter matrix.
  Python 3.14 was not prohibited by `requires-python`; this was a validation-
  context mismatch, not proof that Python 3.14 was incompatible.
- Cause: the runtime-only environment lacked optional development dependencies;
  the machine's default interpreter was newer than the CI matrix.
- Fix: synchronize locked `dev` dependencies with managed Python 3.13.13; invoke
  the CI uv version with `uvx --from 'uv==0.11.21' uv` without replacing system uv.
- Verification: `uv pip check`, imports, tests, numerical reports, Ruff and strict
  docs passed in the recorded development environment. Package and runtime
  details are in the [normalization evidence bundle](../validation/audits/gpu-normalization-2026-11-10/README.md).
- Prevention: check the actual interpreter, extras and tool versions before
  attributing an import/environment failure to the numerical implementation.

## ENV-002 — Reproducible CUDA dependencies

Discovery: 2026-11-10, GPU installation and synchronization review.

- Symptom: installing `jax[cuda13]==0.10.1` directly added plugins and NVIDIA
  libraries that were not selected by the project lock. Ordinary project sync
  could remove them; a JAX version alone did not pin CUDA transitive versions.
- Impact: acceleration could disappear after routine environment maintenance,
  and the GPU dependency set was not reproducible from the project alone.
- Fix: add a Linux x86-64 `cuda13` optional extra and deliberately regenerate and
  review `uv.lock`, keeping CPU CI's `dev` selection and existing JAX version.
  Keep CPU and GPU environments separate and select both `dev` and `cuda13` for
  GPU sync/run commands.
- Verification: locked GPU sync preserved the plugins, dependency compatibility
  passed, and JAX detected and executed on the RTX 5070 Laptop GPU.
- Limits: the Python lock does not install or pin drivers/hardware; the declared
  extra does not claim support for every platform.
- Prevention: declare optional accelerator dependencies and use explicit extras;
  do not depend on packages left over from an ad hoc installation.

## NUM-001 — GPU dense-history precision

Discovery: 2026-11-10, float32 GL FFT/full-history regression investigation.

- Symptom: the multi-component GPU comparison failed by about `0.00665` against
  an independent float64 discrete convolution. The FFT path's error was only
  about `2.4e-6`; eager and JIT both exhibited the dense-path problem.
- Impact: the supposed full-history reference could be the less accurate path,
  corrupting fractional derivative values and invalidating reference comparisons.
- Cause: default accelerator matrix-product precision was insufficient for the
  cancellation-sensitive history contraction. The exact low-level arithmetic
  mode was not established; no native FFT defect was demonstrated.
- Fix: request `precision="highest"` for dense uniform/per-feature and nonuniform
  operator history matrix products, without changing the full-history method.
- Verification: independent GL/RL multi-component float32 regressions and the
  original FFT/JIT comparisons pass on CPU/CUDA. See
  `tests/unit/test_gpu_history_precision.py` and `test_fft_history.py`.
- Limits: highest precision is not exact arithmetic and is not a universal
  accuracy guarantee for arbitrary sequence lengths or scales.
- Prevention: compare candidate and reference implementations independently;
  explicitly select appropriate contraction precision in scientific references.

## NUM-002 — Legacy SOE fitting precision

Discovery: 2026-11-10, expanded GPU suite after the dense-history fix.

- Symptom: selected legacy GL SOE cases had relative errors about `0.217` and
  `0.583`, exceeding the existing `0.2` case-specific bound. A well-conditioned
  independent normal-equation probe differed by about `2.57e-5`.
- Impact: the approximation fitted on GPU could depart substantially from the
  already limited legacy-fit evidence.
- Cause: reduced-precision products also affected the fitting Gram matrix and
  projected updates; the history-application fix alone did not cover this path.
- Fix: request highest precision for legacy fitting normal equations and
  projected-gradient products. Preserve the fitting objective, pole policy and
  fixed iteration counts.
- Verification: independent normal-equation regression and existing SOE/history
  tests pass on CPU/CUDA. The focused combined history test matrix passed 112
  cases on each backend at that recorded increment.
- Limits: better arithmetic does not repair ill-conditioning, approximation
  floors or poor legacy regimes; legacy fitting remains an explicitly limited
  default policy, not a tolerance-controlled fit.
- Prevention: trace precision through fitting as well as application and test a
  well-conditioned independent problem before diagnosing approximation quality.

## VAL-001 — Float32 FFT comparison tolerance

Discovery: 2026-11-10, after isolating the dense precision defect.

- Symptom: the scalar GPU full/FFT difference was about `1.02e-5` to `1.20e-5`,
  slightly above a fixed absolute `1e-5` pairwise threshold. Each implementation
  was closer to an independent float64 discrete sum than their mutual difference.
- Impact: a pairwise test could report a defect caused by combined rounding,
  while the separate multi-component defect must not be hidden by a wider bound.
- Fix: retain the strict multi-component regression; add independent host-float64
  discrete-sum assertions for each scalar implementation (`1.5e-5` absolute) and
  use a scoped `2e-5` pairwise bound for that selected float32 case.
- Verification: CPU/CUDA tests pass; float64 `1e-10` equivalence checks remain.
- Limits: these are case-specific arithmetic tolerances, not a general FFT error
  bound or permission to relax unrelated failing checks.
- Prevention: establish independent errors before changing a comparison
  tolerance; record exactly which assertion changed and which protections remain.

## ENV-003 — CPU backend required for debug callbacks

Discovery: 2026-11-10, full CUDA test run.

- Symptom: two executed-work-count tests failed because `jax.debug.callback`
  could not find a CPU device with `JAX_PLATFORMS=cuda`.
- Impact: numerical GPU execution worked, but instrumented tests failed before
  their callback-based work accounting completed.
- Cause: the CPU backend was explicitly disabled, although JAX uses it to stage
  debug callbacks.
- Fix: use `JAX_PLATFORMS=cuda,cpu`. CUDA initialization is still required and
  CUDA remains the default backend; CPU is available for callbacks.
- Verification: both callback tests passed on targeted rerun; recorded full GPU
  suite passed. Documented commands use the corrected setting.
- Prevention: distinguish a GPU default device from a GPU-only backend list;
  test diagnostic/instrumentation paths as well as forward computations.

## NUM-003 — Gamma normalization versus Newton residual

Discovery: 2026-11-10, independent GPU implicit-residual check.

- Symptom: the cubic first-step corrector reported scaled residual `0.03296`,
  but an independent host-float64 reconstruction measured `1.79460` against the
  unchanged limit of one.
- Impact: algebraic convergence of the rounded working equation was mistaken
  for accuracy relative to the intended discrete equation.
- Cause: general-purpose CUDA float32 Gamma evaluation shifted the corrector
  normalization by approximately `-1.27e-6` relatively. Actual float32 order
  rounding was included in the reference and was not the main cause. Additional
  Newton iterations did not remove the coefficient bias. Constant-folded and
  dynamic-order JIT evaluations could differ.
- Fix: use a JAX-native reciprocal-Gamma series on the supported order interval
  and derive the corrector scale by recurrence. Do not change Newton iteration
  counts, diagnostic tolerances, independent thresholds or global x64 settings.
- Verification: the independent residual falls to `0.0783186`, while the reported
  value is `0.0247206`; both CPU/CUDA 28-row aggregate checks pass. Constants,
  values, order derivatives, eager/JIT behavior and dtype preservation have
  independent regression coverage in `tests/unit/test_solver_normalization.py`.
  Raw data, source hashes and complete recorded baseline suites are in the
  [normalization bundle](../validation/audits/gpu-normalization-2026-11-10/README.md).
- Limits: the normalization fix is not a global integration-error bound or a
  guarantee of positivity/nonlinear stiffness robustness.
- Prevention: reconstruct residuals independently of production coefficients;
  test strong float32 orders dynamically under JIT, not only constant-folded
  examples, and distinguish coefficient error from iteration error.

## VAL-002 — Dedicated FFT order-gradient coverage

Discovery: 2026-11-10, Phase A acceptance audit.

- Symptom: the capability matrix explicitly identified missing dedicated FFT
  order-gradient finite-difference evidence.
- Impact: full-history order-gradient evidence could not automatically justify
  corresponding FFT or per-state claims.
- Fix: add a selected Caputo/GL/RL matrix with scalar/per-state orders,
  float32/float64, eager/JIT gradients, finite differences and full-history
  derivative comparisons; update only the supported evidence claims.
- Initial verification: 24 cases passed separately on CPU and CUDA in
  `tests/unit/test_fft_order_gradients.py`. Review subsequently found the nominal
  float32 cases promoted forward computation to float64; those initial passes
  do not establish float32 execution coverage. See VAL-004 for correction.
- Limits: odd/padding-sensitive lengths, broader broadcasting, JVP/VJP and
  scheme-specific sensitivity/convergence work remain in the Phase A programme.
- Prevention: track differentiation coverage by history, dtype and order shape;
  passing one path is not coverage of all implementations.

## DOC-001 — Nonuniform solver stability assignment

Discovery: 2026-11-10, independent candidate audit.

- Symptom: the exported `NonUniformPredictorCorrector` was omitted from the
  canonical API stability table despite being described elsewhere.
- Impact: the public contract was incomplete and could obscure its provisional,
  scalar-order, full-history boundary.
- Fix: add the existing solver to the table without promoting its tier or adding
  capabilities.
- Verification: 11 documentation consistency/status guards and strict MkDocs
  build pass.
- Prevention: reconcile exported symbols, capability descriptions and stability
  assignments whenever reviewing public-surface changes.

## VAL-004 — Fixture-induced precision promotion

Discovery: 2026-11-10, independent pre-commit review of the FFT order-gradient
matrix; reproduced independently on CPU before accepting a fix.

- Symptom: with the x64 fixture enabled, strongly float32 inputs and orders
  produced float64 FFT/full outputs and a float64 objective, while differentiation
  still returned a float32 order gradient. The existing gradient-dtype assertion
  therefore passed without validating float32 forward execution.
- Impact: the proposed float32 coverage claim was unsupported even though the
  initial 24-case matrices were green. The original 713-test normalization bundle
  predates this matrix and is preserved unchanged.
- Cause: JAX default-dtype weight/padding construction promoted the computation
  when the shared fixture enabled x64 unconditionally. Declared input dtype is
  not necessarily the effective computation dtype.
- Fix: add forward-output/objective dtype assertions first, then scope x64 off
  for float32 and on for float64 outside JIT, restoring the prior setting
  afterward. Numerical tolerances and finite-difference steps are unchanged.
- Verification status: reproduction confirmed all three operators had float64
  forward/objective outputs with x64 enabled and float32 outputs with it disabled.
  The new assertions first produced 12 failed/12 passed cases. After correction,
  all 24 cases passed on each CPU/CUDA backend, with forward/objective and
  gradient dtypes checked. Replacement full suites passed 737 tests on each
  backend, one performance case deselected, with both process exits zero.
  Independent review approved tree `a7b1ac4e2d1b3cbab24d83124637157a448d9299`,
  subsequently committed as `ec75960`. The two pre-review
  full-suite processes were deliberately stopped as superseded, not reported
  as successful checks. Raw red/green results and corrected source/runtime
  provenance are preserved in the [FFT precision bundle](../validation/audits/fft-order-gradients-2026-11-10/README.md).
- Prevention: assert effective forward/objective dtypes as well as input and
  gradient dtypes; test each precision mode explicitly rather than inferring it
  from parameter labels.

## ENV-004 — GitHub write authentication

Discovery: 2026-11-10, authorized push preflight.

- Symptom: public fetch/API reads succeeded, but an HTTPS push dry run failed
  because a username could not be read with prompts disabled. A read-only SSH
  probe with strict host checking also failed because no trusted GitHub host key
  was configured.
- Impact at discovery: local review and commits could proceed, but a remote push
  and candidate CI verification were blocked by the authentication setup.
- Cause: public read access does not establish authenticated repository write
  access; neither tested transport supplied a usable authenticated path.
- Resolution: the user completed their trusted GitHub authentication workflow.
  `gh auth status` and the push dry run succeeded; `ec75960` was pushed to
  `origin/fix/phase-a-gpu-hardening` and the exact remote SHA was verified using
  `git ls-remote`. No credentials were requested in chat or manually extracted
  from stores; no remote URL, SSH host key or key generation workaround was
  applied. [Remote CI](https://github.com/dave2k77/hpfracc/actions/runs/42880342249)
  was observed in progress; its result is not implied by authentication or the
  local test passes.
- Prevention: test write access with a non-mutating push preflight early, and
  never infer push permission from a successful public fetch.

## Open findings and next work

### SCI-001 — General adaptive error calibration

The default step-doubling and optional subdivision indicators share old-history
error and have documented underestimation/cancellation counterexamples. The
bounded prescribed-C2-forcing method does not establish calibrated nonlinear
feedback control. Impact: completion and decreasing errors cannot be interpreted
as an accuracy guarantee. Resolution requires a justified scoped error-control
contract and independent evidence, or explicitly reviewed scope deferral; more
passing sweeps alone are insufficient. See the [Phase A programme](phase-a-completion.md).

- Reconstruction finding: a checked one-panel PECE feedback example differs from
  the fractional integral of its corrected-state RHS interpolant by
  -0.0099770623402870312. This is an 80-digit mathematical design check, not a
  newly observed production regression. PECE corrects with the predictor RHS,
  then stores the corrected RHS; that fractional reconstruction need not pass
  through the actual stored state. Ignoring the nodal mismatch could invalidate
  a proposed error certificate.
- Intervention: preserve a [private Volterra-defect design and feedback tracer](../validation/audits/a1-defect-design-2026-11-11/README.md).
  The design uses a continuous reconstruction through stored states and retains
  all old-panel defect/propagation contributions. The scalar constant-candidate
  helper demonstrates nonzero feedback propagation; it is not a controller or
  general supplied-trajectory enclosure implementation.
- Verification: independent high-precision checks reproduce, the missing-helper
  feature test goes RED to GREEN, and 15 private tests pass. Independent review
  reproduces those tests and 27 additional scalar probes. Five intentionally
  incorrect shortcut assertions fail as design negative controls, not production
  failures. The original nondiscriminating old-panel fixture and its versioned
  preimplementation correction are both retained.
- Limits and prevention: artifact review passes only for the explicit private
  diagnostic; A1 acceptance remains false. Verified arithmetic, complete
  continuum-defect enclosures, nonlinear/stiff propagation, acceptance matrices
  and reviewed controller/JAX integration remain open. Check which trajectory
  an oracle actually encloses; never promote high-precision agreement or a raw
  unpropagated defect into total working-precision certification.
- Subsequent implementation: the [supplied-trajectory diagnostic](../validation/audits/a1-trajectory-defect-2026-11-11/README.md)
  evaluates the complete PL/affine-power defect and signed continuum-cell model
  ranges without changing any library runtime or controller. Five actual feature
  RED/GREEN cycles are retained. After VAL-008/009/010 hardening, 81 focused cases
  pass in each CPU/CUDA process. Diagnostic arithmetic is host float64, including when
  represented float32 solver outputs are supplied; this is not GPU-kernel or
  float32-diagnostic arithmetic evidence. Full-suite/review records are separate.
- Arithmetic finding: an independent fixed positive-width cell for `Y=0`,
  `A=0`, `q=1`, half order misses the true endpoint residual by approximately
  `1.53e-17` using ordinary float64 natural-extension arithmetic. Root cause:
  nearest-rounded gamma, powers, coefficients and sums do not meet the exact
  inclusion theorem's enclosing-arithmetic premises. Impact: even an
  exact-arithmetic enclosure formula is not a numerical error certificate.
- Applied guardrail: name the outputs model ranges, keep both verification/
  certification flags false, fail explicitly on invalid/nonfinite inputs or
  cell-work exhaustion, and preserve the under-enclosure as a limitation
  regression. No arbitrary padding, tolerance relaxation or global x64 repair
  is substituted. Verified arithmetic and feedback propagation remain open;
  preventing a false claim does not fix the arithmetic limitation itself.

### SCI-002 — Approximation selection and memory evidence

Fixed SOE poles/iterations do not certify a requested error budget; forward
recursive state size does not describe reverse-mode or measured peak memory.
Impact: accuracy-preserving efficiency claims remain incomplete. Resolution work
must separate fitting/application/gradient costs, record unattainable budgets,
and measure rather than infer memory. Existing conditional bounds and CPU
compiler-accounting artifacts remain valid only within their declared scope.

A2 preparation, 2026-11-11: the owner authorized getting ready to start, not
implementation. The [draft contract](a2-compression-budget-protocol.md) separates
exact finite GL/RL compression decisions from working arithmetic, continuous
error and measured memory. Existing fixed fitting/recursive application remain
unchanged. It proposes exhaustive supplied-candidate assessment, a deterministic
minimum among qualifying candidates and distinct unattainable-supplied-budget,
invalid-input and indeterminate-work outcomes; none is yet implemented.

Preparation evidence: 68 exact synthetic signal-prefix identity cases and 136
exact-rational bound cases pass, including controls exposing omitted initial
values/normalization. These are planning calculations, not fitter/selector or
numerical benchmark results. Independent mathematical and validation/
resource reviews reject the initial and first corrected packets; see VAL-013.
Decision semantics and finite mathematics pass, but schema/export/provenance and
output-ownership interlocks block the first revision. Final independent review
passes the repaired specification's consistency but rejects freeze readiness:
the whole-index byte reservation cannot admit eight required float32 cases even
after correcting conservative branch assumptions. The owner has now authorized
one bounded budget-only preparation revision and independent re-review. That
attempt stopped at incomplete export accounting before any packet edit; independent
assessment confirms no new budget is justified. No budget, case/check or numerical
tolerance has changed. Implementation
and A2 acceptance remain gated. No numerical source or historical evidence is
changed to resolve preparation findings.

### VAL-003 — Scientific acceptance versus integrity

Some audit `--check` policies intentionally validate complete, coherent
measurements, not continuous accuracy or successful integration in every row.
Impact: a green audit process can be misread as a scientific accuracy gate.
Preserve those existing semantics and add a separately named scientific policy
with independently defined required-success/error criteria, expected-failure
classification, exact coverage and corruption sensitivity. This is the next
Phase A work package; it does not change the meaning of the existing reports.

Implementation started 2026-11-10 in
`benchmarks/numerical/scientific_acceptance.py`. The first regression uses the
existing independent half-order decay oracle: a finite, Newton-converged coarse
float64 trajectory has dimensionless error above 0.3 despite an acceptable
independent algebraic residual. The new decision rejects it. Separate real
regressions reject nonfinite and finite under-iterated trajectories.

- Fix: a [separate frozen policy](../validation/scientific-acceptance.md) defines
  exact required-success and rejection-control coverage, continuous-error
  budgets and strict reported/independent residual requirements. Decisions and
  booleans are recomputed, not trusted from supplied result flags. Multiple
  rejection reasons are retained; invalid measurements cannot pass as expected
  numerical failures. Original measurement-report semantics remain unchanged.
- Verification so far: observed red/green cycles, 81 focused CPU tests and 103
  focused CUDA tests pass, both processes exited zero. Deliberately corrupted
  metrics/flags/metadata are regression inputs, not benchmark data. Eight
  executable reports preserve 48 measured configurations: 30 accepted
  required-success trajectories, two required-success failures and 16 correctly
  rejected controls. Both direct float32 gates exit one; six other report
  variants exit zero. Raw trajectories/diagnostics reconstruct every reported
  row and decision on the host; all 61 source-input hashes per run and all
  manifest/CSV/NPZ digests match. See the
  [development evidence](../validation/audits/scientific-acceptance-2026-11-10/README.md).
- Review correction: the inherited float32 accuracy evidence used stable, not
  direct weights. Evaluation mode is now an explicit manifest dimension, with
  no automatic stable substitution. New float32 nonlinear budgets are labelled
  policy choices, not borrowed evidence. Fully coherent fabricated data cannot
  be authenticated by a row-only checker; raw arrays/digests support numerical
  reinspection, not execution attestation.
- Additional verification: the replacement full CPU suite passed 818 tests,
  one performance case deselected, with process exit zero. Ten later
  documentation-consistency checks are not counted in that run; all 21
  documentation/status guards pass separately. Initial review found only
  documentation drift (VAL-006), corrected against the final measurements.
- Remaining: the default direct precision limit is separately tracked as
  NUM-004. Adaptive calibration, broader nonlinear/stiff scope and release
  sign-off remain open Phase A work; local passes do not imply new remote CI.
- Prevention: freeze configuration-specific criteria before measurements, test
  corruption and exact coverage, and report accepted trajectories separately
  from verified rejection controls and measurement integrity.
- Evidence-handling lesson: raw pytest failure XML contains trailing whitespace
  that triggered `git diff --cached --check`. Preserve red JUnit as gzip with
  verified raw/compressed hashes instead of stripping historical evidence or
  disabling repository-wide whitespace checks.

### VAL-005 — Canonical configuration identity

Discovery: 2026-11-10, pre-review corruption testing of the new policy. This was
caught in development before the policy was committed or published.

- Symptom: Python/dataclass equality treated boolean `True` as strength `1.0`
  and floating `129.0` as integer node count `129`. Four new mutation cases
  passed the original configuration comparison instead of failing closed.
- Impact: a forged or malformed manifest/measurement could appear to match
  the fixed protocol even though its canonical types differed. Immutability
  and value equality were not sufficient configuration identity checks.
- Cause: field and dataclass equality compare numeric values across these
  types. Earlier strict boolean-result tests did not cover configuration fields.
- Fix: match both exact canonical field types and values against the separately
  authoritative manifest, for specifications and measurement identities; keep
  explicit boolean-result checks and decision recomputation.
- Verification: four expected failures in the 81-case CPU regression, then all
  81 focused CPU and 103 targeted CUDA tests pass with zero process exits.
  Archive the pre-correction reports/source snapshots and regenerate all eight
  reports after the fix; host/raw/hash verification still preserves the two
  scientifically failing direct float32 cases. The earlier full CPU run was
  stopped as superseded; its pytest summary is not final process-exit evidence.
- Limits: type/coverage/digest checks are not authentication of fully coherent
  fabricated observations; raw-data reconstruction and scope disclosure remain
  necessary. The replacement full CPU suite passed 818 tests; later
  documentation guards are recorded separately, not counted in that run.
- Prevention: test Python numeric coercion edge cases in scientific metadata,
  not just modified numerical values or truthy result flags.

### VAL-006 — Current-outcome citations drifted after regeneration

Discovery: 2026-11-10, independent review of staged tree
`26d778a342224cf24877bbf3d0df4ce217876404`; no numerical-code or security blocker
was found, but the current-evidence citations did not match the final bundle.

- Symptom and impact: two rounded README cells and two full-precision CUDA
  citations in NUM-004 still used the pre-type-guard report values. Scientific
  decisions were unchanged, but the displayed measurements had the wrong
  evidence identity. Historical values remain valid only for their archived run.
- Cause: reports were regenerated after the configuration-type fix while
  manually copied outcome text retained the previous CUDA measurements.
- Fix: derive current values from the final bundled JSON; retain the old
  reports/source snapshots unchanged. Add ten lightweight checks comparing
  every README outcome/exit code, verification summary and current NUM-004
  CPU/CUDA residual citations against the actual report rows.
- Verification: four failed/six passed checks against stale citations, then
  all 21 documentation/status guards pass. The numerical inputs and all 61
  source hashes per report remain unchanged. Red/green evidence is archived
  alongside the initial failed review and the full-suite record.
- Prevention: regenerate all displayed outcomes from the final evidence set
  after a rerun, and test the publication surface as well as raw artifacts.

### NUM-004 — Direct float32 history-weight cancellation repair

Discovery: 2026-11-10, the new separately named scientific-acceptance gate;
consistent with the older direct-weight limitation, not a new solver regression.

- Historical symptom: the pre-repair 257-node, order-0.5, strength-1 decay case
  reported Newton convergence and met its 1e-3 continuous-error budget, but its
  independent maximum scaled residuals were 12.453928382961097 on CPU and
  8.497625302550595 on CUDA, above the frozen threshold of one. These remain
  historical failures in the immutable
  [pre-repair bundle](../validation/audits/scientific-acceptance-2026-11-10/README.md),
  not measurements of the repaired arithmetic.
- Impact: small trajectory error and a converged reported equation do not imply
  an independently converged product-integration equation. Stable-path or older
  aggregate passes cannot establish default-path acceptance.
- Evidence-backed cause: the original direct power-difference formulas suffer
  working-precision cancellation. The archived dynamic-JIT CPU probe at step 256
  compares five positions with the existing 70-digit interval-integral oracle:
  legacy direct corrector max relative error was 0.0019559953760854842 versus
  5.843177619535567e-08 for stable evaluation. This probe diagnoses the original
  arithmetic; it is not a current coefficient measurement.
- Applied repair: `_pece_weights` evaluates the same Diethelm coefficients with
  an `expm1`/`log1p` predictor and positive binomial corrector series, in the
  working dtype. Boundary degrees 28/56 and 14/28 interior even terms bound
  exact-arithmetic truncation below float32/float64 epsilon. Inactive operands
  are finite before selection, including singleton/masked cases. See
  [uniform weights](../validation/uniform-weights.md) for the bounds and limits.
  The full-history discretization, default `direct` selector, timestep/Gamma
  normalization, Newton updates and tolerances are unchanged. **Legacy direct
  floating-point roundoff is not preserved.** The explicit `stable`
  interval-moment route and separate stochastic kernel are unchanged.
- Selected verification: two pre-repair regressions failed on each CPU/CUDA
  backend; source checkpoint and raw red evidence are retained in the
  [NUM-004 repair bundle](../validation/audits/num004-direct-precision-2026-11-10/README.md).
  Its eight finalized reports all pass the unchanged `implicit-scientific-v1`
  policy: all 32 required-success trajectories are accepted, and all 16 negative
  controls have `accepted=False`, `policy_passed=True`. Current direct independent
  maximum scaled residuals are 0.016331607987060983 on CPU and 0.01743271498279139 on CUDA,
  below the unchanged limit of one. Current stable residuals are
  0.052358017489466745 on CPU and 0.01643224778294578 on CUDA.
  These two stable values are unchanged from the archived baseline. Host replay
  verifies the 48 raw configurations, protocol/artifact digests and 61 source
  hashes per report; this is not universal scientific acceptance.
- Verification completion: both captured-source full CPU/CUDA processes exited
  zero with 850 passing tests and one performance case deselected per backend.
  Both backends also passed the 187-test
  existing selection and the 14-test expanded precision/gradient selection.
  Independent final review and remote CI are recorded separately from these
  local runs; neither release nor Phase A closure is inferred.
  No general arbitrary-nonlinear/stiff float32 accuracy claim follows. Working-
  precision model evaluation, normalization and accumulation still matter, and
  neither evaluation cures the coarse-grid stiff-decay positivity/inaccuracy
  control. Phase A remains open.
- Prevention: always identify weight evaluation and effective precision in
  numerical claims, and verify against an independently evaluated equation
  rather than only the residual of the computed working-precision weights.
- Evidence-drift prevention: derive current citations from the final bundled
  JSON rows after reruns, preserving superseded evidence unchanged. The docs
  consistency regression keeps all eight historical README checks, adds eight
  repair-bundle outcome checks, and binds this entry's two current paired
  residual claims and their evidence attribution to the new bundle.

### NUM-005 — Stable float32 stiff-case independent residual disagreement

Discovery: 2026-11-10, while preserving the historical stable-versus-legacy
comparison during the NUM-004 repair.

- Symptom: for CPU float32 half-order decay, strength 30, 257 nodes, eight
  Newton iterations, `rtol=1e-5`, `atol=1e-7`, the explicitly selected stable option
  reports convergence but its independently reconstructed maximum scaled
  residual is 1.1998344389138271, above one. This was identical before and after
  NUM-004; that repair left the stable kernel and history contraction unchanged.
- Impact: the historical improvement relative to legacy direct arithmetic is
  real, but does not establish an absolute independent residual pass. This
  exploratory case is outside the frozen six-case matrix and provides no
  continuous-accuracy acceptance or general stiffness guarantee. NUM-004's
  selected success must not be generalized to this different configuration.
- Initial cause assessment at the NUM-004 checkpoint: working-precision
  coefficient, model and accumulation roundoff were hypotheses, not an
  established complete explanation.
- Historical NUM-004 intervention: preserve the comparison using an explicitly frozen
  legacy evaluator, retain the unchanged 5% relative-improvement assertion, and
  separately require the repaired direct result's independent residual to be
  at most one. No stable-path arithmetic or tolerance has been changed.
- Verification: the archived
  [float64-reference observation](../validation/audits/num004-direct-precision-2026-11-10/stiff-residual-float64-reference.json)
  records the before/after runs. Legacy direct residual 673.7955854875966 becomes
  0.20526358291781596; the stable value above is unchanged. The corresponding
  updated regression passes in both 187-test CPU/CUDA selections.
- Instrumentation correction: the first exploratory caller did not cast states
  to host float64 before the half-decay RHS. It is explicitly superseded in
  `stiff-probe-history.json`; original files remain unchanged. Frozen-policy
  collectors already cast reference states and all eight reports are unaffected.
- Boundary: this limitation was not silently accepted, deferred from Phase A,
  or fixed by NUM-004. The selected algebraic repair below does not establish
  continuous accuracy or general stiff-solver robustness.
- Prevention: compare independently evaluated absolute criteria as well as
  relative improvement, and establish the reference dtype before evaluating
  the RHS rather than promoting an already rounded residual afterwards.
- Pre-repair evidence: the source checkpoint is `415fc21`; its 116 numerical,
  benchmark and test inputs are archived before this repair attempt. The exact
  CPU case fails the new unchanged residual-at-most-one regression, whereas
  the CUDA control already passes at 0.23292174906264487. A separate dyadic
  scalar-history example loses a representable aggregate of small terms with
  the original contraction; trailing state axes change the rounding behavior.
- Controlled counterfactual: requesting highest dot precision leaves CPU/CUDA
  results unchanged. Replacing only history contraction with an elementwise
  weighted reduction lowers the CPU stable-case residual to 0.2770276016181242,
  with coefficients, normalization, dtype, Newton settings and thresholds
  unchanged; the measured CUDA case is unchanged. Raw red/control XML, exact
  test/probe sources and measurements are retained in
  `docs/validation/audits/num005-stable-precision-2026-11-10/`.
- Confirmed cause: independent analysis reproduces worst step 209 bit-for-bit
  using the public trajectory and a matching compiled replay. History product
  and reduction error contributes 1.1270974033428172 units of the independent
  tolerance; stable coefficient error contributes 0.022570372065334462 and
  Newton's runtime-equation error 0.0021260254540220965. The signed decomposition
  reconstructs the failing residual; 80-digit panel integration confirms it.
  Sequential float32 summation reproduces the observed dot, while highest dot
  precision has no effect. Cancellation with the initial condition amplifies
  history error; the final known-term operation's own rounding is small.
- Applied repair: `_history_dot` now broadcasts weights over trailing state axes
  and uses an explicit weighted array reduction. This changes floating-point
  accumulation, not the full-history discretization, coefficients, normalization,
  selector defaults, working dtype, Newton iterations or acceptance thresholds.
  It covers the shared uniform explicit/implicit, nonuniform and adaptive paths.
- Verification so far: the unchanged four-case regression file passes on CPU
  and CUDA (four tests each). Before the fix, CPU had two failures/two passes;
  CUDA already passed all four controls. Eight new frozen scientific-policy
  reports preserve 48 raw cases: all 32 required successes are accepted and all
  16 negative controls remain rejected. `verification.json` replays every case,
  exact backend/dtype/mode coverage, manifests, CSVs and source/artifact hashes.
- Additional checks: CPU/CUDA aggregate (28), hardening (12) and history (8)
  gates pass with zero exits. A newly built wheel passes isolated, hash-locked,
  runtime-only CPU import/JIT/float32 objective-gradient and selected independent
  residual smoke checks, including the stable stiff case; see the bundle's
  `additional-gates.json` and `installed-wheel-smoke.json`.
- Compatibility: 25 additional cases pass on CPU and CUDA, covering signed and
  mixed-scale histories, trailing shapes, real/complex private-helper dtypes,
  analytic JVP/VJP and finite differences, dynamic-order JIT stiff controls and
  shared uniform/nonuniform/adaptive paths. Original tests and production bytes
  are unchanged; eight raw evidence files and their digests are retained in
  `compatibility/` and `compatibility-manifest.json`. This does not establish
  public complex implicit support or general adaptive accuracy.
- Full-suite progress: CPU and CUDA each pass 879 tests, one performance case
  deselected, with confirmed zero exits and all 118 numerical/benchmark/test
  inputs unchanged. Raw XML and readback verification are retained. VAL-007
  records differing isolated/full-suite CUDA residuals, both below one, and
  their reproduced compilation-path difference. These measurements were
  collected on uncommitted source; independent review and delivery are separate
  records, not Phase A closure. The prior NUM-004 commit's remote
  Python 3.11–3.13 CI is separately verified successful in `prior-commit-ci.json`.
- Instrumentation lesson: returning intermediate weights or adding debug
  callbacks changed fusion and made the unfixed public case appear to pass.
  Require a bit-for-bit match to the original public trajectory before using
  an instrumented trace to attribute roundoff. The complete matching replay,
  basis-coefficient decomposition and counterfactuals are preserved verbatim in
  `independent-analysis.tar.gz`, with 42 file digests in its manifest.
- Delivery follow-up: the independently reviewed repair is committed and pushed
  as `7a11d2d`. Remote run `42971612268` is now read back successful for that
  exact SHA: Python 3.11–3.13, lint, strict docs and the separate performance job.
  The readback is retained in the subsequent API-001 bundle, not retroactively
  relabelled as evidence for the later metadata changes. Phase A remains open.
- Citation prevention: retain NUM-004 values as checkpoint-specific history and
  publish the changed NUM-005 weak-case values separately. The archive replay
  checker first failed on the missing current-evidence section, then binds all
  eight new weak/stiff table entries to their measured reports. Numerical and
  test source bytes are unchanged by this documentation guard.

### VAL-007 — Exact cross-context CUDA residual comparison

Discovery: 2026-11-11, while reconciling completed NUM-005 full-suite artifacts.

- Symptom: a new post-run bookkeeping assertion required exact equality between
  properties from the isolated 25-case compatibility file and the full suite.
  CUDA stable/direct values were 0.23292174906264487 / 0.24529762807235306 in
  isolation and 0.22442286756041424 / 0.2788388698063575 in the full suite.
  CPU values were identical across the two contexts.
- Impact: the extra reconciliation assertion failed after both actual suites
  had already passed 879 tests with zero exits. This is not a pytest failure or
  a failed independent-residual criterion: every recorded value is below the
  unchanged limit of one, with strong float32 forward/objective assertions.
- Established bookkeeping cause: exact cross-context equality was an additional,
  unsupported predicate, not the frozen scientific acceptance criterion. At
  discovery, config leakage and compilation effects were hypotheses; the
  bounded follow-up below distinguishes observed mechanism from unresolved
  emitter selection.
- Intervention: retain both raw XML datasets, the failed postcheck source/error,
  and context-labelled properties in `full-suite-verification.json`. Check exact
  source/test identities and each observation against the same residual limit;
  do not overwrite one observation with the other or relax a scientific budget.
  No production or test-source change is made for this observation.
- Bounded follow-up: both pairs recur under identical target-only commands.
  Captured StableHLO/configuration are identical and operations f32-only, with
  no dot; optimized history reductions differ between Triton and native
  emitters. Five replays per executable are bitwise stable. Parent host replay
  reproduces all four residuals; maximum trajectory differences are below 1e-7.
  Ten bounded pytest processes pass 66 executions with actual zero exits. The
  74-file archive and manifest retain IR, states, commands, logs and XML.
- Limits: collection breadth, preceding tests and verbosity do not uniquely
  determine the pair. Why XLA chooses different emitters remains unresolved;
  the original historical optimized executables were not captured, so their
  precise emitter cannot be established retrospectively. No nondeterministic-
  autotuning explanation or cross-compilation bitwise guarantee is asserted.
- Diagnostic failure: a separate optional DebugOptions introspection subprocess
  exits 139. Its exact command/exit is retained, and a NumPy-only comparison
  subsequently succeeds. This is not a pytest or solver acceptance failure;
  no claim that every investigation command succeeded is made.
- Evidence: the NUM-005 bundle's `postcheck-exact-property-source.py.txt`,
  `postcheck-exact-property-red.txt`, isolated `compatibility/` files and
  `full-{cpu,cuda}.xml` preserve the discrepancy and successful test outcomes.
  `cuda-context-investigation.tar.gz` and its manifest retain the follow-up.
- Prevention: distinguish bit-matching replay of one executable for causal
  decomposition from numerical comparison across separately compiled execution
  contexts. Claim only the reproducibility actually checked, and preserve every
  observed context rather than selecting a preferred numerical value.

### API-001 — Strong-order solver metadata is not plain-JSON serializable

Discovery: 2026-11-10, additional metadata inspection during precision tests.

- Symptom: both uniform deterministic solver classes accept a strong float32
  scalar array order, but `json.dumps(result.solver_info.to_dict())` raises
  `TypeError: Object of type ArrayImpl is not JSON serializable`. The equivalent
  Python-float order exports successfully. Numerical trajectories are unaffected.
- Cause: `SolverInfo.to_dict()` uses `dataclasses.asdict` without converting
  the array stored in `fractional_order` to a JSON scalar. This qualifies the
  JSON-compatible-export expectation; numerical array-order support is not
  evidence of metadata serialization support.
- Intervention: no metadata implementation change in the weight repair.
  Record the failure separately; the selector test compares metadata identity,
  history and diagnostics without claiming successful JSON export.
- Verification: the
  [metadata observation](../validation/audits/num004-direct-precision-2026-11-10/metadata-export-observation.json)
  reproduces all four class/order-type combinations on the pre-repair source
  and repaired source. `solvers/base.py` is byte-identical across checkpoints;
  only the two strong-array cases fail in each checkpoint.
- Remaining at discovery: provide and test host-side JSON normalization without
  introducing array-to-Python conversion into numerical JIT/gradient paths. At
  that checkpoint the issue was open, not a new NUM-004 regression or a completed
  contract fix.
- Applied correction: `SolverInfo.to_dict()` now converts a non-missing concrete
  scalar order to a Python float only in its exported payload. The stored order,
  dataclass fields, diagnostic normalization and all numerical functions are
  unchanged. The method explicitly documents its host-only boundary.
- Regression evidence: two CPU tests first fail on the original `ArrayImpl`
  JSON error, then both pass after the correction. A 150-test targeted selection
  passes on CPU and CUDA, including scalar float32/float64 export through shared
  uniform/nonuniform/adaptive paths, direct/stable selectors, immutable metadata,
  independent analytic state/parameter/order gradients and lazy metadata import.
  These targeted runs precede final comment/style cleanup; their source and XML
  remain distinct from the final full-suite evidence now being collected.
- Final full suites each pass 900 tests on CPU/CUDA, one performance case
  deselected, with fresh process readbacks confirming zero exits after teardown.
  All 119 source inputs match the final capture and all enumerated test identities
  match collection and each other. Two initial full-suite launches were cancelled
  before completion to correct a stale selector-test comment; no test result is
  inferred from those runs. Their source snapshot is retained separately. The
  final capture records 119 source inputs and 900 selected tests. AST checks
  verify the original RED regression is unchanged, the selector edit is comment-
  only, and the only existing runtime change is the host export method.
- Independent source review passes without blocking findings. Its non-blocking
  suggestion to verify raw/compressed RED hashes is applied to the archive
  checker and exercised by the final host replay. Tree-specific review, later
  evidence additions and delivery are separate records, not Phase A closure.
- Boundary: this repair does not add per-state orders or whole-result JIT
  returns, and does not serialize arbitrary user-supplied diagnostic objects.
  Evidence is in the [API-001 bundle](../validation/audits/api001-metadata-export-2026-11-11/README.md).
- Prevention: test metadata export using actual supported scalar-array orders,
  not only Python configuration scalars, independently of numerical dtype tests.

## VAL-008 — Private A1 oracle forcing-ratio cancellation

- Discovery and impact: an additional pre-commit edge probe of the new private
  PL/affine-power diagnostic uses `Y=0`, `A=0`, `q=t**beta`, half order,
  `beta=1e300`, horizon one. It returns a fractional RHS of 1 instead of about
  `1e-150`. This is a new validation-helper defect caught before commit, not a
  production solver regression. Unverified arithmetic flags do not excuse a
  catastrophically wrong point diagnostic within the declared finite-power scope.
- Root cause: subtracting `gammaln(beta+1)` and `gammaln(beta+alpha+1)` loses the
  increment at large beta and produces zero, so exponentiation produces one.
  The same expression occurs in both pointwise and cell-term evaluation.
- Fix: both paths use one shared reciprocal-Pochhammer gamma-ratio helper,
  `1/poch(beta+1,alpha)`, without large log-gamma subtraction. No tolerance,
  working precision, library runtime, dependency or solver setting changes.
- Verification: nine represented-input cases (three large finite powers and
  three orders) fail before the repair. Independent log-gamma references use
  sufficient mpmath precision for the large arguments; comparisons have zero
  absolute slack so tiny nonzero coefficients cannot pass as zero. The Gamma-only
  checkpoint passes 64 focused cases per CPU/CUDA process, including both paths
  and the distinct float64 under-enclosure limitation regression.
- Evidence handling: the first two full-suite processes were explicitly killed
  to replace their numerical/test source snapshot after this discovery. They are
  cancellations, not passing or failing full suites. Their source, collection,
  static scan and original focused XML are preserved before the repair. Final
  full suites and corrected-tree implementation review are separate records in
  the [A1 trajectory bundle](../validation/audits/a1-trajectory-defect-2026-11-11/README.md).
- Remaining and prevention: ordinary reciprocal-Pochhammer evaluation is still
  not outward-rounded proof; both certification flags remain false. Check extreme
  finite arguments and tiny expected coefficients with relative tests and no
  permissive absolute floor. Prefer specialized ratios over cancellation-prone
  differences of large special-function values, and fix all sibling call paths.

## VAL-009 — Global PL hinge cancellation in the private defect diagnostic

- Discovery: the second independent code review rejects the Gamma-only tree
  `8dcbfafd1a5373088e75437551692a85fd7f7b77`, despite both full 964-test suites
  passing with zero exits and matching source/identity checks. On times
  `[0,1e-17,1]`, states `[0,1,1]`, half order and unit feedback, it reports zero
  reconstructed endpoint, history integral and defect. The endpoint must be one;
  independent 130-digit panel quadrature gives an integral about 1.128379 and
  defect about -0.128379. This is a pre-commit private-helper bug, not a solver bug.
- Root cause: large opposite global slope-jump terms lose the tiny time offset
  and cancel after rounding. Merely retaining every panel symbolically does not
  ensure its contribution survives arithmetic; finite-output guards miss this.
- Applied repair: local interpolation and positive per-panel PL basis integrals
  replace global hinges in point evaluation. Cell ranges use those same basis
  weights, retaining the active left weight's interior maximum and old-panel
  decay. A direct near-one hypergeometric prototype is rejected after a
  relative error above `2e-5`; equivalent log1p/expm1 expressions handle that
  regime. The failed design probe is preserved, not a claimed library failure.
- Verification: three graded-history regressions go RED before repair. The
  combined five new failure cases then pass within a 69-case selection; twelve
  compatibility cases bring the focused count to 81 in each CPU/CUDA process.
  Eighteen separate high-precision design checks corroborate selected basis
  formulas. Final source, suites and review are separate from the earlier
  passing-but-rejected checkpoint; all historical failures remain in the
  [trajectory bundle](../validation/audits/a1-trajectory-defect-2026-11-11/README.md).
- Limits/prevention: basis formulas and model ranges are not outward-rounded
  proofs. Check reconstruction at stored states and severe grading, compare
  independent panel quadrature, and validate special functions at both kernel
  limits before using a formally exact global expansion as a numerical oracle.
  The final review also rejects the replacement implementation for the distinct
  interpolation and active-gap cases in VAL-011; the selected repair is not a
  general numerical clearance.

## VAL-010 — Premature underflow before finite forcing scaling

- Discovery and impact: the same independent review uses half order,
  `beta=1100`, forcing coefficient `1e300`, zero candidate/feedback and target
  one half. Both point RHS and first-cell range silently return zero, although
  the true integral is representable, about `1.5690812007270112e-33`. Independent
  high-precision Gamma, integer-beta recurrence and transformed quadrature agree.
- Root cause: the time power underflows before multiplication by the large
  coefficient. Overflow/invalid/divide traps and final isfinite checks do not
  detect a finite zero that has lost the entire contribution.
- Applied guardrail: both host paths also trap intermediate underflow and raise
  explicitly. This conservatively rejects the unsafe calculation; it does not
  compute the rescued value or certify the remaining outputs. No global
  precision promotion, relaxed tolerance or approved scope deferral is implied.
- Verification: point/cell failure-path regressions both go RED, then pass with
  explicit FloatingPointError behavior. The old zero-output evidence and rejected
  review stay intact. All 81 focused cases pass per CPU/CUDA process; final
  full-suite and review evidence has its own source identity.
- Limits/prevention: a future stable scaled evaluation or verified arithmetic
  budget is still needed to accept this finite case. Test intermediate range,
  not just final finiteness, and prefer a clear unsupported-arithmetic failure
  over a plausible-looking zero used as evidence of accuracy.
  The outer trap does not cover `np.interp` internals: VAL-011 demonstrates a
  separate silent interpolation underflow in both APIs. Thus this guardrail
  covers the tested power path, not every operation in the diagnostic.

## VAL-011 — Final-cycle interpolation and active-panel arithmetic rejection

- Historical 92ff rejection checkpoint: implementation was paused after the
  then-final repair cycle. Independent review rejects immutable source tree
  `92ff3dceb217f6097bae8ccde6cf159d8daa8d14`. These are three remaining
  pre-commit private-helper defects, not regressions in the public solver. The
  helper is not an approved validation oracle; false certification flags do not
  excuse its incorrect point values or cell ranges.
- B1, local cancellation: times `[-2,-1,1e-17]`, scalar states `[0,1e20,0]`,
  zero feedback/forcing and target zero produce reconstruction/defect zero and
  last-cell range `[0,0]`. The exact represented-input PL value is approximately
  1000. `np.interp` still uses arithmetic that cancels this resolvable endpoint
  tail; replacing a global expansion by a local library primitive was insufficient.
- B2, silent internal underflow: times `[0,1e300]`, states `[0,1e-100]` and
  midpoint target `5e299` produce zero instead of the normal representable
  `5e-101` in both APIs. The internal interpolation slope is about `1e-400`.
  `np.errstate(under='raise')` catches scalar division but does not trap this
  internal `np.interp` operation; final finiteness checks also miss it.
- B3, active complement: for active panel `[-1,1e-17]`, target zero and accepted
  order `1e-20`, the fraction rounds to one. Forming `1-fraction` erases the
  independently represented gap `right-target`. The left weight is about
  `1e-20` instead of `1.001e-17`. A coupled-state probe returns component defect
  `-1.3862943611198906` instead of approximately `-1001.38629436112`, and the
  cell range excludes the reference. Protecting the old-panel complement did
  not protect this sibling active-panel conversion.
- Applied action, not a numerical fix: freeze the rejected bytes, retain all
  three failed source reviews and preserve all six independent RED cases. Do
  not initiate another repair loop, commit/push as approved, narrow the accepted
  mathematical input/order scope or claim A1 completion without an owner decision.
- Verification: all 33 indexed review files match their hash/size handles; the
  original fixture and all 125 current source inputs still match the frozen
  snapshot. Parent execution of the archived `test_independent_blockers.py`
  independently reproduces six assertion failures, no errors, actual exit one.
  It uses `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu uv run --frozen --extra dev
  python -m pytest`, the scratch review test path, `-c pyproject.toml -q -p
  no:cacheprovider --tb=short`, and a separate parent JUnit destination. Exact
  binary-input references and original executable driver are retained in
  `final-code-review-rejected.tar.gz`; raw parent XML is preserved byte-for-byte
  in `final-blocking-parent.xml.gz`, with hashes in
  `final-review-preservation.json` in the
  [trajectory audit bundle](../validation/audits/a1-trajectory-defect-2026-11-11/README.md).
- Test boundary: the existing 81 focused CPU tests pass in the reviewer process;
  selected 90 panel-weight comparisons also pass. The default CPU full suite
  passes 981 cases with one performance deselection, actual exit zero and exact
  source/identity checks. These default tests do not contain the six additional
  independent RED assertions and do not override the rejection. At the rejection
  checkpoint, completion of the current CUDA full suite was not yet verified.
- Subsequent historical 92ff completion readback: both CPU/CUDA full processes
  exit zero
  with 981 passing cases and one performance deselection each. Both XML identity
  sets match the frozen collection; all 125 declared source inputs are unchanged.
  `rejected-checkpoint-results.json` and `pre-arithmetic-checkpoint.tar.gz` members
  `cpu-readback.json`, `cuda-readback.json` and `executions.json` retain this
  completed evidence for the rejected source. The mutable root filenames are
  now current-candidate completion targets, not historical references. This
  evidence neither supersedes the six independent RED failures nor authorizes
  another implementation cycle, commit/push, scope deferral or A1 acceptance.
- Remaining/prevention: review the arithmetic/input-domain contract before more
  implementation. Audit library primitives as well as explicit scalar formulas,
  and compute/check sibling gap and scaling paths independently. No tolerances,
  global precision, core runtime, dependencies or public API were changed. Working
  arithmetic verification, total numerical certification and A1 acceptance remain
  false; no scope deferral or phase boundary change is implied by the pause.

- Owner-authorized next step, 2026-11-11: "okay, agreed. proceed with your
  recommendation." This approves the proposed arithmetic-contract review followed
  by one new bounded repair/review cycle, not the existing rejected bytes. The
  [contract](../validation/audits/a1-trajectory-defect-2026-11-11/arithmetic-contract.md)
  passed independent technical review before numerical implementation. Its exact
  reviewed draft and six hash-verified evidence files are archived separately;
  technical clearance is not source approval. Canonical failures must recover
  their reference values, while
  genuinely unsafe intermediate arithmetic must fail explicitly. No tolerance
  relaxation, global precision promotion, mathematical-domain exclusion or
  history truncation is authorized. `pre-arithmetic-checkpoint.tar.gz` and its
  layout preserve the previous mutable evidence and original 54-file manifest.
  A further blocking final-source review must again stop for an owner decision.

- Expanded regression evidence: the independently designed 36-case matrix has
  13 actual RED failures and 23 passes on the pinned rejected source, exit one.
  The failures comprise six canonical and four signed-state inaccurate-return
  cases, plus three masked intermediate-underflow policy failures. The latter
  do not demonstrate gross final-value error: a dominant retained term can hide
  a tiny product while leaving the rounded final value unchanged. The compact
  `arithmetic-test-design.tar.gz` also preserves its earlier scratch test-design
  run without combining counts or treating an overly strict failure control as
  a scientific accuracy defect.
- Default-suite port: `tests/unit/test_volterra_arithmetic.py` preserves all
  reference/test logic, inputs and tolerances, with only the current-module
  loader, formatting, a local variable rename and dispatch typing adapted.
  AST comparison verifies those declared non-numerical adaptations. Parent RED
  independently reproduces 36 tests / 13 failures / no errors, actual exit one;
  exact pre/post bytes for 126 inputs match. Raw XML, source, observations and
  the port record are retained in `arithmetic-regression-red.tar.gz` and its
  index. At that RED checkpoint, the fresh implementation context was applying
  the approved shared repair; GREEN, source approval and full suites were still
  pending. Subsequent results are separate steps below, not changes to the RED.

- Implementation GREEN checkpoint, before source approval: a fresh context
  changes only the benchmark
  helper. Shared checked float64 primitive operations replace opaque interpolation
  and contractions; both endpoint distances and the active-panel right gap are
  independent, stored-knot identity is exact, and products are checked before
  fixed-order accumulation. All 117 focused cases pass in parent CPU/CUDA-
  configured processes, actual exits zero; only the helper differs from RED.
  All 126 source inputs match the frozen four-file reference
  `bf3d4fa06026f4f2efedcf9d899c75ebca53cdbd`. This is GREEN evidence, not approval.
- Pre-review parent probe, classification then pending: constant represented states
  `[1e20,1,-1e20]` with a feedback row of ones produce zero point defect/cell
  range versus an exact-input component defect about `-1.1283791670955126` at
  time one, half order. This is fixed-order cancellation, not underflow, and is
  outside the 117-case focused selection. The final reviewer has the exact raw
  probe and must adjudicate the contract's conditioning limits versus a blocking
  point-diagnostic failure without using false certification flags as an excuse.
  No further numerical changes occurred during that one final review. At that
  stage current-source full suites were deliberately deferred until its verdict,
  to avoid spending resources on another potentially rejected source. Prior 981-case successes
  remain historical evidence for the rejected 92ff checkpoint only.

- Final scoped source review: `bf3d4fa06026f4f2efedcf9d899c75ebca53cdbd` passes
  with empty security/logic-error lists. All six canonical recoveries independently
  succeed; 117 focused CPU cases and additional mathematical/range probes pass
  or reject unsafe intermediates as required. All 38 indexed review files,
  four pinned sources, 126 archive inputs and the fixture match their handles.
  The parent cancellation probe is explicitly classified as SCI-003, not ignored
  or repaired by this cycle. Current CPU/CUDA full suites subsequently completed
  with 1,017 passes, one deselection and confirmed exit zero each. Exact selected
  XML identities and all 126 input hashes match. The separate VAL-012 artifact-
  linkage correction is implemented and exercised; corrected independent artifact
  review passes separately;
  no scientific acceptance or numerical certificate is implied.

## SCI-003 — Cancellation-dominated private diagnostic values

- Status/impact: open research/arithmetic limitation. A constant represented
  trajectory `[1e20,1,-1e20]`, feedback row `[1,1,1]`, half order and time one
  returns zero component defect and cell range `[0,0]`, versus exact-input
  defect about `-1.1283791670955126`. These are substantive false-zero values,
  not evidence of exact zeros or a small scientific residual.
- Evidence-backed cause: checked products are finite and no underflow occurs.
  Fixed-order addition loses the unit term between opposite `1e20` contributions.
  The exact feedback is one; the absolute product sum is
  `200000000000000000001`, condition number about `2e20`, and its product with
  binary64 unit roundoff about 22,204.46. The unit loss is about `5e-21` of that
  absolute sum and consistent with ordinary floating addition. Reordering the
  unchanged components restores the term. Range guards do not control this error.
- Review distinction: the independent reviewer classifies this as nonblocking
  only for the owner-approved contract, which explicitly permits checked fixed-
  order additions and excludes conditioning/summation/global relative-accuracy
  bounds. That does not endorse the inaccurate value, use false certification
  flags as an excuse, narrow the mathematical domain or close A1. A source pass
  is not scientific residual acceptance.
- Applied action: preserve the exact binary inputs, output, high-precision
  reference, condition calculation and permutation controls in the final source
  review archive. Disclose the limitation in the audit/programme/changelog. No
  cancellation-robust numerical redesign is applied in this bounded cycle.
- Verification: parent independently reproduced the false zero; the final
  reviewer reproduced and quantified it while all 117 focused cases passed.
  `arithmetic-code-review.tar.gz` retains `conditioning-probe.json`, executable
  checks and raw results. This probe is not one of the 117 focused tests and
  must not disappear behind their passing count.
- Remaining/prevention: require independent conditioning and evaluation-error
  control before treating these outputs as a scientific acceptance oracle.
  Distinguish computed zero coefficients from exact algebraic zeros, intermediate
  range safety from cancellation accuracy, and ordinary floating model ranges
  from certified enclosures. A cancellation-robust redesign requires separate
  authorization; all arithmetic/certification flags and A1 acceptance stay false.

## VAL-012 — Delivery documentation and completion-readback linkage

- Status/impact: resolved at the artifact-integrity gate. The initial artifact
  audit failed independently of the scoped source pass; its delivery block was
  retained until corrected checks and independent re-review passed. This is not
  scientific acceptance. The reviewed numerical source, its 126 inputs and
  1,017 selected tests are not
  changed by this issue. No numerical repair or source-approval cycle is reopened.
- Symptoms: DOC-01 found an old paragraph calling rejected `92ff...` the current
  source. EVID-01 found that finalization could accept a readback with an incorrect
  `xml_sha256` and zero `source_inputs_verified`; the verifier did not inspect
  current backend readbacks. Regenerating checksums would preserve those fields
  without validating their meaning.
- Evidence-backed causes: chronological documentation retained an unqualified
  current-tense label after a later source freeze. Completion checks validated
  selected counters/flags independently but omitted their cross-record binding
  to raw XML, the complete source snapshot and the originating process plan.
  The audit executed the existing readback predicates against deliberately bad
  synthetic metadata and observed acceptance. This is controlled integrity-test
  evidence, not a fabricated candidate run or a candidate numerical failure.
- Preservation: all 41 indexed audit proof files, the ten frozen review inputs
  and original pre-fix CPU readback are hash/size checked and archived in
  `document-integrity-audit-failed.tar.gz`; the preservation index retains the
  original failed verdict. The historical 54-file numerical manifest is separate
  and unchanged.
- Applied documentation fix: label 92ff explicitly as the rejected historical
  panel-repair checkpoint; cite historical 981-case readbacks inside
  `pre-arithmetic-checkpoint.tar.gz` rather than mutable root targets. Mark
  earlier pending steps as chronological checkpoints instead of current status.
- Evidence fix: both completion CLIs use the same standard-library validator for
  XML digest/counts/exact identities, every source-input hash/count, reviewed tree,
  canonical types and backend/session/PID/exit-path observations. Actual local
  exit bytes are preserved and hashed. Artifact-only replay checks the preserved
  bytes; current-source mode additionally checks actual local inputs/exit files.
  A parent portability RED exposed a remaining unshipped scratch-producer
  dependency in the controlled tests; the byte-identical capture CLI is now
  included in the audit and selected through a repository-relative path.
- Verification: initial strict pre-fix controls have four failures/two passes,
  exit one. Expanded original-script replay has 101 failures/23 passes, exit one;
  corrected scripts have 124 passes/exit zero and catch all 19 guard-removal
  mutants. Parent portability RED fails once as expected, then all 125 controlled
  metadata/portability tests pass. These synthetic controls are not candidate
  numerical outcomes. The early no-op XML mutation and corrected replay are
  preserved honestly, not counted as an implementation regression. All 49 indexed
  implementation files and 19 raw mutation XML files are verified in the 87-entry
  `completion-integrity-repair.tar.gz` archive/index. Fresh actual CPU/CUDA
  readbacks pass the new capture checks, retaining 1,017 exact selected identities
  per backend, all 126 inputs and actual zero exits. Both verifier modes pass,
  including isolated portable replay without local source files. Corrected
  independent artifact review passes with empty security/logic lists, 125 passing
  controls and 43 additional copied-record CLI controls. Its 313 indexed proof
  files and the exact original 85-file manifest are hash/size checked and retained
  in `corrected-artifact-review.tar.gz` and the preservation record. Status-only
  clarifications/new review artifacts/fresh delivery records are separately bound
  after that frozen review; no numerical helper, test, contract or acceptance
  criterion changes. Passing numerical suites alone did not supply this clearance.
- Remaining/prevention: validate the relationships represented by metadata before
  checksumming it, including canonical types and exact process/source identities.
  Require actual process completion, not a printed pytest summary. Preserve failed
  audits and synthetic-control labels; record integrity, scientific acceptance
  and numerical source approval remain separate gates. SCI-003 and all negative
  certification/A1 flags remain unchanged.

## VAL-013 — A2 protocol completeness before freeze

- Status/impact: initial and first corrected preparation reviews failed. This
  blocks protocol freeze and implementation, not a measured fitter or selector
  result; no A2 numerical
  implementation or JAX fitting/application/gradient measurement has occurred.
- Symptoms: DECISION-01 leaves unsupported requests versus incompatible/missing
  candidate evidence ambiguous. DECISION-02 leaves duplicate content identity
  undefined. DECISION-03 overstates why any unknown assessment prevents a minimum:
  an unknown higher-ranked candidate need not displace a known minimum, although
  mandatory exhaustive-assessment policy can still require indeterminate status.
- Additional findings: B1 lacks exact fixture/evidence identities, notably the
  real-fit signal (zero input would qualify at zero budget). B2 lacks complete
  shape-safe comparison/analytic-input-gradient/float32 eager-JIT policy mappings.
  B3 lacks closed canonical record/array/producer/source bindings; the existing
  private fit object does not itself store order, timestep or horizon. B4 leaves
  resource accounting and bounded partial evidence after export/timeout exhaustion
  insufficiently specified. The initial draft also has a broken internal anchor.
- Evidence-backed cause: sound finite mathematics and high-level selection rules
  were written before closing observable request/candidate/failure precedence,
  schema, exact fixture coverage and resource-controller contracts. An exhaustive
  policy requirement was incorrectly presented as a universal rank-uncertainty
  theorem. These are specification defects, not proof that the numerical fitter
  fails a tolerance.
- Preserved evidence: both original `passed:false` verdicts for draft SHA-256
  `331d0d72ade81c242475bc9b3b1a0d37b4c3393251661e5b113bcbb0d48dd066`, all
  23 mathematical and 35 validation indexed proof files, original indexes/final
  readbacks, parent planning checks and the reviewed draft are hash/size checked
  in the 66-entry `initial-contract-reviews-failed.tar.gz` and preservation index
  under `docs/validation/audits/a2-preparation-2026-11-11/`.
- Positive boundary: exact signed-GL/prefix-increment algebra, exact-head bound,
  normalization and singleton statements pass independent mathematical review.
  Synthetic rational controls support the proposed resource caps for the exercised
  schedules only; no universal watchdog/export-size guarantee or production-fit
  observation follows. Scalar and stacked planning cases do not validate JAX AD,
  accelerator execution, measured peak memory or A2 completion.
- Historical first correction: the revised draft specifies failure
  precedence, ID-free candidate keys, exact cases/comparison targets, raw
  producer/dtype/byte/source bindings and bounded incomplete-evidence outcomes.
  Static JSON enumerates 544 request cases, 10 parser and 4 identity controls,
  20 comparison policies and 24,128 planned evidence slots; these are fixture
  inventories, not executed worker counts. No real-fit arrays are fabricated.
- Parent verification: seven packet hashes/sizes, all 66 original archived files,
  unchanged mathematical-section bytes, 125 numerical inputs and index entries
  checked. The captured stdlib planning script replay exits 0: 352 synthetic
  cases/648 candidate assessments, 5,368 output and 5,368 analytic-gradient
  components, and 648 exact quadratic-FD equalities. All are planning witnesses,
  not production selector/AD/FD measurements. Original correction scripts,
  intermediate attempts, raw outputs and exits are preserved in the 74-entry
  `corrected-draft-planning-evidence.tar.gz` and its preservation index. Ruff,
  23 documentation tests, strict MkDocs and the actual built anchor pass for
  that rejected snapshot; none establishes cross-record representability.
- First corrected packet rejected: both independent verdicts remain
  `passed:false` for draft SHA-256
  `6fd2868bd31e1bcc49af73834540c1858343b15fec0bb1075fa706d0ff90c1df`.
  MATH-REREVIEW-01/V-B3-01 expose unrepresentable candidate-free full-history
  observations and fit-only configuration/producer types. MATH-REREVIEW-02 finds
  contradictory `not_applicable` reasons. MATH-REREVIEW-03/V-B4-02 finds admitted
  unsupported N=18 failure shapes forbidden by the export schema. V-B1-03 finds
  empty/duplicate-list mutations targeting a field already removed by expansion,
  with stale occurrence inventories. V-B4-04 finds missing typed timing/phase
  records; V-B3-05 finds standalone slots excluded from root schema routing;
  V-B3-06 finds undefined binding-digest payloads and missing invocation/receipt
  crosslinks. V-S01 identifies absent private output ownership/no-follow/overwrite
  invariants: a scratch symlink satisfies the lexical path grammar while escaping
  the request root. This is a missing security contract, not an exploit against
  nonexistent implementation or the current numerical library.
- Revised evidence preservation: all 60 mathematical and 34 validation indexed
  artifacts, both failed verdicts/indexes, original reviewed packet/status bytes
  and parent quality receipts are verified in the 124-entry
  `corrected-contract-reviews-failed.tar.gz` with preservation metadata. The indexed
  symlink is captured as hash-checked readlink UTF-8 data with an explicit
  representation map; it is neither followed nor recreated in the archive.
- Additional cause/prevention: scalar mathematics, case counts and checksums were
  checked without exercising the complete export schema for every required
  success/failure role. Require labelled static representability witnesses for
  each artifact type, status/reason combination, invalid occurrence, unsupported
  shape, candidate-free observation and timeout receipt; test cross-file semantic
  relationships and an acyclic binding recipe, not just individual schemas.
- Final bounded preparation correction completed as a proposal: nine packet
  files include explicit record variants/routing, occurrence-safe expansion,
  reason-state rules, phase/binding/ownership contracts and labelled static
  representability fixtures. Parent replay exits 0 for 17 hypothetical bundles,
  37 structural negatives and 12 semantic tamper controls. The current inventory
  is 560 cases/840 raw descriptors/24,336 quantity slots, retaining the original
  544 case IDs and byte values of all 832 prior raw descriptors. All 80 indexed
  correction artifacts plus their index/capture source are preserved and read
  back in the 82-entry `final-preparation-correction-evidence.tar.gz` and its
  preservation record. This is not independent approval or actual worker data.
- Residual resource blocker: under the unchanged conservative preflight rule,
  eight required float32 tau=1 FIT-REAL templates are refused as
  `reservation_exceeded/indeterminate`. Maximum initial+new-index transaction
  plus receipt allowance is 792,318 bytes versus the fixed 524,288-byte failure
  reservation; a complete metadata template reaches 36,687 nodes versus 32,768.
  Input maximum 183,673 bytes and single-index maximum 446,741 bytes alone would
  not expose the whole-transaction/node conflict. All comparison/fit-check
  dispositions remain present. These are static conservative templates, not
  observed runtime/hardware infeasibility or numerical fitter failures. The final
  independent classification below supersedes any suggestion that the all-complete
  node excess is unavoidable. No cap increase, omitted evidence or case
  substitution is authorized to make them pass.
- Final independent verdict: `contract_consistency_passed:true`, `passed:false`,
  `freeze_ready:false`, no retained security/logic findings, and resource blocker
  `R-FINAL-RESOURCE-01`. All 17 prior blocker IDs are adjudicated as resolved at
  specification level; B4 resource readiness remains blocked. The reviewer
  independently checks all 560 cases, 64 candidate-free full bindings, 17 roots,
  new rank/combined controls and 15 non-no-op semantic mutants with refreshed
  outer digests. Actual worker execution/security enforcement is not established.
- Independent resource result: the all-complete model incorrectly completes
  mutually exclusive nonselected AD/FD branches and unused quantities. Its
  36,687-node count is a conservative template, not a node lower bound. However,
  two deliberately smaller-than-valid full-inventory index skeletons still need
  614,748–617,184 bytes across the eight cases, excluding process interiors and
  receipts. This lower bound exceeds 524,288 regardless of winning candidate,
  array values or timing. The duplicated full-index representation and atomic
  old/new coexistence therefore obstruct admission; this is not observed fitting
  failure, numerical unattainability, hardware infeasibility or measured memory.
- Final review preservation: all 51 indexed artifacts, original failed-readiness
  verdict/index, exact reviewed packet/status bytes and input manifest are
  verified in the 66-entry `final-independent-review.tar.gz`; the original verdict
  is also copied byte-for-byte to `final-independent-review.json`, with a separate
  preservation record. Verdict SHA-256 is
  `be2b12396554750de96b53a17aebbc96864ce61c61d48f001c8c49d42d4e692c`.
- Stop boundary: both automatic preparation correction cycles are used. Exact
  mathematics, thresholds/FD steps, tiny domain and diagnostic caps stay fixed;
  all 125 numerical inputs and index entries are unchanged. Final independent
  review is complete; an owner decision is required before any further correction
  or implementation. Options are a separately reviewed representation redesign
  retaining caps, an explicitly approved metadata-budget revision after valid
  export accounting, retaining the paused draft, or an explicitly narrower
  failure-only stage with positive obligations still open. None is applied by
  this review. No JAX worker, numerical implementation or scientific acceptance
  is inferred from the static checks.
- Owner authorization, 2026-11-12: the current user relays Davian's Telegram
  reply “Okay, go ahead with your recommendation.” to the recommendation for one
  narrowly scoped metadata byte/node budget revision after valid export accounting
  and independent re-review, preserving every case/check and numerical tolerance.
  `metadata-budget-revision-authorization.json` records this additional single
  preparation cycle and its exclusions. The historical two-cycle stop and failed
  readiness verdict remain unchanged; no freeze or numerical implementation is
  authorized, and no solver/per-state/deferral/Phase B decision is inferred.
- Pre-revision RED: all eight necessary index-byte-admission controls fail with
  actual pytest exit 1 against the unchanged proposal. They compare the independent
  smaller-than-valid byte lower bounds to the current reservation; a later GREEN
  establishes only that necessary condition, not full valid export feasibility.
  The nine packet files are snapshotted before edits; all 125 numerical pins,
  complete index and protected historical evidence remain unchanged. Revised
  budgets must be justified by valid selected/unselected, process/receipt and
  byte/node accounting, not by the impossible all-complete template alone.
- Authorized attempt outcome: the budget revision was not completed. All nine
  packet files remain byte-identical and caps stay at 524,288 index-transaction
  bytes and 32,768 nodes. The scratch accounting produced 512 projections,
  including 48 agreeing selected profiles, but a representative seven-record
  projection leaves 372 conditional artifact/link obligations unclosed. It also
  omits required payload/copy, observation, binding, log and manifest accounting.
  An arbitrary ancillary allowance and discretionary padding are not derived
  bounds. Exploratory 1 MiB/65,536-node figures are explicitly rejected as
  unsupported, not applied and not a new proposal. All eight necessary admission
  tests remain RED; no GREEN or full export-admission pass is claimed.
- Independent assessment confirms this accounting/harness failure, not a new
  packet/schema defect or numerical impossibility. The path-presence gate is only
  a necessary refusal check; its hard-coded failed flag is not a complete semantic
  validator. Merely adding filenames would not establish bindings or lengths.
  Metadata accounting can be completed in principle with constrained symbolic
  lengths/identities or labelled static witnesses, without actual fitter arrays,
  JAX execution or an inherently required representation redesign.
- Separate budgets correctly: referenced standalone payloads do not all belong
  inside the index-only transaction reservation. JSON-node limits apply per
  record, including any inline payload copies, not cumulatively across files.
  The separate live/cumulative 16 MiB whole-request ledger must cover every
  required emitted copy, conditional payload, observation/binding, log, manifest,
  receipt and temporary/coexisting file. Routing is an allowlist, not a mandate
  to emit every permitted route in every branch. The incomplete projection's
  conditional result has 51,560 recounted nodes, not a proved full-export maximum.
- Attempt/assessment preservation: 64 attempt artifacts and all 65 archive members
  are independently verified in `metadata-budget-accounting-blocked-evidence.tar.gz`.
  The assessment's 56 indexed artifacts and its 58-member proof archive are verified
  separately. Raw failed harness launches and the isolated genuine eight-failure
  RED rerun remain distinct. Earlier mutable script versions were not all captured
  before execution; final indexed scripts are final references, not invented
  historical snapshots. Ambient cron heartbeat changes prevent any whole-profile
  equality claim; no reviewer-write or numerical-defect inference follows.
- Current stop: the additional authorized bounded attempt and independent
  assessment are complete, but no budget revision is delivered. All 125 numerical
  files, 18 protected files, packet/index/HEAD and scientific requirements are
  unchanged. Completing the finite linked ledger and obtaining sufficient bounds
  requires a new owner decision; no further automatic writer cycle, cap increase,
  implementation, protocol freeze or scope deferral is inferred.
- Resumed scope, 2026-11-12: the owner's “okay, let's pick up from where we left
  off.” authorizes one further bounded ledger-completion/metadata-only revision
  and independent-review cycle. `metadata-ledger-continuation-authorization.json`
  records the exact message and exclusions. Previous failed attempts and the
  paused outcome remain unchanged historical records. Fresh capture verifies all
  nine packet files, 125 numerical inputs, 21 historical/authorization files and
  the complete index/HEAD. Eight necessary admission assertions fail with actual
  pytest exit 1; 23 docs tests, Ruff, strict docs and whitespace pass. These are
  baseline metadata/development checks, not numerical or full-admission results.
  Complete every required conditional link/copy and derive bounds without ad hoc
  allowances before changing metadata limits; keep all nonmetadata limits and
  scientific requirements fixed. Parent-owned status updates are separate from
  the protected proof inputs; no live profile-tree equality is required.
- Resumed author outcome: `metadata-ledger-author-outcome.json` remains
  `BLOCKED_STOPPED_NO_BUDGET_PROPOSAL`. The 505-record representative includes
  372 payload terms and 1,389 links, but 131 checked obligations remain: 42
  same-mode descriptor identities, 18 perturbed-order configuration identities,
  69 pass statuses inferred from presence without conditional predicates, and
  unclosed exit-confirmation/safe-output conditions. These are faults in the
  scratch proof, not evidence of numerical failure or a new packet defect.
  The 1,392 structural profile exercises cover all 560 case identities but prove
  zero complete valid exports; selected/no-qualifier labels are not outcomes.
- Regression limits: the archived seven-record model gives four genuine RED
  failures; the expanded model passes those four necessary structural checks.
  The semantic gate still exits 1. Eleven negative controls have five failures
  and zero errors: uncharged log, manifest, old-index and temporary bytes, plus
  unsupported `[18]` changed to `[17]`, survive. Component names and guard counts
  do not prove correct sums, lifecycle accounting, descriptor equality or shape
  retention. No full-admission GREEN, sufficient bound or dynamic refusal
  guarantee is delivered; no new budget is justified or applied.
- Advisory distinction: the independent obligation design is not approval of
  the parallel author's model. A bounded initial/required-qualifier-checkpoint/
  final schedule is defensible under the existing cumulative-write contract;
  authenticate/account for every retained artifact on finalization and preserve
  the declared fault checkpoint. The author's 374-generation, every-array
  schedule and its large cumulative estimate are not unavoidable lower bounds
  or proof that the unchanged 16 MiB cap is impossible. Advisory node bounds are
  placement-specific, not necessary minima over every allowed layout. Its weaker
  614,416–616,852-byte index skeleton bounds independently retain the 512 KiB
  obstruction without replacing the earlier historical bounds.
- Resumed preservation: parent verifies 180 author-indexed files (13,364,848
  bytes), all 210 author archive members, 201 advisory-indexed files (6,830,424
  bytes) and all 218 advisory archive members. Excluded index-build/verification
  captures are additionally preserved, not mislabeled as originally indexed.
  Exact pre-run sources and failed harness launches remain separate from actual
  behavioral RED and structural GREEN. All nine packet files, 125 numerical
  inputs, 21 historical files and complete index/HEAD remain unchanged. Future fit
  arrays and full observations remain unobserved; no model repair follows this
  bounded stop.
- Independent resumed assessment: `overall_passed:false`,
  `accounting_complete:false`, `cap_revision_justified:false`, `stop_correct:true`.
  Fresh unchanged-source replay reproduces the semantic exit 1/131 conditions,
  negative-control exit 1/11 tests/5 failures/0 errors, and necessary structural
  four-RED-to-four-GREEN transition. Independent raw-record reduction, not the
  author's semantic helper, verifies the 42/18/69/1/1 categories. Unrelated
  symbolic descriptors are unproved equalities, not actual measured mismatches.
  Parsing 1,392 profiles and checking all 505 record-node counts does not rerun
  every model or prove complete exports; even zero findings from the partial
  checker are not complete validation.
- Stronger reviewer negatives: wrong-shape comparison operands and cross-mode
  identity references retain pass, and selected-with-unknown-assessments plus
  unconfirmed exit is accepted. These are deliberate invalid-success fixtures,
  not genuine successful exports. The original four accounting mutants change
  summary totals; they demonstrate missing independent equation checks, not
  necessarily omitted terms in the pristine formula. A new control removes the
  actual affine stdout term and recomputes its bounds while keeping the component
  name; the audit still accepts. Presence/guard-count predicates cannot support
  semantic completion or a sufficient ledger claim.
- Final resumed preservation: all 102 review-indexed files (1,654,964 bytes), all
  124 review archive members, original assessment and exact pre-run replay sources
  are verified in `metadata-ledger-independent-assessment-evidence.tar.gz` and its
  preservation record. Assessment SHA-256 is
  `0491f645d40f5b28c12f3b5ba58d8eb4c3af46ddd6833c3fb317336258c463a1`.
  The reviewer verifies 162 protected repository files and all 210 author/218
  advisory files against the frozen review manifest, plus complete index/HEAD.
  The parent independently verifies those protected inputs and the new archive.
- Current resumed stop: the authorized bounded attempt and independent assessment
  are complete as a failed investigation, not a delivered budget revision. Index
  transaction stays 524,288 bytes, nodes stay 32,768 per record and whole evidence
  stays 16 MiB live and cumulative. Existing caps still obstruct required cases;
  leaving them unchanged is not an adequacy claim. Neither schema redesign nor
  physical fitter execution is established as necessary by these method failures.
  No implementation, freeze, numerical/scientific acceptance, scope deferral,
  commit/push or further repair cycle is inferred. Further work needs a separately
  authorized finite acceptance target and stop rule.
- Prevention for this method failure: first close one end-to-end conditional
  semantic trace and independently recompute its whole-ledger equation; reject
  arithmetic-term omissions and invalid-success fixtures before expanding profile
  counts. Preserve a defensible bounded checkpoint schedule rather than treating
  a chosen every-array rewrite schedule as an intrinsic resource lower bound.
- Single-trace scope authorized, 2026-11-12: “alright, go with your
  recommendation.” approves one end-to-end conditional trace before any bulk
  expansion. `single-trace-authorization.json` records the fixed existing STATIC
  `REF-PASS-f64-GL` witness (N=3, two provided synthetic candidates, 22 slots),
  semantic/byte checks, exclusions and stop boundary. The rejected broad model is
  not repaired or restarted; all its original outcomes remain failed.
- Applied method change: a fixed-case audit checks the actual schema, supplied
  inventory, payload/descriptor/source/mode/alpha/dt identities, exact static
  payload equations, qualification/minimum supplied rank and conditional
  completion/phase/publication order. The builder derives a closed-form byte sum;
  a separate consumer replays every canonical body through a byte-valued state
  map, exclusive writes, old/new coexistence and one atomic index replacement.
  Per-body charges and actual live/cumulative totals are compared, not merely
  component names. Manifest's own bytes, both nonempty illustrative logs,
  overwritten initial index and final staging bytes are included; rename is not
  double-counted. Input/index/result snapshots are checked before their own write.
- Vertical evidence: seven source-captured RED/GREEN stages end at twenty passing
  checks. The first RED is a collected absent-module assertion with end-to-end
  behavior assertions behind it; later REDs are genuine missing rejection rules.
  Controls retain names and recompute totals after dropping a positive charge;
  semantic controls rebuild outer digests and false-payload controls also rebind
  returned descriptors. Invalid selected-with-unknown/exit-unconfirmed/wrong-rank,
  mode/alpha/shape mismatches, false exact outputs/gradients/bounds, inconsistent
  phase intervals and publication-before-binding are rejected. Isolated code/data
  replay passes twenty; controlled missing authority input fails closed.
- Self-check sizes are concrete, not universal bounds: 36 JSON records plus two
  raw logs, 22 slots, two index generations; 70,427 cumulative bytes, 62,970 final
  live bytes, 65,229 peak live bytes and 19,070 bytes for the chosen index/process/
  export/control transaction. Largest record is 1,204 nodes, depth at most nine.
  All nine packet files, 125 numerical inputs, 44 historical/authorization files
  and complete index/HEAD remain unchanged. Initial review outcome follows; no
  metadata cap change, FIT-REAL evidence, worker execution or broader feasibility
  follows.
- Limits remain explicit: process/ownership/timing and operation/bit statistics
  are inherited hypothetical representability fields, not runtime observations or
  a proof of a worker's counter/security behavior. The external repeated-1 digest
  is labelled hypothetical. Read-only Fraction checks concern only the supplied
  N=3 static identities, not a production reference worker or general numerical
  acceptance. All certification/implementation flags remain false.
- Initial fixed-trace independent review: rejected for `FIXED-TRACE-ROUTING-01`,
  with no security findings. Changing `records/binding-host-0.json`'s
  `binding_id` to `host-1` while rebuilding outer hashes, binding links and byte
  snapshots is accepted, despite the frozen
  `records/binding-{binding_id}.json` route. Root schema and lexical safety are
  enforced, but `x-artifact-routing`'s path-to-ID equality is not. This can retain
  an incorrectly identified binding and duplicate a sibling ID without breaking
  any checked content digest; hash consistency is not route identity.
- Evidence/impact: all twenty supplied tests and seven archived RED/GREEN stages
  reproduce. The reviewer independently checks the positive 36-record/22-slot
  exact payloads, qualification/rank, links, phases and materialized byte ledger,
  but its targeted run has 22 passed / 1 failed / 0 errors, actual exit 1. The
  routing test fails with `DID NOT RAISE`; a separate parent process confirms the
  checker accepts the malformed trace, then actually exits 1 on the required
  route assertion. This rejects scoped acceptance without discarding the valid
  finite positive arithmetic/size evidence or implying a numerical core defect.
- Preservation before correction: verified 210 author-indexed files (5,388,615
  bytes), 452 reviewer-indexed files (15,627,201 bytes), all 170 protected
  repository-union files and full index/HEAD. All 220 author and 457 review archive
  members are read back byte-for-byte. See
  `single-trace-failed-review-preservation.json`,
  `single-trace-rejected-author-evidence.tar.gz` and
  `single-trace-independent-review-failed-evidence.tar.gz`. Original verdict SHA
  is `2c52c108aa98c89cdb882e4194c75898fc4d3f7f4bd474dc5cc1a7b04fc4055a`.
  The first RED reaches only the collected module-absence assertion; nineteen
  later RED cases exercise missing rejection behavior. Pre-spawn source capture
  and matching isolated replay do not attest against historical between-capture
  and execution changes; no OS/process attestation is claimed.
- Bounded correction decision: `single-trace-routing-correction-plan.json`
  permits one fresh third-context fix in a new scratch copy, not an original
  source overwrite or broad-model restart. Enforce the existing route identity
  for this trace's bindings, exercise fully re-signed colliding and noncolliding
  wrong IDs as behavioral RED/GREEN, preserve the identical positive artifact and
  re-review. No case, schema, allowlist, cap, mathematics or tolerance changes;
  this is an execution decision within the original target, not a new owner
  authorization. Stop for the owner if this correction/fresh review fails.
- Routing correction self-verification: the new scratch candidate adds only
  `binding_id == 'host-' + suffix` inside the existing three-binding loop; its
  fixed lookup already supplies the corresponding filename. Exact implementation
  SHA is `4b28a7dfc4587dd6d6bfb63cdb5ea5362b0f5cf5ab3c1c9d31fe211465dfbd4e`.
  The original twenty tests and all frozen data/schema/protocol bytes are intact.
  Six fully re-signed duplicate/fresh-ID regressions across the three bindings
  first produce 20 passed / 6 `DID NOT RAISE` / 0 errors, actual exit 1; corrected
  and isolated suites each pass 26, actual exit 0. Schema, payload, digest, inline
  copy and ledger preconditions are checked before the verifier. Portable positive
  replay exits 0; controlled missing preserved trace exits 1. These are metadata
  audit/portability controls, not numerical-worker tests.
- Parent correction verification/preservation: independently verify all 235
  corrector entries (6,356,655 bytes), exact one-guard delta, original protected
  sources/HEAD/index and a fresh 26-test plus portable replay. All 38 emitted
  canonical byte strings and the 76,188-byte canonical trace container remain
  identical to the rejected source's valid positive artifact; ledger values above
  are unchanged. The container's journal/ledger wrapper is not itself a worker
  export. Preserve all 238 archive members byte-for-byte in
  `single-trace-routing-correction-evidence.tar.gz`; archive its three indexed
  symlinks as labelled regular UTF-8 readlink data with an explicit representation
  map. See `single-trace-routing-correction-preservation.json`. Fresh independent
  re-review outcome follows; no correction cycles remain under this plan.
- Harness limits preserved: two preliminary guard failures concern SHA-string
  versus size/hash-object pin representations and newline versus NUL Git-stage
  hashing. Their original source versions and terminal transcriptions are
  preserved, but original pre-run/raw-child captures are absent. Later captured
  failures are honestly labelled reproductions, not retroactive originals or
  scientific RED. Current audit runs execute the preserved pre-run source copies;
  no OS attestation is inferred.
- Corrected fixed-trace independent re-review: scoped PASS, with empty security
  and logic-error lists. The reviewer reproduces all 26 corrected tests and
  isolated copies (actual exit 0), the exact rejected source's 20 passed / six
  `DID NOT RAISE` / zero-error outcome (actual exit 1), and all 23 prior reviewer
  negative controls now passing on the corrected source. Six independently
  re-signed duplicate/fresh-ID controls reject specifically with
  `binding path-to-ID`; removing just that guard restores their acceptance and
  produces the exact rejected source. The preserved original counterexample also
  rejects for that relation, not an earlier schema/hash mismatch. Repeated runs
  and control families are not added into scientific case/test totals.
- Fresh positive proof again independently derives all 22 supplied exact payloads,
  verifies authoritative fixture/record links and physically replays every write
  and replacement. All 38 artifact bodies and the finite ledger are unchanged.
  Parent readback verifies all 373 reviewer-indexed entries (14,989,426 bytes),
  three readlink representations, final seal/binding sources/raw exits and all
  390 members of `single-trace-independent-review-passed-evidence.tar.gz`.
  Assessment SHA is
  `443ef0528876130b3db9146c494a829bc0a2ced28184681dad21938d3b9bd60b`.
  The 12 post-index seal/binding supplements are explicitly separate from the
  373-entry count. See `single-trace-passed-review-preservation.json` and
  `single-trace-outcome.json`. Every original failed verdict remains failed.
- Disposition: close only `FIXED-TRACE-ROUTING-01` in the one-case metadata audit
  and the separately approved finite-trace method increment. VAL-013's broader
  accounting/admission blocker, existing caps, all scientific gates and unanswered
  scope decisions remain open. No runtime/security/operation-budget proof,
  numerical implementation, protocol freeze, bulk expansion or A2/Phase A
  acceptance follows. Current source/proof bytes remain frozen; zero correction
  cycles remain under this plan.
- Prevention: validate frozen dynamic route identities in addition to closed
  schemas, lexical path safety and digest graphs. Test an allowed-looking wrong
  ID after recomputing outer bindings, and include a fresh noncolliding ID so an
  ID-uniqueness check cannot masquerade as the required path equality.
- Remaining/prevention: close observable contracts before implementation. Separate
  conservative completeness policy from mathematical rank facts; retain every
  supplied identity even when work/export/timeout prevents complete observations.
  Never fill missing data with zero, retune future gates from outcomes, relax
  scientific scope or relabel an initial failed verdict as approval.

- Separately authorized continuation: `CAP-after-qualifier-f64-GL` is now the
  single next static conditional trace, not an additional repair cycle for any
  previous rejected review. The parent has pinned 903 baseline files (901
  immutable), verified the prior completion/proof, and reconstructed the 13
  available exact quantities while preserving nine explicit unknowns. The
  candidate-0 qualifier does not authorize selection under the exhaustive
  policy. The `primitive_cap` checkpoint is hypothetical, not measured resource
  exhaustion. Test-first extension, complete conditional accounting and fresh
  independent review are pending. Both earlier complete artifacts/tests remain
  regressions; caps, numerical sources and eight FIT-REAL admission blockers are
  unchanged. See `cap-after-qualifier-trace-2026-11-13/authorization.json` and
  `parent-frozen-reference.json` in the A2 preparation audit.

## VAL-014 — Private receipt copies in a rejected-review archive

- Symptom/impact: a final recursive archive privacy check, after the scoped
  source/finite-proof pass, flags eight copied native Telegram receipts carrying
  personal delivery identifiers in the initial 457-member rejected-review
  archive. The case audit does not require those receipts. No identifier values
  were printed during diagnosis, and nothing was committed or pushed. The
  completion notice was premature: source correctness and hash-consistent
  preservation did not establish publishable evidence privacy.
- Evidence-backed cause: original `a2_capture.py:18-22` captures every top-level
  Python/JSON file, including a subsequently created delivery receipt. The outer
  author index excludes private filenames, but `a2_tdd_review.py:27-38` later
  verifies/copies the broader per-run manifest and its replay captures copy them
  again. The rejected-review index/archive therefore reintroduces eight private
  receipt bodies. Parent whole-index readback preserved them faithfully but did
  not detect the privacy violation; this is an evidence boundary failure, not a
  numerical defect or evidence that the metadata tests failed.
- Safe mitigation applied: verify the original archive SHA
  `264cfcc67365499f19705b4aef5c2ba0163e5bc0e3cb23d53ae263ff590a02c2`,
  preserve the same bytes in a new private durable directory outside the repo
  (directory mode 700, archive mode 600), verify readback, then remove only the
  own untracked repository copy. Preserve raw originals, original indices and
  failed verdicts; do not silently rewrite an immutable archive or substitute a
  sanitized one under its old hash/path. Exact quarantine location and
  field-name-only findings are in `single-trace-privacy-hold.json`.
- Verification/limits: structured JSON scan of the five current trace archives
  finds eight identifier-bearing members only in the rejected-review archive;
  the other four contain no matching identifier fields. This is a bounded scan,
  not universal secret detection. The original remains byte-identical privately;
  its old repo path is intentionally unavailable. Earlier archive-verification
  and exclusion claims are historical and must be read with this exception,
  not treated as a current publication clearance. All 26 audit checks and scoped
  source/finite-proof evidence remain valid; no cap/core/scientific change is made.
- Initial disposition (2026-11-12): publication/artifact completion on hold. Qualify
  `single-trace-outcome.json` and the earlier completion notice rather than
  relabel the preserved review. Zero correction cycles remain; request an owner
  decision before an explicitly sanitized derivative and a separate privacy/
  integrity review. No new repair, commit/push or numerical phase is started.
- Prevention: capture/replay only an explicit task input allowlist, not every
  scratch JSON. Keep delivery receipts outside that closure. Check nested archive
  bodies for sensitive identifiers before declaring completion or notifying it;
  filename filtering at one layer and matching checksums do not prevent
  transitive reintroduction. Separate private raw preservation from a labelled,
  independently checked publishable derivative with explicit exclusions.
- Subsequent owner-authorized remediation (2026-11-13): see the
  [privacy-remediation package](../validation/audits/a2-preparation-2026-11-11/privacy-remediation-2026-11-13/README.md).
  Its new archive omits the eight exact receipt members and retains the other
  449 bodies byte-for-byte. A new inventory exhaustively maps all 457 original
  members; historical manifests are retained with an explicit incomplete-replay
  warning, not silently rewritten. The original remains hash-identical with
  mode 600 under a mode-700 private directory, absent from its old repository
  path. The packaging helper and tests are new support tools, not changes to the
  fixed-trace audit or numerical core.
- Verification of the remediation: 23 packaging checks pass and all ten copied
  guard mutants are rejected. Fresh independent review passes with no security
  or logic blockers, independently reproduces the retained-member mapping,
  passes 23 packaging checks and rejects 16 CLI tamper controls. Its bounded
  recursive scan covers 100 explicit public files with no findings; the private
  positive control finds exactly the eight known members. The parent verifies
  all 198 indexed review files and 199 archived members, re-scans the indexed
  proof without disclosing identifiers, and confirms protected source/index
  identity. Reviewer harness failures and successful reruns remain separate.
- Current disposition: the specific sanitized derivative passes its scoped
  privacy/integrity gate. The unsanitized original is not publishable; no
  universal privacy guarantee, complete historical replay closure, scientific
  acceptance, broader A2 accounting, protocol freeze or delivery is inferred.
  Final QA and the allowed four-file publication/status delta are recorded in
  the remediation package. No commit/push or next-phase work follows this step.

## VAL-015 — Missing-bound validation order in the additional static trace

- Scope: the owner-authorized `REF-FAIL-f64-GL` accounting preparation increment,
  not the numerical library, production selector or a new scientific result.
- Symptom/evidence: the copied `06-missing-bound-red` test replaces the exact
  `S-0-B` payload with null and rebuilds outer links. Pytest records one failure,
  actual exit 1: `TypeError: 'NoneType' object is not subscriptable`, rather than
  the intended explicit semantic refusal. Source and raw XML are preserved in
  the [partial author package](../validation/audits/a2-preparation-2026-11-11/nonqualifier-trace-2026-11-13/README.md).
- Cause: `verify_trace` calls `check_completion` before `check_links`; qualification
  dereferences `bounds['values']` before the complete exact-slot/payload/shape
  predicate runs. The existing selected-case positive/tests do not establish
  this missing-bound refusal behavior.
- Proposed correction: move `check_links(trace)` before `check_completion(records)`
  in the scratch checker. The edited source is preserved, but **not verified**.
  The next `07-missing-bound-green` invocation triggered a one-shot execution
  approval request; no child started and no alternate route was attempted.
- Verification limits: earlier source passed the new positive case and the
  unchanged 26 selected-case regressions. Those results do not cover the latest
  ordering patch. Remaining semantic/ledger controls, positive artifact equality,
  independent byte replay, portable replay and independent review are unfinished.
  Parent readback verifies 132 indexed files, 133 archived members and 234
  protected files unchanged, plus Git-stage identity. The bounded field/token
  scan finds no matches; this is not final privacy clearance or independent review.
- Disposition/prevention: pause for the pending test-execution approval, retaining
  the same bounded task. Validate completeness and shape before arithmetic
  dereference; test malformed-but-schema-representable records after re-signing
  outer links. Do not count an unlaunched test, a prior-version pass or an
  exception of the wrong type as the current semantic gate passing.
- Approved resumption: after the owner replied “approved.”, a new copied-input
  process tests source
  `8b7049e3e5ee25f7266f0228cb72efc0726d55eb98d2a1699e3bab3263ef1c2a`.
  Both the positive no-qualifier and missing-bound refusal tests pass, actual
  exit zero, with source hashes unchanged. The parent verifies the XML and all
  13 archived members in `approved-resumption-evidence.tar.gz`; its index is
  `approved-resumption.json` in the additional-trace package. The former blocked
  attempt is not relabeled executed. This clears the focused ordering test,
  not the remaining full scoped checks or independent review.
- Final scoped verification: the checker bytes remain identical to that approved
  resumption. The final author selection passes 31 nonqualifier checks, the
  unchanged 26 regressions, one complete positive-artifact equality check and one
  frozen-authority derivation check. Five guard mutants and four missing-input
  plus one Python read-guard controls fail as intended. Independent review
  reproduces the 59 selected tests and rejects 33 additional posthoc controls;
  the parent replays those independent controls and exact/physical checks.
- Evidence closure: all 329 indexed independent proof files and 330 archived
  members are verified, including raw test receipts, exact 13-slot reconstruction,
  full per-body byte accounting and equality of the old 38-body positive artifact.
  A completion-message HTTP 400 did not erase the saved passing verdict; parent
  readback verifies it rather than treating transport status as test status.
  Incomplete reviews, approval pauses and reviewer-only lint-context failures
  remain preserved. New author helpers/tests are lint-clean, while 108 inherited
  archived checker/test findings and a capture-helper E501 remain disclosed.
- Current disposition: VAL-015 is fixed within the two named static-case checker
  scope. Only the one additional no-qualifier accounting increment is closed;
  no general malformed-input validator, operational/security or scientific
  certification, full-packet resource sufficiency, A2 acceptance, protocol freeze,
  numerical implementation or delivery is claimed. Original packet/caps and
  numerical source are unchanged. Final repository QA, bounded privacy checks
  and status-only changes are recorded in the additional-trace package.

## VAL-016 — Preserve the static counter-placeholder boundary

- Symptom/impact: a re-signed static trace could change `primitive_count` or
  `max_normalized_bits` from the frozen zero placeholders without semantic
  rejection. This concerns fixed hypothetical metadata, not observed numerical
  work or measured cap exhaustion. The schema's legal numeric range alone did
  not enforce this narrower trace contract.
- Scoped correction: the private checker explicitly retains the two zero
  placeholders and false certification/authorization flags across its three
  named static cases. No worker instrumentation, cap, schema or numerical source
  changed. Available/unknown slot handling derives the indeterminate case without
  evaluating the unavailable candidate's quantities.
- Evidence: `runs/boundary-red` in the new author archive contains two genuine
  missing-counter refusals. Four flag tests already received schema refusals;
  their message mismatches are not claimed as new behavioral RED tests. The
  earlier single positive RED is the old checker's fixed-authority refusal.
  Final selection: 66 new-case checks, all 59 original selections unchanged,
  plus a separate full no-qualifier artifact equality check. Both complete old
  canonical traces and all 38/27 body bytes remain equal. The parent checks all
  1,706 indexed files/1,707 archive members and a fresh exact/physical replay.
- At the author checkpoint: self-verification only; independent review was pending.
  Ten meaningful guard mutants are killed, while one inventory-guard attempt
  remains inconclusive because another guard rejects the same chosen input.
  Earlier diagnostic-message and schema-masked control failures are preserved.
  Broader A2 gates and the eight FIT-REAL admission blockers remain open.
- Prevention: distinguish a frozen illustrative value from a measured counter;
  check the declared fixed boundary, and distinguish missing-refusal RED tests
  from an already-correct refusal with a different diagnostic message.

## VAL-017 — Typed fixed-authority identity is not Python equality

- Finding: `CAP-AUTH-TYPE-01`, independent review of the cap-after-qualifier
  increment. The new `frozen_case.py:37-43` compares complete case/bundle/scalar
  structures with Python `==`, which equates boolean/integer and float/integer
  lookalikes. Matching values do not establish the required canonical types.
- Reproductions: changing `case.required_evidence_slots[13].candidate_index`
  from `1` to `true`, slot 4's index from `0` to `false`, or
  `working_scalars.float64.alpha.nbytes` from `8` to `8.0` all pass
  `authenticate`, `make_trace` and `verify_trace`. Canonical authority bytes
  differ, while emitted trace bytes remain identical. This is a promised
  authentication/type-boundary defect, not an observed worker failure or wrong
  exact arithmetic.
- Evidence: all 532 indexed independent files and 533 members are preserved in
  the cap-after-qualifier package's `independent-review-rejected-evidence.tar.gz`.
  The parent verifies the complete index and reproduces all three cases from
  unchanged copied inputs; the probe's actual child exit is 2. See
  `independent-review-verdict.json`, `independent-review-preservation.json` and
  `parent-rejection-readback.json`. The failed verdict remains failed.
- Other observations: 59 previous selections, 66 new-case tests and the added
  artifact equality test pass. Known arithmetic/physical accounting and both
  old artifacts remain unchanged. A new extra-candidate control isolates and
  validates the inventory guard; that resolved coverage issue does not clear
  the independent typed-identity failure.
- Rejected-checkpoint disposition: **blocked; no source repair performed or
  authorized at that checkpoint**. Any next
  repair must preserve this rejected source/proof, enforce exact types as well
  as values across the affected identity predicates, add the counterexample
  regressions, preserve legitimate artifact bytes and obtain fresh independent
  review. Passing pre-existing suites cannot substitute for this gate.
- Prevention: test bool/int and float/int substitutions in every authenticated
  nested structure, not just emitted records. Use type-preserving canonical
  comparison or explicit recursive type checks rather than coercive equality.
- Subsequent owner approval: one bounded predicate-only repair is now authorized
  under `identity-repair-2026-11-13/authorization.json` within the cap-after-qualifier
  package. The correction must add genuine failing rejection tests before the
  helper edit, address the affected nested identity comparisons, preserve all
  original tests and all three valid artifacts, then pass fresh independent
  review. The rejected helper and all failed proof/blocked records remain frozen;
  no automatic additional repair after a failed re-review, cap revision, numerical
  implementation, commit/push or broader A2 closure is authorized.
- Correction self-checks: the case, witness bundle and working-scalar predicates
  compare canonical JSON bytes, preserving JSON type distinctions. Only the
  authorized helper changed; the main checker, original tests/fixtures and all
  three valid canonical artifacts/38-27-28 body sets remain unchanged. All 27
  typed-identity tests pass after RED (26 missing refusals and one sibling
  verifier TypeError, not 27 identical failures), together with 126 previous
  selections and eight posthoc defensive checks. Isolated predicate reversions
  expose the case/bundle/scalar failures. The parent verifies 1,632 indexed
  files/1,633 archive members, checks the original input bytes and fresh-replays
  the preserved counterexamples to rejection. Historical failed attempts remain
  preserved. This author/parent checkpoint did not close VAL-017; the subsequent
  independent outcome is recorded below. See the nested identity-repair package
  for exact source pins and proof.
- Final scoped disposition: fresh independent re-review passes, with no security
  or logic findings. The reviewer reproduces 230 passing checks, including 69
  additional independent controls, and all three complete canonical artifacts/
  38-27-28 body sets remain equal. The parent verifies 720 indexed review files,
  all 721 archive members and a fresh 69-control execution with actual exit zero.
  The corrected helper is the reviewed canonical-JSON predicate derivative;
  no further source change occurred after review.
- Close VAL-017 only within this three-named-case local JSON identity audit, and
  record the associated static cap-after-qualifier increment and VAL-016 guard as
  verified. The old failed verdict, counterexamples, blocked records and earlier
  attempts remain historical failures. Inherited archive lint, privacy limits,
  all eight FIT-REAL admission blockers and broader scientific/resource gates
  remain disclosed. No numerical-source/cap/criterion change, protocol freeze,
  commit/push, release or phase closure follows from this result.
