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
| API-001 | Metadata contract limitation | Strong array orders prevent plain JSON export of SolverInfo | Reproduced before and after NUM-004; open |
| VAL-002 | Validation gap | No dedicated FFT order-gradient finite-difference matrix | Selected coverage verified after VAL-004 correction |
| DOC-001 | Documentation | Nonuniform solver missing from API stability table | Fixed |
| SCI-001 | Research limitation | General adaptive tolerance indicator is not a calibrated error bound | Open |
| SCI-002 | Research / validation gap | History selection and measured reverse/peak-memory evidence incomplete | Open |
| VAL-003 | Validation semantics | Measurement-integrity checks are insufficient scientific acceptance | Implemented for selected implicit cases; broader gates open |
| VAL-004 | Validation defect | x64-enabled fixture masked nominal float32 FFT execution | Fixed; corrected CPU/CUDA matrices pass |
| VAL-005 | Validation defect caught before commit | Numeric value equality weakened configuration identity | Fixed; strict canonical types and regressions |
| VAL-006 | Evidence documentation | Current citations retained superseded CUDA measurements | Corrected; ten report-to-documentation checks |
| VAL-007 | Evidence bookkeeping / reproducibility scope | Exact isolated/full-suite CUDA residual equality was assumed | Bookkeeping corrected; emitter variation reproduced, selection cause unresolved |
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

### SCI-002 — Approximation selection and memory evidence

Fixed SOE poles/iterations do not certify a requested error budget; forward
recursive state size does not describe reverse-mode or measured peak memory.
Impact: accuracy-preserving efficiency claims remain incomplete. Resolution work
must separate fitting/application/gradient costs, record unattainable budgets,
and measure rather than infer memory. Existing conditional bounds and CPU
compiler-accounting artifacts remain valid only within their declared scope.

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
- Remaining: provide and test host-side JSON normalization without introducing
  array-to-Python conversion into numerical JIT/gradient paths. This issue remains
  open; it is neither a new NUM-004 regression nor a completed contract fix.
- Prevention: test metadata export using actual supported scalar-array orders,
  not only Python configuration scalars, independently of numerical dtype tests.
