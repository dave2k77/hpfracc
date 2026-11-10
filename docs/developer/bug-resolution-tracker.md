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
| VAL-002 | Validation gap | No dedicated FFT order-gradient finite-difference matrix | Selected coverage verified after VAL-004 correction |
| DOC-001 | Documentation | Nonuniform solver missing from API stability table | Fixed |
| SCI-001 | Research limitation | General adaptive tolerance indicator is not a calibrated error bound | Open |
| SCI-002 | Research / validation gap | History selection and measured reverse/peak-memory evidence incomplete | Open |
| VAL-003 | Validation semantics | Measurement-integrity checks are insufficient scientific acceptance | Open; next work package |
| VAL-004 | Validation defect | x64-enabled fixture masked nominal float32 FFT execution | Fixed; corrected CPU/CUDA matrices pass |
| ENV-004 | Environment / delivery | GitHub write authentication is not configured | Open; user authentication required |

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
  gradient dtypes checked. Full-suite reruns are pending. The two pre-review
  full-suite processes were deliberately stopped as superseded, not reported
  as successful checks. Raw red/green results and corrected source/runtime
  provenance are preserved in the [FFT precision bundle](../validation/audits/fft-order-gradients-2026-11-10/README.md).
- Prevention: assert effective forward/objective dtypes as well as input and
  gradient dtypes; test each precision mode explicitly rather than inferring it
  from parameter labels.

## Open findings and next work

### ENV-004 — GitHub write authentication

Discovery: 2026-11-10, authorized push preflight.

- Symptom: public fetch/API reads succeeded, but an HTTPS push dry run failed
  because a username could not be read with prompts disabled. A read-only SSH
  probe with strict host checking also failed because no trusted GitHub host key
  was configured.
- Impact: local review and commits can proceed, but a remote push and candidate
  CI verification cannot be completed with the current authentication setup.
- Cause: public read access does not establish authenticated repository write
  access; neither tested transport supplied a usable authenticated path.
- Resolution: pending user-completed authentication through their own trusted
  Git/GitHub CLI workflow. No credentials were requested in chat, credential
  stores read, remote URLs changed, SSH host keys accepted, or keys generated.
- Prevention: test write access with a non-mutating push preflight early, and
  never infer push permission from a successful public fetch.

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
Phase A work package; no new scientific-acceptance implementation is claimed here.
