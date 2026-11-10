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

For any claimed tolerance-controlled case, test accepted-node errors against
the stated componentwise reference scale, not just decreasing errors. Require
an independently justified error-control contract before extending a bound to
nonlinear feedback. General feedback control requires a trajectory reconstruction,
an enclosed Volterra defect and a valid propagation bound; Jacobian sampling or
additional trial subdivisions alone do not supply those assumptions.

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
