# A1: Volterra-defect design and private feedback tracer

Status: checked mathematical design and a working private host prototype;
not an adaptive-controller implementation, total numerical certificate, public
API, approved scope deferral or Phase A completion.

## Preserved design

`protocol-and-checks.tar.gz` preserves the initial design report, both manifests,
80-digit checks, their output and five deliberately invalid shortcut assertions.
Those five failures are design negative controls, not production regressions.
The parent reran the mathematical checks in a separate copy and reproduced the
output. No numerical/library source was changed for this work.

The design reconstructs a continuous piecewise-linear trajectory through stored
states and encloses the complete Volterra defect before feedback propagation.
It accounts for all old panels. The first proposed feedback scope is an affine
constant matrix with a weighted contraction factor below one; nonlinear tube
closure and broader stiff propagation remain open. Weak-power/Hölder forcing
does not assume finite C2 curvature at the origin.

An initially nondiscriminating old-panel fixture is retained in the first
manifest and check output. The second design manifest corrects that fixture
before implementation; this is not a retuned completion experiment.

## Executed private vertical tracer

`feedback-prototype.tar.gz` contains an implemented host-only mathematical
helper and its tests. Its first feature test fails because the helper is absent,
then passes after implementation; the expanded compatibility selection passes
15 tests. This is not integrated into the library or any solver/controller.

For half-order feedback `D_C^(1/2)y=y/4`, initial value one and the constant
candidate reconstruction `Y=1` on `[0,1]`, the independently checked raw defect
is 0.28209479177387814347, while the actual endpoint error is
0.35864237010472211521. Contraction propagation gives the conditional model
bound 0.39294155905472337601. Omitting propagation would incorrectly accept an
absolute budget of 0.3; the implemented model diagnostic rejects 0.3 and passes
0.4 conditionally.

The helper explicitly returns `arithmetic_verified=False` and
`total_numerical_certified=False`. Mpmath evaluation is not outward-rounded
proof. It rejects nonfinite/invalid inputs and marks contraction factors at or
above one as outside this protocol's scope, not solver instability. Scaling,
signed feedback, conservative relative budgets and scoped precision are tested.

## Independent review and evidence limits

`initial-review.json` preserves an overall `passed=false` verdict because the
review did not clear the remaining A1 acceptance gates. It reports no security
or scoped-formula logic errors, reproduces the 15 private tests and performs 27
additional scalar probes. This is not converted into an A1 pass or a deferral.
The subsequent `scope-review.json` separately approves archiving this limited
artifact while explicitly retaining `a1_acceptance_passed=false` and every open
gate; it does not replace the initial verdict.

The initial feature-test source is recovered verbatim from its original
authoring input in `feature-red-source-as-authored.py.txt`; its 972-byte length
matches the original write. It was not separately byte-snapshotted before the
RED process, and is not claimed to be an independently timestamped source
capture. The archived current test additionally imports the implemented helper.
The original archives and their checksums are unchanged.

## Next implementation gates

1. Preserve the design manifest and implement complete piecewise-linear
   reconstruction/affine-power defect evaluation, including PECE nodal mismatch.
2. Enclose the continuum defect over verification cells, not sampled maxima.
3. Retain every old panel and add componentwise Neumann propagation/tail checks.
4. Extend the checked weak-forcing remainder and independently justified feedback
   domains; do not infer nonlinear bounds from sampled Jacobians.
5. Supply verified arithmetic/evaluation budgets before any total working-
   precision certification, then validate a separately reviewed opt-in controller.

The per-state solver-order decision remains separate and unanswered. No scalar
prototype result resolves that decision or closes A1/A2/A3/A4.
