# NUM-005: CPU float32 history accumulation

Status: unreleased local verification record; not Phase A closure or release
sign-off. Independent review and source delivery are recorded separately.

The directory retains its investigation-start date, 2026-11-10. The repair and
new scientific-policy reports were produced on 2026-11-11 UTC; execution journals
record their actual completion times. Earlier evidence bundles are unchanged.

## Defect and scope

The pre-repair CPU stable-weight solver reports Newton convergence for half-order
linear decay with rate 30, 257 nodes, eight Newton iterations, `rtol=1e-5` and
`atol=1e-7`, but its independent maximum scaled algebraic residual is
1.1998344389138271. This case already passed on CUDA at 0.23292174906264487.
It is outside `implicit-scientific-v1`; neither its identity nor the frozen
manifest's success/rejection roles were changed to accommodate the repair.

A reduction-only counterfactual lowers the CPU residual to 0.2770276016181242.
The applied `_history_dot` change uses broadcast multiplication and an explicit
sum over the leading history axis instead of a contraction. It preserves the
full-history discrete method, coefficients, Gamma normalization, selectors,
working dtype, Newton iterations and acceptance thresholds. Floating-point
accumulation is intentionally changed. No arbitrary-sum error bound, continuous
accuracy guarantee or general stiff-solver claim follows from this result.

## Preserved pre-repair evidence

- `before-checkpoint.json` and `before-sources.tar.gz`: actual source checkpoint
  `415fc21e8e62d062a180aafb7cda8fbcec098f2b`, 116 numerical/benchmark/test inputs
  with individual and archive digests. This is not the source of the new reports.
- `red-test-source.py.txt`, `red-cpu.xml.gz`, `baseline-cuda.xml` and
  `red-regressions.json`: exact four-case test source and pre-repair outcomes.
  CPU has two failures/two passes; CUDA already passes all four controls. Gzip
  preserves the raw failure XML, including whitespace, with both digests.
- `reduction-probe.py.txt` and `reduction-counterfactuals.json`: parent-controlled
  default/highest-contraction/explicit-sum comparisons on both backends.
- `independent-analysis.tar.gz` and `independent-analysis-manifest.json`: 42
  verbatim probe/report/raw-data files. Inside the archive,
  `num005_basis/report.json` is the authoritative signed decomposition at step
  209; `num005_crosscheck/report.json` contains the 80-digit panel cross-check.

The independent fixed-history replay reproduces the public trajectory
bit-for-bit. History product/reduction error contributes 1.1270974033428172
independent-tolerance units; coefficient error contributes 0.022570372065334462
and Newton's runtime-equation error 0.0021260254540220965. Sequential float32
summation reproduces the observed dot; requesting highest dot precision does
not fix it. Known-term cancellation amplifies the accumulated history error,
while that final operation's own rounding is small.

Instrumentation matters: returning intermediate weights or adding debug
callbacks changed fusion and made the unfixed case appear to pass. Those
counterfactuals are retained but are not the causal decomposition. Do not rerun
pre-repair probes against the repaired checkout and relabel their output as
baseline evidence; their original source checkpoint and reports are preserved.

## Current focused verification

`green-regressions.json` and `green-{cpu,cuda}.xml` record four passing original
regressions on each backend with confirmed zero exits. Their test-source digest
matches the RED snapshot; the production helper digest is recorded separately.

`compatibility/` retains eight verbatim files from the completed independent
test expansion, bound by `compatibility-manifest.json`. All 25 cases pass on
each CPU/CUDA backend: signed/mixed-scale histories, trailing shapes, effective
float32/float64/complex dtypes in the private helper, analytic JVP/VJP and finite
differences, strong-float32 dynamic-order JIT stiff controls, and shared
uniform/nonuniform/adaptive paths. Complex helper evidence is not public
complex implicit support; adaptive checks do not establish general accuracy.

Eight `scientific_acceptance_{cpu,gpu}_{float32,float64}_{direct,stable}` report
sets retain 48 raw NPZ cases. All 32 required-success trajectories are accepted;
all 16 expected-rejection controls remain `accepted=False`, `policy_passed=True`.
`executions.json` records every actual command and exit; `verification.json`
replays decisions from the raw arrays and checks exact eight-way coverage,
canonical manifests, complete source hashes, artifact hashes and CSV contents.
The frozen manifest and its budgets are unchanged. A citation guard first
failed on the missing current NUM-005 section, then verifies the eight measured
weak/stiff table entries in `docs/validation/uniform-weights.md`; its initial
failure and the earlier replay/checker are retained separately.

From the repository root, replay the current report bundle with:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu uv run --frozen --extra dev python docs/validation/audits/num005-stable-precision-2026-11-10/verify.py
```

This host replay verifies the archived measurements and selected policy, not
universal accuracy. The stiff regression remains separate from the six-case
scientific-policy matrix.

`additional-gates.json` and six CSVs record the CPU/CUDA aggregate (28 checks),
core-hardening (12) and history-validation (8) gates, all with zero exits.
`wheel-provenance.json`, `wheel-executions.json` and `installed-wheel-smoke.json`
record a built wheel installed into a new runtime-only, hash-locked environment.
The isolated CPU interpreter verifies the installed helper digest, lazy import,
effective float32 JIT/objective/order-gradient execution and three selected
independent algebraic residuals, including the stable stiff case. The initial
probe/result before a lambda-to-function style cleanup are retained separately;
the final probe was rerun successfully. This is smoke evidence, not release
or general continuous-accuracy acceptance.

## Separate delivery and remaining gates

`prior-commit-ci.json` records a fresh successful readback of remote run
42924720362 for commit `415fc21e8e62d062a180aafb7cda8fbcec098f2b`: Python
3.11–3.13, lint, strict docs and performance jobs succeeded. That is the prior
NUM-004 delivery, not remote validation of this uncommitted NUM-005 candidate.

The background watch also exits zero. `prior-commit-ci-watch.json` preserves its
non-blocking annotations: `actions/checkout@v4` targets deprecated Node.js 20
and is forced onto Node.js 24; `ubuntu-latest` carries an Ubuntu 26 migration
notice. These did not fail the recorded run. Action/runner maintenance is
separate; no workflow upgrade is bundled into the numerical repair.

Full CPU/CUDA suites each pass 879 tests, one performance case deselected, with
confirmed zero exits. `full-{cpu,cuda}.xml` and `full-suite-verification.json`
verify matching test identities, every enumerated case and all 118 source inputs
unchanged from `suite-source-inputs.json` and `suite-sources.tar.gz`.
`full-suite-executions.json` records the actual completed process handles.
The earlier CPU-only verification remains a time-specific intermediate record.

A post-run assertion incorrectly required exact CUDA residual-property equality
between the isolated compatibility run and the full suite. Both meet the same
unchanged limit of one, but their stable/direct values differ:
0.23292174906264487 / 0.24529762807235306 in isolation versus
0.22442286756041424 / 0.2788388698063575 in the full suite. Both raw datasets,
the failed postcheck source/error, and the context-labelled properties are
retained. CPU values match across contexts.

The bounded follow-up in `cuda-context-investigation.tar.gz` reproduces both
pairs even with identical target-only commands. Captured StableHLO and
whitelisted configuration are identical and floating operations are f32-only,
with no dot. Optimized history reductions use Triton versus native emitters,
corresponding to the two pairs. Five replays per executable are bitwise stable;
the captured trajectories differ by less than 1e-7. Parent host reconstruction
reproduces all four residuals. This supports a compilation-path explanation,
not a unique dependence on collection breadth, verbosity or preceding tests.

The archive retains 74 verified files and its manifest. Ten bounded pytest
processes pass 66 test executions with actual zero exits and no full-suite
rerun; an optional DebugOptions introspection process exits 139 and is preserved
separately, followed by a successful NumPy-only comparison. No numerical/test
source, working precision or scientific threshold is changed. Why XLA selects
different emitters remains unresolved, and historical optimized executables
were not captured; no nondeterministic-autotuning claim is made (VAL-007).
Independent review and source delivery remain separate from this measurement
record. The four CPU examples pass separately
with zero exits and raw output digests in `examples.json`. API-001 metadata export,
adaptive calibration and the broader Phase A gates are not closed here.
