# Coupled and Scaled Adaptive Audit

Status: Unreleased local CPU evidence. Subdivision remains experimental and opt-in.

The [development checkpoint](development-2026-09-14.md) fixes this audit's design
before its execution. It extends the existing scalar stress/runtime evidence to
coupled states and multiple fractional orders; it is not a runtime comparison.

## Independent problem and measurements

Let `S=diag(s)` and `y(t)=S*u(t)`, with:

```text
u(t) = (1+t^2, 0.5+t^3)
e = S^-1*y - u(t)
A = [[-1, 0.5], [0.25, -2]]
f(t,y) = S * (q_alpha(t) + A*(e + 0.1*e^3))
q_alpha(t) = (2*t^(2-alpha)/Gamma(3-alpha),
              6*t^(3-alpha)/Gamma(4-alpha)).
```

The cubic is componentwise, and multiplication by `A` couples the components.
The feedback term vanishes on the exact solution but influences off-solution
trials. Test-only SciPy weighted quadrature independently checks the continuous
Caputo integral of `u'`, and a Jacobian check verifies both off-diagonal couplings.

The 48 configurations use:

- `s=(1,1)` or `(1e-3,1e3)`, orders `0.25,0.5,0.8`;
- both estimator modes and `rtol=1e-2,1e-3`, with `atol=0.01*rtol`;
- bounds `[0,1]`, initial state `S*(1,0.5)`, feedback strength 1;
- `first_step=0.01`, `max_steps=1024`, float32 and float64 separately.

Initial state, scale, fractional order and feedback are dynamic JIT inputs.
References are evaluated on the actual accepted times in host float64, using
the component scales as represented in the input dtype. Reports retain both
componentwise absolute errors and errors divided by component scales. The
reference-scaled metric is `max |y-y_exact|/(atol+rtol*|y_exact|)`, not the trial
indicator. A common scalar `atol` deliberately allows different relative accuracy
for the small component; rescaling the state need not preserve the selected mesh.

## Recorded results

Both checked commands exited 0: all 48 solves completed with valid finite
accepted prefixes, count/status/mask consistency, and repeated-tail padding.
For every case/order/mode/precision, tightening the tolerance reduced the maximum
dimensionless error at accepted nodes. At equal tolerances, subdivision had a
smaller such error in all these selected pairs, and used more accepted steps.

At `rtol=1e-3` in float64:

| Scale case | Order | Default max dimensionless error | Subdivision max dimensionless error | Default accepted steps | Subdivision accepted steps | Subdivision max reference-scaled error |
| --- | --- | --- | --- | --- | --- | --- |
| Balanced | 0.25 | 3.375e-3 | 1.016e-3 | 121 | 442 | 1.002 |
| Balanced | 0.5 | 3.725e-3 | 7.574e-4 | 36 | 138 | 1.037 |
| Balanced | 0.8 | 4.543e-3 | 5.600e-4 | 17 | 60 | 0.998 |
| Mixed scale | 0.25 | 8.283e-3 | 5.236e-3 | 120 | 443 | 1.538 |
| Mixed scale | 0.5 | 4.518e-3 | 1.699e-3 | 36 | 138 | 1.207 |
| Mixed scale | 0.8 | 4.556e-3 | 5.600e-4 | 17 | 61 | 1.016 |

Several global errors still exceed the reference tolerance scale. These
accepted-node maxima are not dense-output bounds or matched-work measurements.
The earlier [dense runtime audit](adaptive-runtime-dense.md) remains the relevant
efficiency evidence. No estimator or default is promoted by these new results.

## Differentiation and failure checks

`tests/unit/test_adaptive_coupled.py` additionally checks:

- coupled initial-state, feedback-parameter, and order sensitivities against
  finite differences on the frozen accepted mesh, for both estimators at order
  0.5; the manufactured forcing is held fixed during the order perturbation;
- agreement with the nonuniform solver's frozen-mesh derivatives under JIT;
- a compiled quarter-only nonfinite failure after one accepted step, including
  exact retained prefix, counts, mask, and finite padding;
- invalid-padding detection, suppressed full-span errors after budget failure,
  and missing/duplicate/failed report-matrix rejection.

This is case-specific differentiation evidence, not a general nonlinear
stiffness, controller-gradient, or calibrated error guarantee.

## Reproduce and artifacts

Use a new output directory to preserve the recorded experiment:

```powershell
uv run --frozen python -m benchmarks.numerical.adaptive_coupled --check --output-dir docs/validation/audits/coupled-new
$env:JAX_ENABLE_X64 = '1'
uv run --frozen python -m benchmarks.numerical.adaptive_coupled --dtype float64 --check --output-dir docs/validation/audits/coupled-new
```

`--check` requires the exact matrix and finite complete measurements, not global
error below tolerance. Failed solves retain diagnostics with empty accuracy fields.

- [Float32 CSV](audits/coupled-2026-09-14/adaptive_coupled_float32.csv) and
  [JSON/provenance](audits/coupled-2026-09-14/adaptive_coupled_float32.json).
- [Float64 CSV](audits/coupled-2026-09-14/adaptive_coupled_float64.csv) and
  [JSON/provenance](audits/coupled-2026-09-14/adaptive_coupled_float64.json).

The Windows CPU run used Python 3.13.13 and JAX/jaxlib 0.10.1. JSON records the
dirty development source, source digests, configuration, and CSV digest. The
checkpoint file remains the pre-execution declaration rather than being edited
after observing the results. This is not a clean-candidate release dossier.
