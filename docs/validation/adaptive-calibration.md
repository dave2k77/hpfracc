# Adaptive Tolerance Calibration Audit

Status: Unreleased development audit; calibration correction remains open.

The adaptive controller is operationally tested, but its step-doubling indicator
is **not a calibrated bound on the retained coarse-step error**. The first-panel
counterexample below establishes this even before any old-history error exists.
The tolerance sweeps also show that smaller absolute errors do not imply errors
below the requested global scale. No numerical controller updates were made in
this audit; source docstrings/comments now state the measured limitation.

The [startup-scaling follow-up](adaptive-startup.md) derives the weak-forcing and
autonomous error powers and adds a smooth cubic case with zero pair difference
but nonzero true error. It explains why a universal rescaling alone is not a
general correction.

## Reproduce

Run from the source checkout with the test/validation dependencies installed:

```bash
uv run --frozen python -m benchmarks.numerical.adaptive_calibration --check
uv run --frozen python -m benchmarks.numerical.adaptive_calibration --first-step --check
JAX_ENABLE_X64=1 uv run --frozen python -m benchmarks.numerical.adaptive_calibration --dtype float64 --check
JAX_ENABLE_X64=1 uv run --frozen python -m benchmarks.numerical.adaptive_calibration --dtype float64 --first-step --check
```

Default precision is explicitly float32. Float64 requires the caller to enable
x64; the report does not change global precision or silently downgrade the request.
`--output-dir docs/validation/audits/adaptive-calibration` additionally writes
CSV and companion provenance JSON. `--max-steps` controls the sweep budget only;
the first-panel probe always uses two attempts at most.

`--check` has a deliberately narrow meaning:

- Sweep: nonempty, completed, finite full-span measurements, with consistent
  accepted-prefix shape, mask, counts, initial state, and endpoint.
- First panel: nonempty, precision-appropriate agreement of numerical coarse/fine
  values with independent closed-form product integrals. Adaptive acceptance is
  an observation, not a requirement that future controllers preserve this behavior.

A zero exit code is **not** a certificate that error/tolerance is at most one.
Incomplete or invalid sweep trajectories have empty error fields, rather than
misleading partial-span accuracy numbers. This standalone report does not change
existing aggregate CSV columns or cases.

## Independent continuous references

All problems start at zero time. The four scalar cases are:

| Case | Equation and initial value | Exact solution |
| --- | --- | --- |
| Constant forcing, alpha=0.5 | `D_C^alpha y = 1`, `y(0)=0` | `t^alpha / Gamma(alpha+1)` |
| Quadratic forcing, alpha=0.5 | `D_C^alpha y = t^2`, `y(0)=0` | `2*t^(alpha+2) / Gamma(alpha+3)` |
| Half-order decay, alpha=0.5 | `D_C^0.5 y = -y`, `y(0)=1` | `exp(t)*erfc(sqrt(t))`, evaluated as `scipy.special.erfcx(sqrt(t))` |
| Manufactured feedback, alpha=0.8 | `D_C^alpha y = 2*t^(2-alpha)/Gamma(3-alpha) - (y-t^2)`, `y(0)=0` | `t^2` |

References use host float64 arithmetic at the actual stored nodes, not solver
weights or a self-refinement trajectory. The erfcx oracle is cross-checked against
60-digit mpmath arithmetic. Float32 measurements include coefficient and numerical
roundoff; stored fractional orders are recorded rather than relabeled as exact
decimal inputs.

The sweep uses `[0,1]`, `first_step=0.01`, `max_steps=512`,
`rtol=(1e-3,1e-4,1e-5)`, and `atol=0.01*rtol`. It records maximum and endpoint
absolute error and the following reference-scaled error:

```text
max_scaled_error = max over accepted nodes of
                  abs(y_n - y_exact(t_n)) / (atol + rtol*abs(y_exact(t_n)))
```

This is not the controller's pair-normalized RMS indicator. It measures global
solution error at the returned nodes, not error between nodes or a dense-output
bound. Different tolerances produce different node sets.

## Measured tolerance sweeps

All 24 sweep configurations completed on the recorded local CPU environment
(Python 3.13.13, JAX 0.10.1). In float64:

| Case | rtol | Maximum absolute error | Maximum reference-scaled error |
| --- | --- | --- | --- |
| Quadratic forcing | 1e-3 | 1.346397e-3 | 3.634916 |
| Quadratic forcing | 1e-4 | 2.585439e-4 | 6.316750 |
| Quadratic forcing | 1e-5 | 4.732918e-5 | 10.468320 |
| Half-order decay | 1e-3 | 1.120849e-3 | 1.790564 |
| Half-order decay | 1e-4 | 2.450158e-4 | 2.550739 |
| Half-order decay | 1e-5 | 2.606445e-5 | 2.619954 |
| Manufactured feedback | 1e-3 | 2.431945e-3 | 3.890044 |
| Manufactured feedback | 1e-4 | 2.878880e-4 | 6.627845 |
| Manufactured feedback | 1e-5 | 2.991706e-5 | 9.445952 |

Absolute errors decrease for these nontrivial cases, while reference-scaled
errors increase. Float32 shows the same qualitative behavior; the tightest
quadratic case has scaled error 10.471617. Constant forcing is an exact
product-integration control up to roundoff: maximum absolute error is about
5.86e-8 in float32 and 4.44e-16 in float64, unchanged by this tolerance sweep.

## Exact first-panel counterexample

For `D_C^alpha y=t^2`, `y(0)=0`, `0<alpha<1`, the Volterra integral gives:

```text
y(h) = integral_0^h (h-s)^(alpha-1) s^2 ds / Gamma(alpha)
     = 2*h^(alpha+2) / Gamma(alpha+3).
```

Because forcing is state-independent, the predictor introduces no forcing error.
The corrector integrates a linear interpolant against the power kernel. Write
`A=h^(alpha+2)/Gamma(alpha+2)` and `R=2^(1+alpha)-1`. For one panel and two half
panels, respectively:

```text
L_coarse(s) = h*s
L_fine(s)   = h*s/2                 for 0 <= s <= h/2
              3*h*s/2 - h^2/2      for h/2 <= s <= h

Y_coarse = A
Y_fine   = A*(1/2 + 2^(-alpha-1))

E_coarse = Y_coarse - y(h) = A*alpha/(alpha+2)
E_fine   = Y_fine   - y(h) = A*(1/2 + 2^(-alpha-1) - 2/(alpha+2))
Delta    = Y_coarse - Y_fine
I        = abs(Delta) / R           (the current indicator formula)

E_coarse/I = alpha*2^(alpha+1)*(2^(alpha+1)-1)
             / ((alpha+2)*(2^alpha-1)).
```

Both signed errors are positive: linear interpolation overestimates a strictly
convex quadratic and the integration kernel is positive. Since
`E_coarse=Delta+E_fine>Delta` and `R>1`, **the indicator underestimates the retained
coarse error for every order in this interval**. This is an exact identity for
this forcing, not a general estimate of nonlinear solver error.

For `h=0.1`, the actual float64 numerical pairs agree with those closed forms:

| alpha | Coarse absolute error | Indicator I | Coarse error / I |
| --- | --- | --- | --- |
| 0.1 | 3.614493e-4 | 2.222509e-4 | 1.626312 |
| 0.5 | 4.757664e-4 | 1.905309e-4 | 2.497056 |
| 0.9 | 2.138068e-4 | 5.851529e-5 | 3.653862 |

The executable probe also calls the actual adaptive solver with
`first_step=h`, `rtol=1e-12`, and `atol=1.1*I` from the analytic oracle. At
alpha=0.5, the first trial's scaled indicator is 0.909091, but its true coarse
error divided by the same scale is 2.270051. The solver completes with one
accepted step and no rejections. All six precision/order probes recorded coarse
retention; the report records this behavior rather than making it an invariant
for future controller implementations.

## Common history and Richardson interpretation

For the same time-only quadratic forcing, splitting a new trial panel leaves
older accepted panels unchanged. Let `H` be their interpolation error integrated
against the kernel at the new endpoint. Then exactly:

```text
E_coarse_total = H + E_coarse_new_panel
E_fine_total   = H + E_fine_new_panel
```

The common `H` cancels from the trial difference. Making only the new panel
smaller does not remove old-history error. State-dependent problems add further
propagation effects; a general calibration theorem does not follow from this
special-case decomposition.

Even classical Richardson algebra requires a target distinction: if
`E_fine ~ 2^(-q)*E_coarse`, then `Delta/(2^q-1)` estimates the **fine** error;
the coarse estimate would include a factor `2^q`. The exact pair above does not
generally satisfy that scaling relation with `q=1+alpha`. Multiplying the current
indicator by that factor is therefore **not a proven general correction**.
The raw quadratic indicator scales as `h^(alpha+2)`, consistent with the current
controller exponent only when its normalization is approximately constant; this
does not establish the exponent for weak startup singularities or general dynamics.

Background: [Detailed Error Analysis for a Fractional Adams Method (2004)](https://doi.org/10.1023/B:NUMA.0000027736.85078.be)
discusses assumption-dependent fractional Adams error analysis and asymptotic
expansions. The first-panel identities above are an independent derivation for
this implementation, not a calibration result attributed to that paper.

## Evidence and remaining correction gate

The [float32 sweep](audits/adaptive-calibration/adaptive_calibration_float32.csv),
[float64 sweep](audits/adaptive-calibration/adaptive_calibration_float64.csv),
[float32 first-panel probes](audits/adaptive-calibration/adaptive_first_step_float32.csv),
and [float64 first-panel probes](audits/adaptive-calibration/adaptive_first_step_float64.csv)
contain the raw measurements. Each has a same-stem `.json` companion with runtime
provenance, dirty-tree status, dtype/x64 configuration, source SHA-256 hashes, and
a hash of its CSV. These are development audit artifacts, not a release bundle.
Hashes identify the measured files but do not preserve uncommitted source by
themselves; archive or commit reviewed source before claiming external reproducibility.

`tests/unit/test_adaptive_calibration_report.py` checks independent reference
identities, selected refinement, corrupted trajectories and panel pairs,
nonfinite/incomplete runs, and the intentionally limited CLI gate.

Before changing the controller:

1. Specify whether it controls the retained panel defect or a full-trajectory
   quantity; separate this from operational completion.
2. Derive the coarse/fine scaling for relevant forcing regularity, startup
   singularities, and state-dependent dynamics rather than fitting one multiplier
   to this quadratic example.
3. Validate acceptance/calibration on independent problems, with explicit
   history-error limitations and case-specific quantitative criteria.
4. Rerun dtype, JIT, frozen-mesh gradient, failure-status, and report regressions;
   review trajectory/work changes explicitly. Keep full-history behavior and
   published interfaces unless an intentional additive change is approved.
