# Adaptive Startup Scaling and a Cubic Blind Spot

Status: Unreleased derivation/validation follow-up; no controller change.

Subsequent work adds an [opt-in subdivision prototype](adaptive-subdivision.md).
This page records the derivation and pre-prototype evidence, not a claim that
the new indicator provides a universal error bound.

The [first calibration audit](adaptive-calibration.md) showed underestimation
for quadratic forcing. This follow-up derives the retained-coarse scaling for
weak time forcing and smooth scalar autonomous dynamics, then checks it against
independent references. It also gives a smooth forcing for which the current
coarse/fine indicator vanishes despite nonzero error.

**Outcome:** there is no universal exact alpha-only rescaling or startup exponent
across these classes. More strongly, no finite multiplier of this pair difference
can bound error for the cubic example below. This is not a claim that all
possible safeguarded estimators fail, or that every adaptive run is inaccurate.

## Reproduce and interpret

```bash
JAX_ENABLE_X64=1 uv run --frozen python -m benchmarks.numerical.adaptive_startup --check --output-dir docs/validation/audits/adaptive-startup
uv run --frozen python -m pytest tests/unit/test_adaptive_startup_report.py
```

The audit requires float64 rather than silently extracting small startup-error
coefficients from float32 cancellation. It leaves JAX precision and solver
behavior unchanged. The CSV contains 12 case/order summaries. Companion JSON
contains their 48 raw step-size measurements, three cubic probes, exact run
configuration, environment, source fingerprints, and a CSV digest.

`--check` verifies finite, correctly shaped float64 trajectories, independent
reference agreement, and the selected asymptotic checks: finest-step coarse/fine
error coefficients within 3% of their derived limits, and observed coarse-error
power within 0.05 of the predicted power. The cubic numerical pairs must agree
with their closed-form integrals. **Passing these checks validates the audit's
claims, not calibrated tolerance control.** Cubic adaptive acceptance is recorded,
not required as an invariant for a future corrected controller.

All startup pairs target the same endpoint: one full PECE step versus two
full-history half steps. Existing nonuniform PECE and adaptive trials share the
same product-integration weights; no alternative numerical formula is substituted
for the library under test.

## Smooth autonomous startup

Consider `D_C^a y=f(y)`, with fixed `0<a<1`, scalar initial value `y0`, and smooth
local dynamics. Define:

```text
f0 = f(y0),  g = f'(y0)
c = 1/Gamma(a+1),  d = 1/Gamma(a+2),  q = 2^(-a),  z = h^a.
```

The Volterra expansion gives:

```text
y(h) = y0 + c*f0*z + g*f0*z^2/Gamma(2*a+1) + O(z^3).
```

This follows by inserting `y-y0=A1*t^a+A2*t^(2*a)+...` into the Volterra
integral, expanding `f(y)=f0+g*(y-y0)+O((y-y0)^2)`, and integrating each power.
The exact PECE stage identities on the first panel are:

```text
C = y0 + d*z*[a*f0 + f(y0+c*z*f0)]
M = y0 + d*q*z*[a*f0 + f(y0+c*q*z*f0)]
P = y0 + c*z*[(1-q)*f0 + q*f(M)]
F = y0 + d*z*[(a-1+q)*f0 + 2*(1-q)*f(M) + q*f(P)].
```

Here `M` is the corrected midpoint, `P` the second-half predictor, and `C,F`
the coarse and fine endpoints. Taylor expansion gives:

```text
C = y0 + c*f0*z + g*f0*c*d*z^2 + O(z^3)
F = y0 + c*f0*z + g*f0*c*d*q*(3-2*q)*z^2 + O(z^3)
C-F = g*f0*c*d*(1-q)*(1-2*q)*h^(2*a) + O(h^(3*a)).
```

For `g*f0 != 0`, the first-panel discrepancy has power **2*a**, not generally
`a+2`. The signed retained-coarse error/discrepancy ratio tends to:

```text
R_aut(a) = [c*d - 1/Gamma(2*a+1)] / [c*d*(1-q)*(1-2*q)].
```

This is a **restricted startup asymptotic result**, not an error bound at an
arbitrary later step. If `g*f0=0`, the term disappears; equilibrium or higher
terms can give different behavior. The leading coefficients also vanish as
`a` approaches the classical endpoint. A fixed-order small-step expansion must
not be treated as uniform near `a=1`; the classical limit can have a different
leading power. This audit samples orders 0.25, 0.5, and 0.75, not that boundary.

## Weak time-only forcing

For `D_C^a y=t^beta`, `y(0)=0`, `beta>0`, the predictor state does not enter the
forcing. Consequently the following are exact, not asymptotic:

```text
C = d*h^(a+beta)
F = d*[2*(1-q)*2^(-beta)+q]*h^(a+beta)
y(h) = Gamma(beta+1)/Gamma(a+beta+1)*h^(a+beta)
C-F = d*(1-q)*(1-2^(1-beta))*h^(a+beta).
```

The discrepancy power is **a+beta**, and, when `beta != 1`, the coarse ratio is:

```text
R(a,beta) = [d-Gamma(beta+1)/Gamma(a+beta+1)]
           / [d*(1-q)*(1-2^(1-beta))].
```

At `beta=1`, both interpolations integrate the linear forcing exactly and this
ratio is undefined, not a fallback calibration constant. The quadratic case
recovers the previous audit. At the same `a=0.5`, the gains relative to the raw
pair difference are about 1.365685 for quadratic forcing, 1.557539 for
`beta=0.25`, and 1.467992 in autonomous startup. These are different exact or
limiting targets, not one alpha-only calibration.

The current controller uses a raw-power assumption `a+2` in its exponent.
Startup powers alone would suggest `2*a` or `a+beta` in these respective regimes,
but replacing the solver-wide exponent with either is not justified: forcing
regularity, state dependence, normalization, old history, and cancellations matter.

## Independent references and measurements

The autonomous examples are `f(y)=-y` and `f(y)=-y^2`, both with `y0=1`.
Their references are independently derived local Volterra series
`y(t)=sum b_n*t^(n*a)`, with `b0=1`:

```text
linear:    b_(n+1) = -b_n*Gamma(n*a+1)/Gamma((n+1)*a+1)
nonlinear: b_(n+1) = -sum_(k=0)^n b_k*b_(n-k)
                     *Gamma(n*a+1)/Gamma((n+1)*a+1).
```

Coefficients and Horner evaluation use 70-digit mpmath arithmetic, comparing
80 and 120 terms before conversion to host float64. This truncation-agreement
check is not a rigorous interval/tail certificate. The helper is explicitly
restricted to `h^a <= 0.04`; the audit uses only
`h^a=(0.01,0.005,0.0025,0.00125)`. Independent controls check half-order linear
decay against `exp(t)*erfc(sqrt(t))` and the nonlinear series at `a=1` against
`1/(1+t)`; the latter validates the reference only, not unsupported order-one
library execution. Time-only references use the closed-form power integral.

Observed coarse-error powers from the last two step sizes on the recorded CPU
(Python 3.13.13, JAX 0.10.1, mpmath 1.4.1):

| a | Weak forcing beta=0.25 | Predicted | Linear decay | Nonlinear decay | Autonomous predicted |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 0.500000 | 0.5 | 0.496796 | 0.493413 | 0.5 |
| 0.5 | 0.750000 | 0.75 | 0.995482 | 0.990950 | 1.0 |
| 0.75 | 1.000000 | 1.0 | 1.492938 | 1.486926 | 1.5 |

Quadratic-forcing powers match `a+2` at the tested precision. All 12 selected
case/order summaries satisfy the stated coefficient and power checks. The
nonlinear values need not equal the limiting powers exactly at finite `h`;
no tolerance was interpreted as a general theorem about all states or orders.

## A zero-indicator cubic counterexample

On `[0,1]`, choose the smooth, nonnegative, nondecreasing forcing:

```text
D_C^a y = f(t) = t^2 - (2/3)*t^3,   y(0)=0.
```

Its values at `0`, `1/2`, and `1` are `0`, `1/6`, and `1/3`, lying on the same
line `t/3`. Both the coarse and two-half-panel interpolants are therefore that
line. For every `0<a<1`:

```text
C = F = 1/[3*Gamma(a+2)]
y(1) = 2/Gamma(a+3) - 4/Gamma(a+4)
C-y(1) = a*(a-1)/[3*(a+2)*(a+3)*Gamma(a+2)] != 0.
```

Thus a finite multiplicative rescaling of `abs(C-F)` still gives zero while the
true error is nonzero. Changing only the controller exponent also cannot detect
this first-trial error. This is a sampling/interpolation blind spot, not roundoff
or accumulated history.

The float64 executable probes produced exactly zero pair differences for all
three tested orders:

| a | True coarse absolute error | Actual adaptive accepted/rejected steps |
| --- | --- | --- |
| 0.25 | 0.00754367625 | 1 / 0 |
| 0.5 | 0.00716431217 | 1 / 0 |
| 0.75 | 0.00376819135 | 1 / 0 |

These calls explicitly set `first_step=1`, `rtol=1e-8`, `atol=1e-10`, and
`max_steps=2`; they do not use the default initial step. Each returned operational
success. This is an existence counterexample, not a claim that every step on this
forcing or every adaptive run has a zero indicator.

## Implications for the correction

The derived gains are useful within their assumptions, but the blind spot rules
out repairing general acceptance by a scalar rescaling alone. The next numerical
prototype should obtain additional independent information (for example an
extra subdivision/defect probe), explicitly test for pair cancellation, and
validate its behavior under weak startup and state feedback. Adding samples by
itself still does not establish a universal bound for unrestricted forcing.

Before promotion, define the controlled quantity and admissible problem class,
show case-specific calibration evidence, retain explicit old-history limitations,
and rerun JIT, frozen-mesh gradients, dtype and failure-status checks. A new
controller should be named/opt-in until the behavior and compatibility decision
is reviewed. This audit changes no numerical library implementation.

## Artifacts

- [Summary CSV](audits/adaptive-startup/adaptive_startup_float64.csv).
- [Raw measurements, cubic probes, and provenance](audits/adaptive-startup/adaptive_startup_float64.json).

These identify dirty development source using hashes; they are not a tagged
release bundle and cannot reconstruct uncommitted source from fingerprints alone.
The earlier calibration artifacts are left intact.
