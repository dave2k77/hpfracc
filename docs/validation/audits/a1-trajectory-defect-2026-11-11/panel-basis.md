# Positive panel-basis diagnostic: exact model and arithmetic limits

This is an offline reference construction, not a numerical certificate or
controller. The stored trajectory is reconstructed locally, not by summing large
opposite global slopes. Keep the original lower terminal and every source panel.

For a source panel `[l,r]`, width `h`, target `t>l`, let `tau=t-l`,
`c=min(tau,h)`, `q=c/tau`, and `m=tau**alpha/Gamma(alpha+1)`. Integrate the two
nonnegative linear basis functions on the clipped panel. For a target inside the
panel, the clipped weights are `m*alpha/(alpha+1)` and `m/(alpha+1)`. Convert to
the original endpoint basis with `f=c/h`: original right weight is `f*R`, original
left weight is `L+(1-f)*R`.

For an old panel and `q<1/2`, the clipped left/right weights are respectively

- `m*q*alpha/2 * hyp2f1(1-alpha,1,3,q)`;
- `m*q*alpha/2 * hyp2f1(1-alpha,2,3,q)`.

These follow by integrating the positive kernel against `1-z` and `z` on the
unit panel. Direct special-function evaluation near `q=1` was rejected after a
preimplementation probe showed relative error above `2e-5`. The failed probe is
preserved. Instead compute `z=(t-r)/(t-l)` independently of `1-q`, `v=log(z)`,
and use the equivalent expressions

- `L=m/q * (alpha/(alpha+1) + z*expm1(alpha*v-log1p(alpha)))`;
- `R=m/q * (-expm1(alpha*v) + alpha/(alpha+1)*expm1((alpha+1)*v))`.

Do not infer a zero old-panel age merely because `q` rounded to one. The local
kernel clock and actual endpoint comparisons determine the active branch.
Eighteen separate 130-digit panel-integral checks corroborate the repaired
formula on selected orders and kernel limits; this is not a global error bound.

## Continuum-cell model ranges

Split verification cells at all original source knots. After a panel ends, both
basis weights decrease with target time because the positive kernel decreases
with age for `0<alpha<1`. Within the active panel, its right endpoint weight
increases. Its left endpoint weight has derivative proportional to
`tau**(alpha-1)*(alpha-tau/h)`, hence its maximum is at `tau=alpha*h`; retain that
interior critical point, not only endpoint samples. The PL curve itself has
component extrema at cell endpoints. Interval-add the signed `A@y_j` basis
contributions and the monotone analytic forcing powers.

This encloses the ideal model in exact real arithmetic. Gamma, matrix products,
powers, local interpolation and sums are not outward rounded here. Both
verification/certification flags remain false. The fixed adjacent-float forcing
counterexample remains a limitation regression.

## Explicit failure rather than silent loss

The host paths trap intermediate underflow as well as overflow, invalid and
divide errors. In particular, `alpha=.5`, `beta=1100`, forcing coefficient `1e300`
at `t=.5` has a representable final integral, but ordinary time-power evaluation
underflows before coefficient scaling. This implementation rejects that unsafe
calculation explicitly; it does not claim to compute the rescued value. This
conservative arithmetic limitation is not a scope-deferral approval, a numerical
certificate or a closed Phase A gate. No global precision promotion is used.
