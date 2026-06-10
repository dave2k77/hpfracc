# Fractional Operators

HPFRACC v0.1 targets Riemann-Liouville, Caputo, and Grunwald-Letnikov operator
families on uniform grids for scalar orders `0 < alpha < 1`.

Each implementation must document its mathematical definition, discretisation,
boundary convention, differentiability status, and known failure modes.

## v0.1 Discretisations

The first operator implementations use explicit full-history uniform-grid
schemes. They are intended as validation baselines before optimized memory
strategies are introduced.

### Grunwald-Letnikov

`hp.ops.grunwald_letnikov(x, dt=..., order=...)` computes the full-history GL
convolution along the leading time axis. The implementation supports arbitrary
trailing state dimensions.

### Riemann-Liouville

`hp.ops.riemann_liouville(x, dt=..., order=...)` computes the lower-terminal
Riemann-Liouville derivative *through* the Grunwald-Letnikov full-history
discretisation, which is first-order (`O(dt)`) consistent with the RL derivative
on a uniform grid with zero history before `t = 0`. RL and GL therefore share an
implementation by design.

This is validated against the analytic RL reference, not against the identical
GL code path:

```text
D_RL^alpha t**beta = Gamma(beta + 1) / Gamma(beta + 1 - alpha) * t**(beta - alpha)
```

available as `hp.ops.riemann_liouville_power_law(t, power=..., order=...)`. The
decisive case is the constant `beta = 0`, where the RL derivative is

```text
t**(-alpha) / Gamma(1 - alpha)
```

which is nonzero and singular at `t = 0`. This is the term by which RL and
Caputo differ (`D_RL^alpha f = D_C^alpha f + f(0) t**(-alpha) / Gamma(1 - alpha)`)
and it is the reason a constant has a zero Caputo derivative but a nonzero
Riemann-Liouville derivative.

### Caputo

`hp.ops.caputo(x, dt=..., order=...)` computes the L1 Caputo derivative along
the leading time axis. The first output sample is zero because no history
interval has elapsed at `t=0`.

For validation, `hp.ops.caputo_power_law(t, power=..., order=...)` provides the
analytic Caputo derivative of `t**power` for nonnegative powers:

```text
Gamma(beta + 1) / Gamma(beta + 1 - alpha) * t**(beta - alpha)
```

For constants, the Caputo derivative is zero.

## Limitations

- Only scalar `0 < alpha < 1` orders are supported.
- Only positive uniform timesteps are supported.
- Full-history matrices are used, so memory use grows quadratically with the
  number of time samples.
- Gradients with respect to `order` are provisional; gradients with respect to
  input samples are the first validation target.
