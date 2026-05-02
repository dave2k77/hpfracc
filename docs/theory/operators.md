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

`hp.ops.riemann_liouville(x, dt=..., order=...)` currently uses the GL
full-history discretisation as the baseline uniform-grid approximation to the
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
