# Fractional Operators

HPFRACC v0.1 provides Riemann-Liouville, Caputo, and Grunwald-Letnikov operator
families with orders `0 < alpha < 1`. Uniform grids support scalar or per-state
orders; Caputo additionally supports nonuniform grids with scalar order.

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

### Nonuniform Caputo L1

`hp.ops.caputo(x, t=nodes, order=...)` uses the actual, strictly increasing time
nodes in the full-history L1 product integral. Supply exactly one of `t` or `dt`.
This path supports scalar order only.

Power differences on narrow historical intervals use the shared
[stable interval moments](solvers.md#graded-mesh-precision), rather than
subtracting nearly equal lag powers. This retains early-interval contributions
on graded meshes without changing the L1 discretization or enabling x64 globally.

## Limitations

- Nonuniform grids support only scalar-order Caputo with full history. GL/RL
  remain uniform-grid operators; vector orders are not supported by SOE.
- Full-history matrices have quadratic memory cost. The opt-in FFT path changes
  that cost on uniform grids. Short-memory now uses direct windowed convolution;
  SOE now applies fitted exponentials recursively. See the
  [windowed-history contract and evidence](../validation/windowed-history.md) for
  the truncation bound, and the [SOE contract](../validation/soe-history.md) for
  legacy/scaled-tail fitting, the GL-prefix identity, complexity and error limits.
- Order gradients have case-specific validation, not a guarantee for every
  history strategy or numerical regime; see the validation status page.
