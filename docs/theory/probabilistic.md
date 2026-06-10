# Probabilistic Workflows

HPFRACC v0.1 includes an experimental probabilistic layer for small,
reproducible research workflows after the deterministic solver and model layers
are validated.

## Experimental Stochastic Simulation

`hp.prob.simulate_stochastic` provides an additive-noise stochastic extension of
the full-history Caputo predictor-corrector solver:

```python
result = hp.prob.simulate_stochastic(
    model=f,
    diffusion=g,
    ts=ts,
    solver=solver,
    initial_state=y0,
    params=params,
    rng_key=key,
)
```

for `D_C^alpha y(t) = f(t, y) + g(t, y) xi(t)` with Gaussian white noise `xi`.
The diffusion callable uses the same signature as deterministic dynamics:

```python
g(t, state, params, *, rng_key=None, inputs=None)
```

In integral form the stochastic term is the Riemann-Liouville fractional
integral of the noise. It is discretised as a fractional Euler-Maruyama step in
which each Brownian increment is weighted by the exact per-interval kernel
variance

```text
w_{j,n} = integral_{t_j}^{t_{j+1}} (t_n - s)^(2 alpha - 2) ds
```

so that the simulated variance matches the analytic variance of the linear
additive FSDE,

```text
Var(y(t)) = sigma^2 * integral_0^t [tau^(alpha-1) E_{alpha,alpha}(lambda tau^alpha)]^2 dtau.
```

The solver is restricted to `alpha > 1/2`: for `alpha <= 1/2` the stochastic
fractional integral of white noise is singular and the variance integral
diverges, so `simulate_stochastic` raises `ValueError` there.

This surface remains experimental. The variance match is validated for the
linear additive case; general nonlinear drift and state-dependent (multiplicative)
diffusion are not yet validated beyond reproducibility and mean behaviour.

## Calibration

`hp.prob.grid_calibrate_scalar` evaluates a scalar parameter grid under an
independent Gaussian observation model. It returns log likelihoods, normalized
posterior weights, and the best grid value.

`hp.prob.posterior_predictive` evaluates trajectories over a scalar parameter
grid and returns trajectory samples, a posterior-weighted mean, and
posterior-weighted credible intervals. The intervals are quantiles of the
discrete grid-weighted predictive distribution via `hp.prob.weighted_quantile`,
so reweighting the grid changes the band; a near-degenerate posterior collapses
the band onto the dominant trajectory.

## Assumptions

- Observational errors are independent Gaussian errors with a user-supplied
  positive `noise_scale`.
- Calibration currently targets scalar grid searches.
- Posterior predictive intervals are quantiles of the discrete grid-weighted
  predictive distribution, not unweighted grid quantiles. Because that
  distribution is discrete, the bands are step-valued between grid points rather
  than smoothly interpolated.
- NumPyro integration is deferred until these small contracts stabilize.
