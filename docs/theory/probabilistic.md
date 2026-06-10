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

The diffusion callable uses the same signature as deterministic dynamics:

```python
g(t, state, params, *, rng_key=None, inputs=None)
```

This surface is explicitly experimental. It uses an Euler-Maruyama-style
additive perturbation and should not be cited as a validated stochastic
fractional numerical method.

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
