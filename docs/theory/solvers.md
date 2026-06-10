# Fractional Solvers

HPFRACC v0.1 starts with deterministic Caputo initial value problems:

```text
D_C^alpha y(t) = f(t, y(t), params)
y(0) = y0
0 < alpha < 1
```

## Predictor-Corrector Solver

`hp.solvers.PredictorCorrector` configures a fixed-step full-history
predictor-corrector method. `hp.solvers.simulate` returns a `SimulationResult`
with the time grid, latent trajectory, and `SolverInfo` diagnostics.

The time-stepping recurrence is evaluated with a single `jax.lax.scan` over a
preallocated history buffer rather than an unrolled Python loop. The solver is
therefore `jax.jit`-traceable with a bounded autodiff graph whose size does not
grow with the number of time steps, which keeps compilation and reverse-mode
gradients tractable for long trajectories. The arithmetic cost is unchanged: it
remains `O(n**2)` because each step weights the full history.

The dynamics callable should accept:

```python
f(t, state, params, *, rng_key=None, inputs=None)
```

The returned trajectory always has the time axis first. Scalar initial states
produce shape `(time,)`; array initial states preserve arbitrary trailing
dimensions as `(time, ...)`.

## Validation

The first validation target is the scalar linear Caputo FDE:

```text
D_C^alpha y(t) = lambda y(t)
y(0) = 1
```

Its reference solution is:

```text
y(t) = E_alpha(lambda t^alpha)
```

where `E_alpha` is approximated by a truncated Mittag-Leffler series in tests
and validation reports.

## Limitations

- Only scalar `0 < alpha < 1` orders are supported.
- Only fixed, positive, uniform timesteps are supported.
- The method uses full history, so runtime grows quadratically with the number
  of time samples.
- Adaptive timesteps, implicit methods, stochastic FSDEs, memory compression,
  and multi-device execution are deferred.
