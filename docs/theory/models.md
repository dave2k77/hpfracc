# Differentiable Models

HPFRACC v0.1 keeps the first model layer deliberately small. Trainable models
use explicit JAX pytrees for parameters and regular Python callables for
dynamics.

## Neural FODE Wrapper

`hp.nn.NeuralFODE` wraps a dynamics function and a fixed-step solver:

```python
model = hp.nn.NeuralFODE(dynamics=f, solver=solver)
result = model(
    ts=ts,
    initial_state=y0,
    params=params,
)
```

The dynamics callable follows the same convention as the solver:

```python
f(t, state, params, *, rng_key=None, inputs=None)
```

Parameters are explicit pytrees. This keeps gradients visible and avoids
committing v0.1 to a neural-network framework before the numerical core APIs
stabilize.

## Losses and Updates

`hp.nn.trajectory_mse` computes mean-squared error between a model trajectory
and a target trajectory. `hp.nn.sgd_step` applies a simple pytree SGD update for
small examples and validation tests.

These helpers are intentionally minimal. Production training loops should own
optimizer state and experiment management outside the numerical core.

## Current Limitations

- No Equinox dependency is required in v0.1.
- No optimizer package is required in v0.1.
- The model layer is experimental and intended to validate gradient flow through
  parameters and initial conditions.
