# Quickstart

HPFRACC is pre-alpha research software for fractional calculus and fractional
differential equation experiments. It is not clinical, diagnostic, or
subject-specific decision software.

Run examples from the repository root with the project `uv` environment:

```bash
uv sync --extra dev
```

The canonical import is:

```python
import hpfracc as hp
```

## First Operator Call

Evaluate a full-history Caputo derivative for `x(t) = t^2`:

```python
import jax.numpy as jnp
import hpfracc as hp

dt = 0.01
t = jnp.arange(101) * dt
x = t**2

dx = hp.ops.caputo(x, dt=dt, order=0.5)
print(dx[-1])
```

Use `return_info=True` when a validation or research workflow needs method
metadata:

```python
result = hp.ops.caputo(x, dt=dt, order=0.5, return_info=True)
print(result.values[-1])
print(result.operator_info.to_dict())
```

The metadata records the operator family, method, fractional order, time step,
history policy, diagnostics, and warnings.

A runnable version is available at:

```bash
uv run python examples/fractional_ode/caputo_operator.py
```

## First Caputo FDE Simulation

Define a scalar linear model with the v0.1 dynamics signature:

```python
import jax.numpy as jnp
import hpfracc as hp


def linear_model(t, state, params, *, rng_key=None, inputs=None):
    del t, rng_key, inputs
    return params * state


dt = 0.01
ts = jnp.arange(101) * dt
solver = hp.solvers.PredictorCorrector(dt=dt, order=0.7)
result = hp.solvers.simulate(
    model=linear_model,
    ts=ts,
    solver=solver,
    initial_state=jnp.asarray(1.0),
    params=jnp.asarray(-0.8),
)

print(result.latent_state[-1])
print(result.solver_info.to_dict())
```

`result.latent_state` stores the simulated trajectory with the leading time
axis. `result.solver_info` records the method name, fractional order, step size,
step count, diagnostics, and warnings.

A runnable version is available at:

```bash
uv run python examples/fractional_ode/caputo_solver.py
```

## Experimental Training and Calibration Examples

The differentiable and probabilistic namespaces are experimental in v0.1 alpha.
They are useful for testing result contracts and research workflows, but their
APIs may change before a stable release.

Run the synthetic Neural FODE training example:

```bash
uv run python examples/fractional_ode/train_neural_fode.py
```

Run the scalar-grid probabilistic calibration example:

```bash
uv run python examples/fractional_ode/probabilistic_calibration.py
```

See `examples/fractional_ode/README.md` for expected high-level outputs from
these smoke tests.

## API Stability

The public API is intentionally small for v0.1. See
[API Contract](../api/contract.md) for stability tiers and current assignments.
In short:

- operators and deterministic solver result contracts are provisional;
- `hp.nn` and `hp.prob` are experimental;
- underscored modules and unlisted helpers are private.
