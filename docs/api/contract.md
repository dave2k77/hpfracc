# API Contract

The canonical import is:

```python
import hpfracc as hp
```

The v0.1 public namespace starts small: `hp.ops`, `hp.solvers`, `hp.config`,
`hp.typing`, `hp.metrics`, and `hp.experimental`.

## Operator Results

Fractional operators return raw arrays by default:

```python
dx = hp.ops.caputo(x, dt=0.01, order=0.5)
```

When provenance or method metadata is needed, pass `return_info=True`:

```python
result = hp.ops.caputo(x, dt=0.01, order=0.5, return_info=True)
dx = result.values
method = result.operator_info.method
```

The structured result form is intended for validation, benchmarking, and
research reporting. The array-return form remains the ergonomic default for
numerical workflows.

## Solver Results

The first solver surface is a fixed-step Caputo predictor-corrector API:

```python
solver = hp.solvers.PredictorCorrector(dt=0.01, order=0.7)
result = hp.solvers.simulate(
    model=f,
    ts=ts,
    solver=solver,
    initial_state=y0,
    params=params,
)
```

Dynamics functions should accept:

```python
f(t, state, params, *, rng_key=None, inputs=None)
```

`result.latent_state` stores the simulated trajectory with the leading time
axis. `result.solver_info` records the method name, fractional order, step
size, step count, and diagnostics.

## Model Results

The first differentiable model wrapper is `hp.nn.NeuralFODE`:

```python
model = hp.nn.NeuralFODE(dynamics=f, solver=solver)
loss = hp.nn.trajectory_mse(
    model,
    ts=ts,
    initial_state=y0,
    params=params,
    target=target,
)
```

Model parameters are explicit JAX pytrees. The wrapper returns the same
`SimulationResult` contract as `hp.solvers.simulate`, with optional observed
trajectories when an observation transform is supplied.

## Probabilistic Results

The experimental probabilistic namespace exposes small calibration and
posterior-predictive contracts:

```python
calibration = hp.prob.grid_calibrate_scalar(
    model,
    ts=ts,
    initial_state=y0,
    observations=observed,
    parameter_name="rate",
    parameter_grid=grid,
    noise_scale=0.05,
)
predictive = hp.prob.posterior_predictive(
    model,
    ts=ts,
    initial_state=y0,
    parameter_name="rate",
    parameter_grid=calibration.parameter_grid,
    weights=calibration.posterior_weights,
)
```

`hp.prob.simulate_stochastic` returns `SimulationResult` with an experimental
warning in `solver_info`.
