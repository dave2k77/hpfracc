# HPFRACC

HPFRACC is a research-support library for fractional calculus and fractional
dynamical systems in differentiable scientific computing.

The v0.1 line prioritizes numerical correctness, numerical stability,
differentiability, and reproducibility before domain-specific phantom-brain
workflows.

## Scope

The initial implementation targets:

- JAX-native fractional operators.
- Riemann-Liouville, Caputo, and Grunwald-Letnikov operator families.
- Fixed-step Caputo fractional differential equation solvers.
- CPU and single GPU/TPU execution.
- Explicit state, configuration, and provenance objects.

## Current Surface

The current pre-alpha implementation includes baseline full-history fractional
operators on uniform grids and a fixed-step Caputo FDE solver:

```python
import jax.numpy as jnp
import hpfracc as hp

dt = 0.01
t = jnp.arange(101) * dt
x = t**2

dx = hp.ops.caputo(x, dt=dt, order=0.5)
```

Use `return_info=True` when validation or reporting code needs method metadata:

```python
result = hp.ops.caputo(x, dt=dt, order=0.5, return_info=True)
print(result.operator_info.to_dict())
```

```python
def f(t, state, params, *, rng_key=None, inputs=None):
    return params * state

solver = hp.solvers.PredictorCorrector(dt=dt, order=0.7)
solution = hp.solvers.simulate(
    model=f,
    ts=t,
    solver=solver,
    initial_state=jnp.asarray(1.0),
    params=jnp.asarray(-0.8),
)
```

The experimental differentiable model layer wraps solver-backed dynamics with
explicit parameter pytrees:

```python
model = hp.nn.NeuralFODE(dynamics=f, solver=solver)
loss = hp.nn.trajectory_mse(
    model,
    ts=t,
    initial_state=jnp.asarray(1.0),
    params=jnp.asarray(-0.8),
    target=solution.latent_state,
)
```

The experimental probabilistic layer supports scalar-grid calibration and
posterior predictive summaries:

```python
calibration = hp.prob.grid_calibrate_scalar(
    model,
    ts=t,
    initial_state=jnp.asarray(1.0),
    observations=solution.latent_state,
    parameter_name="rate",
    parameter_grid=jnp.linspace(-1.2, -0.2, 11),
    noise_scale=0.05,
)
```

## Validation

Run the test suite:

```bash
python -m pytest
```

Run the Phase 1 operator validation report:

```bash
python -m benchmarks.numerical.operator_validation.report
```

Run the lightweight operator scaling smoke benchmark:

```bash
python -m benchmarks.numerical.operator_scaling
```

Run the Phase 2 solver validation report:

```bash
python -m benchmarks.numerical.solver_validation.report
```

Run the aggregate Phase 3 validation summary:

```bash
python -m benchmarks.numerical.validation_summary
```

Run the CPU-oriented baseline benchmark:

```bash
python -m benchmarks.numerical.baseline
```

Build the documentation:

```bash
mkdocs build --strict
```

## Research Use

HPFRACC is research software. It is not clinical software, is not validated for
diagnosis or treatment, and should not be used as a substitute for empirical
biomedical assessment.

See [RESEARCH_USAGE.md](RESEARCH_USAGE.md) for the project research-use policy.
