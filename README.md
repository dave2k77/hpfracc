# HPFRACC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20942679.svg)](https://doi.org/10.5281/zenodo.20942679)

HPFRACC is a pre-alpha research-support library for fractional calculus and
fractional dynamical systems in differentiable scientific computing.

The v0.1 alpha line prioritizes numerical correctness, numerical stability,
differentiability, and reproducibility before domain-specific phantom-brain or
brain-model workflows.

HPFRACC is research software only. It is not clinical, diagnostic, or
subject-specific decision software.

## Scope

The current implementation targets:

- JAX-native fractional operators.
- Riemann-Liouville, Caputo, and Grunwald-Letnikov operator families.
- Uniform explicit/implicit, nonuniform, and adaptive Caputo FDE solvers.
- Experimental differentiable Neural FODE workflows.
- Experimental scalar-grid probabilistic calibration and additive-noise
  stochastic simulation helpers.
- CPU and single accelerator execution.
- Explicit state, configuration, provenance, and validation-status objects.

## Current Surface

Active phase: Phase A

Latest tagged release: `0.1.0a1` (2026-06-26).

Further core hardening is unreleased. See the [current development state](docs/developer/roadmap.md#current-development-state)
and [validation boundary](docs/validation/status.md) for supported combinations
and remaining gates. Per-state orders are supported by uniform operators, not
the scalar-order solvers; nonuniform Caputo is scalar-order/full-history only.

The examples below start with the uniform full-history operator and explicit
fixed-step Caputo solver baseline:

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
    return params["rate"] * state

solver = hp.solvers.PredictorCorrector(dt=dt, order=0.7)
solution = hp.solvers.simulate(
    model=f,
    ts=t,
    solver=solver,
    initial_state=jnp.asarray(1.0),
    params={"rate": jnp.asarray(-0.8)},
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
    params={"rate": jnp.asarray(-0.8)},
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

## Development Environment

Use the project `uv` environment:

```bash
uv sync --locked --extra dev
```

The lockfile is committed so release-readiness checks can run from a reproducible
development environment.

Ruff is included in the dev extra. CI uses the lockfile for its Python 3.11–3.13
CPU checks, lint, and strict docs. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
matching local workflow.

## Validation

Run the test suite:

```bash
uv run --frozen python -m pytest
```

Run the aggregate validation summary:

```bash
uv run --frozen python -m benchmarks.numerical.validation_summary --check
```

Run detailed operator and solver validation reports:

```bash
uv run python -m benchmarks.numerical.operator_validation.report
uv run python -m benchmarks.numerical.solver_validation.report
```

Run the lightweight operator scaling smoke benchmark:

```bash
uv run python -m benchmarks.numerical.operator_scaling
```

Run the CPU-oriented baseline benchmark:

```bash
uv run python -m benchmarks.numerical.baseline
```

Build the documentation:

```bash
uv run --frozen mkdocs build --strict
```

See `docs/validation/status.md` for the current v0.1 alpha validation boundary
and `docs/developer/release-checklist.md` for the release candidate checklist.
The [capability/evidence matrix](docs/validation/capabilities.md) maps supported
grid/order/history combinations to their tests and remaining gaps.

## Research Use

HPFRACC is research software. It is not clinical software, is not validated for
diagnosis or treatment, and should not be used as a substitute for empirical
biomedical assessment.

See [RESEARCH_USAGE.md](RESEARCH_USAGE.md) for the project research-use policy.
