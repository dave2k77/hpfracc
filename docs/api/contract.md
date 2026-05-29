# API Contract

The canonical import is:

```python
import hpfracc as hp
```

The v0.1 public namespace starts small: `hp.ops`, `hp.solvers`, `hp.config`,
`hp.typing`, `hp.metrics`, and `hp.experimental`.

HPFRACC is pre-alpha research software. Public APIs are intentionally small and
carry explicit stability tiers so users can decide what to depend on.

## API Stability Tiers

| Tier | Meaning | Compatibility expectation |
| --- | --- | --- |
| Stable | Concepts that should remain recognizable across v0.x releases. | Changes should be rare and documented with migration notes. |
| Provisional | Usable v0.1 APIs whose signatures or metadata may change as validation improves. | Breaking changes are allowed before a stable release, but should be documented in the changelog. |
| Experimental | Exploratory APIs for differentiable, probabilistic, stochastic, or domain-specific workflows. | May change or be removed between minor alpha releases. Do not treat as stable research infrastructure. |
| Private | Implementation details, underscored helpers, and modules not listed in this contract. | No compatibility guarantee. |

Current assignments:

| API surface | Tier | Notes |
| --- | --- | --- |
| `import hpfracc as hp` | Stable | Canonical package import form. |
| `hp.ops`, `hp.solvers`, `hp.config`, `hp.typing`, `hp.metrics`, `hp.experimental` namespace names | Stable | Namespace names are part of the v0.1 package shape. Individual objects may be provisional or experimental. |
| Research-use-only project policy | Stable | HPFRACC is not clinical, diagnostic, or subject-specific decision software. |
| `hp.ops.caputo`, `hp.ops.riemann_liouville`, `hp.ops.grunwald_letnikov` | Provisional | Full-history reference implementations for v0.1 numerical validation. |
| `hp.ops.OperatorResult`, `hp.ops.OperatorInfo`, `hp.ops.OperatorFamily`, `hp.ops.FractionalOrder` | Provisional | Structured result and metadata contracts may evolve with provenance needs. |
| `hp.solvers.PredictorCorrector`, `hp.solvers.simulate` | Provisional | First fixed-step Caputo FDE solver surface. |
| `hp.solvers.SimulationResult`, `hp.solvers.SolverInfo` | Provisional | Result metadata is public but may expand during Phase 6 provenance work. |
| `hp.config.Provenance`, `hp.config.ExperimentMetadata`, `hp.config.RuntimeTarget` | Provisional | Intended for reproducibility metadata; fields may be refined before v0.1 alpha finalization. |
| `hp.nn` | Experimental | Differentiable Neural FODE wrapper and training utilities. |
| `hp.prob` | Experimental | Scalar-grid calibration, posterior-predictive utilities, and additive-noise FSDE helpers. |
| `hp.experimental` | Experimental | Staging namespace for APIs that are not ready for the main contract. |
| Placeholder domain namespaces such as `hpfracc.brain`, `hpfracc.observe`, `hpfracc.train`, `hpfracc.data`, and `hpfracc.viz` | Experimental | Importable planning stubs only; no biological or clinical claims. |
| Underscored modules, helper functions, and implementation details | Private | Use documented namespace exports instead. |

The code-level enum `hpfracc.stability.StabilityTier` records these tier names
for internal documentation and tests. It is not exported at top level in v0.1;
users should treat the table above as the public stability contract.

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
