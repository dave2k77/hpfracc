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
| `hp.solvers.ImplicitPredictorCorrector` | Provisional | Fixed-step implicit corrector with configurable residual diagnostics; fixed Newton iteration count. |
| `hp.solvers.AdaptivePredictorCorrector` | Provisional | Adaptive explicit PECE with bounded attempts, padded output, and JIT-accessible completion/failure diagnostics. |
| `hp.solvers.SimulationResult`, `hp.solvers.SolverInfo` | Provisional | Result metadata is public but may expand during Phase 6 provenance work. |
| `hp.config.Provenance`, `hp.config.ExperimentMetadata`, `hp.config.RuntimeTarget`, `hp.config.current_provenance` | Provisional | Intended for reproducibility metadata; fields may be refined before v0.1 alpha finalization. |
| `hp.nn` | Experimental | Differentiable Neural FODE wrapper and training utilities. |
| `hp.prob` | Experimental | Scalar-grid calibration, posterior-predictive utilities, and additive-noise FSDE helpers. |
| `hp.experimental` | Experimental | Staging namespace for APIs that are not ready for the main contract. |
| Placeholder domain namespaces such as `hpfracc.brain`, `hpfracc.observe`, `hpfracc.train`, `hpfracc.data`, and `hpfracc.viz` | Experimental | Importable planning stubs only; no biological or clinical claims. |
| Underscored modules, helper functions, and implementation details | Private | Use documented namespace exports instead. |

## Provenance Metadata

Use `hp.config.current_provenance()` when a script, validation run, or report
needs runtime context:

```python
provenance = hp.config.current_provenance(
    config={"case": "operator_validation"},
)
print(provenance.to_dict())
```

The helper captures:

- HPFRACC package version;
- best-effort git commit hash, or `None` when git is unavailable;
- JAX backend and runtime target;
- JAX version and visible devices;
- Python version and platform string;
- UTC timestamp;
- caller-supplied configuration values.

`Provenance.to_dict()` returns a JSON-compatible dictionary. Default
`Provenance()` construction remains available for deterministic tests and for
callers that want to supply every field explicitly.

The code-level enum `hpfracc.stability.StabilityTier` records stability tier
names for internal documentation and tests. It is not exported at top level in
v0.1; users should treat the stability table above as the public contract.

## Operator Results

Fractional operators return raw arrays by default:

```python
dx = hp.ops.caputo(x, dt=0.01, order=0.5)
```

The optional ``history`` argument selects the history convolution strategy while
keeping the same output contract.  ``history="full"`` is the default and uses the
explicit dense lower-triangular reference.  ``history="fft"`` uses an
FFT-accelerated causal convolution that is mathematically equivalent on uniform
grids but ``O(n log n)`` instead of ``O(n^2)``.  ``history="short_memory"``
truncates the kernel to the most recent ``window_steps`` lags and, in the current
development tree, applies direct windowed convolution without a dense history
matrix. See the [approximation contract](../validation/windowed-history.md).
``history="soe"`` fits a sum-of-exponentials approximation to the discrete
kernel using ``soe_poles`` exponential terms, now applied through recursive
histories. The new optional ``soe_fit="scaled_tail"`` retains the first weight
exactly and fits positive L1 or cumulative-GL tails; ``soe_fit="legacy"`` retains
the released fitting policy. See the [SOE contract and evidence](../validation/soe-history.md):

```python
dx = hp.ops.caputo(x, dt=0.01, order=0.5, history="fft")
dx = hp.ops.caputo(x, dt=0.01, order=0.5, history="short_memory", window_steps=64)
dx = hp.ops.caputo(
    x,
    dt=0.01,
    order=0.5,
    history="soe",
    soe_poles=12,
    soe_t_max=1.0,
    soe_fit="scaled_tail",
)
```

New history methods remain opt-in and must be validated against the
``"full"`` reference before they are used as defaults.

When provenance or method metadata is needed, pass `return_info=True`:

```python
result = hp.ops.caputo(x, dt=0.01, order=0.5, history="fft", return_info=True)
dx = result.values
method = result.operator_info.method
history = result.operator_info.history
```

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

### Uniform Weight Evaluation

Both `hp.solvers.PredictorCorrector` and `hp.solvers.ImplicitPredictorCorrector`
accept `weight_evaluation="direct"` (default) or `weight_evaluation="stable"`.
The latter evaluates the same full-history coefficients through stable interval
moments. The selected value is recorded as `weight_evaluation` in
`solver_info.diagnostics`. Unknown values raise `ValueError`. See the
[uniform-weight contract and evidence](../validation/uniform-weights.md).

### Implicit Corrector Diagnostics

`hp.solvers.ImplicitPredictorCorrector` accepts `newton_iterations=3`,
`newton_rtol=1e-6`, and `newton_atol=1e-8`. Both tolerances must be finite and
positive. They assess the final algebraic residual, not the trajectory error,
and do not change the iteration count or trigger retries.

The following fields are in `result.solver_info.diagnostics`:

| Field | Meaning |
| --- | --- |
| `newton_converged` | Every attempted corrector step met the residual criterion. |
| `newton_failure_count` | Number of steps failing the criterion, including non-finite results. |
| `newton_nonfinite_count` | Steps with non-finite state, dynamics, or residual normalization. |
| `newton_step_converged` | Per-step flags; index zero corresponds to `ts[1]`. |
| `newton_scaled_residuals` | Per-step maximum componentwise scaled residuals. Values at most one pass. |
| `newton_max_scaled_residual` | Maximum over all steps; zero if no Newton steps were taken. |

Eager diagnostics use JSON-compatible scalars/lists. Non-finite residuals are
represented by `None` (JSON null), with explicit failure flags and counts.
Failures attach a message to `result.solver_info.warnings`; they do not raise
an exception or emit a Python warning. Under JIT, the numerical diagnostic
fields remain fixed-shape JAX arrays and non-finite residuals use positive
infinity. Compiled callers should return the flags alongside the trajectory;
data-dependent warning text is only available eagerly:

```python
def checked_simulation(params):
    result = hp.solvers.simulate(
        model=f, ts=ts, solver=implicit_solver, initial_state=y0, params=params,
    )
    return result.latent_state, result.solver_info.diagnostics["newton_converged"]
```

A single-node grid has no attempted Newton steps, so its convergence flag is
vacuously true. The flag is not a certificate of initial-state validity, root
uniqueness, conditioning, biological realism, or global integration accuracy.
See [the residual criterion](../theory/solvers.md#implicit-residual-criterion).

### Experimental Adaptive Subdivision Indicator

The default remains `error_estimator="step_doubling"`. Select the unreleased,
experimental safeguard explicitly:

```python
adaptive_solver = hp.solvers.AdaptivePredictorCorrector(
    order=0.5, rtol=1e-4, atol=1e-6, max_steps=512,
    error_estimator="subdivision",
)
```

It compares full, half, and quarter PECE trial endpoints with a maximum
componentwise normalized discrepancy, without Richardson rescaling, and uses
bounded-factor step proposals. It still retains the coarse solution on the
chosen mesh. The new mode detects the tested cubic pair-cancellation case, but
is not a calibrated local/global error bound. See
[the prototype and measured comparison](../validation/adaptive-subdivision.md).
Unknown estimator names raise `ValueError`; default formulas and public output
shape are unchanged.

### Adaptive Completion and Failure Status

`hp.solvers.AdaptivePredictorCorrector` treats `ts[0]` and `ts[-1]` as integration
bounds. It returns arrays of length `max_steps + 2`, padding unused entries with
the final retained time/state. `max_steps` is an **attempt** budget, including
rejected and terminating failed trials, not a requested number of output nodes.

Check `result.solver_info.diagnostics` before using a trajectory:

| Field | Meaning |
| --- | --- |
| `success`, `reached_t_end` | Completed at the requested endpoint with finite accepted trials. |
| `max_steps_reached` | Attempt budget exhausted before another terminal condition. False if the final allowed attempt succeeds. |
| `nonfinite_detected` | Initial state/dynamics/span or trial state/dynamics/error normalization was non-finite. |
| `step_size_underflow` | Lossy integer-bound conversion, unrepresentable required interior trial nodes (midpoint or quarters), or tolerance failure at the small-step threshold. |
| `n_accepted`, `n_rejected`, `n_attempted` | Counts satisfying `n_attempted = n_accepted + n_rejected`. |
| `n_nodes` | Initial node plus accepted steps. `solver_info.n_steps` is the accepted-step count. |
| `valid_mask` | Boolean array of length `max_steps + 2` selecting the initial node and accepted prefix. |
| `last_time`, `requested_t_end` | Final retained time and requested endpoint. |
| `time_bounds_representable` | Integer bounds survive conversion to the floating time dtype exactly; floating bounds require no integer conversion. |
| `minimum_step` | Small-step failure threshold, `1e-12 * abs(ts[-1] - ts[0])`. |
| `estimator`, `estimator_stability` | Selected indicator and its provisional/default or experimental/subdivision status. |
| `error_norm`, `trial_substeps` | `rms`, `2` for the default; `max`, `4` for subdivision. Substeps name the finest trial, not total work. |

Exactly one terminal condition is reported: success, budget exhaustion,
non-finite failure, or step-size failure. A non-finite initial condition stops
before any attempts; a finite, representable single-node input completes with zero
attempts. Lossy integer-bound conversion stops with a step-size failure before
any attempts, including when conversion collapses distinct endpoints. The
requested endpoint retains its pre-promotion integer value in the diagnostics.
History buffers preserve the precision of both the initial state and dynamics;
finiteness checks inspect the stored initial state, not just the input value.
Terminal failures preserve the retained prefix instead of appending the failed
trial. Eager failures attach a message to `solver_info.warnings`, without raising
or emitting Python warnings.

Counts, flags, times, and the mask are native scalars/lists eagerly and JAX
arrays under tracing; they are no longer replaced by `None` inside JIT. Return
the mask and success flag alongside the trajectory from a compiled wrapper:

```python
def checked_adaptive_simulation(params):
    result = hp.solvers.simulate(
        model=f, ts=ts, solver=adaptive_solver, initial_state=y0, params=params,
    )
    diagnostics = result.solver_info.diagnostics
    return result.latent_state, diagnostics["valid_mask"], diagnostics["success"]
```

Inside JIT, use the fixed-shape mask for reductions rather than dynamically
slicing at `n_nodes`. The mask indicates **unpadded entries**, not initial-state
validity or numerical accuracy. Dynamic warning text is eager-only. Failure-path
gradients are not guaranteed; the validated gradient contract is the successful
solution's frozen-mesh sensitivity. See
[adaptive termination semantics](../theory/solvers.md#adaptive-termination).

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

Its `PredictorCorrector` configuration must use `weight_evaluation="direct"`;
requesting stable evaluation raises `NotImplementedError` pending separate
validation of the stochastic drift/noise kernels.
