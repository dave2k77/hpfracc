# Phase 2: Caputo FDE Solver Plan

Status: Complete in the current working tree.

Phase 2 implements the first deterministic fractional differential equation
solver. The solver remains full-history and uniform-grid only.

## Target Equation

The initial solver targets Caputo initial value problems of the form:

```text
D_C^alpha y(t) = f(t, y(t), params)
y(0) = y0
0 < alpha < 1
```

## Public API

The public entry points are:

```python
solver = hp.solvers.PredictorCorrector(dt=0.01, order=0.8)

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

The solver returns `SimulationResult`.

## Validation Targets

The first validation case is the scalar linear FDE:

```text
D_C^alpha y(t) = lambda y(t)
y(0) = 1
```

The reference solution is:

```text
y(t) = E_alpha(lambda t^alpha)
```

where `E_alpha` is approximated by a truncated Mittag-Leffler series for test
and validation purposes.

## Required Tests

- Reject nonuniform time grids.
- Reject mismatched solver `dt` and `ts` spacing.
- Preserve arbitrary trailing state dimensions.
- Return trajectory shape `(time, ..., state_dim)` or `(time, ...)` depending on
  the initial state.
- Demonstrate timestep refinement on the scalar linear FDE.
- Demonstrate JIT/non-JIT consistency where practical.
- Demonstrate gradients with respect to initial state and scalar model
  parameter.

## Non-Goals

- Adaptive timesteps.
- Implicit methods.
- Stochastic FSDEs.
- Multi-device execution.
- Memory compression.
