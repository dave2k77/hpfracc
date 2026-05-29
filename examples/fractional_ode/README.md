# Fractional ODE Examples

This directory contains minimal examples for the v0.1 fractional-ODE workflow:
operator evaluation, fixed-step Caputo FDE simulation, differentiable training,
and scalar-grid probabilistic calibration.

Run commands from the repository root with the project `uv` environment.

## Caputo Operator

```bash
uv run python examples/fractional_ode/caputo_operator.py
```

Expected high-level output:

- final Caputo derivative value for `x(t) = t^2` at `t = 1`;
- operator metadata showing the `caputo` family, `l1_full_history` method,
  fractional order, time step, history policy, diagnostics, and warnings.

Representative output from the Phase 6 smoke test:

```text
1.5040464
{'family': 'caputo', 'method': 'l1_full_history', 'fractional_order': 0.5, 'dt': 0.01, 'n_steps': 101, 'history': 'full', 'diagnostics': {}, 'warnings': []}
```

## Caputo Predictor-Corrector Solver

```bash
uv run python examples/fractional_ode/caputo_solver.py
```

Expected high-level output:

- final latent state for a scalar linear fractional differential equation;
- solver metadata showing the `caputo_pece_full_history` method, fractional
  order, step size, step count, diagnostics, and warnings.

Representative output from the Phase 6 smoke test:

```text
0.46728116
{'name': 'predictor_corrector', 'method': 'caputo_pece_full_history', 'fractional_order': 0.7, 'step_size': 0.01, 'n_steps': 100, 'diagnostics': {'history': 'full', 'grid': 'uniform'}, 'warnings': []}
```

## Differentiable Neural FODE Training

```bash
uv run python examples/fractional_ode/train_neural_fode.py
```

Expected high-level output:

- learned scalar rate parameter after a short gradient-descent run;
- final mean-squared trajectory loss against synthetic data.

This is an experimental differentiable-model example, not a validated system
identification workflow.

Representative output from the Phase 6 smoke test:

```text
{'rate': -0.7206791043281555, 'loss': 0.0004431932175066322}
```

## Scalar-Grid Probabilistic Calibration

```bash
uv run python examples/fractional_ode/probabilistic_calibration.py
```

Expected high-level output:

- best scalar rate selected from a fixed parameter grid;
- final posterior-predictive mean value.

This is an experimental scalar-grid calibration example. It is intended to show
result contracts and reproducibility mechanics, not broad Bayesian inference
coverage.

Representative output from the Phase 6 smoke test:

```text
{'best_rate': -0.800000011920929, 'posterior_mean_final': 0.608722448348999}
```
