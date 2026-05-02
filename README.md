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

## Current Phase 1 Surface

The current pre-alpha implementation includes baseline full-history fractional
operators on uniform grids:

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

Build the documentation:

```bash
mkdocs build --strict
```

## Research Use

HPFRACC is research software. It is not clinical software, is not validated for
diagnosis or treatment, and should not be used as a substitute for empirical
biomedical assessment.

See [RESEARCH_USAGE.md](RESEARCH_USAGE.md) for the project research-use policy.
