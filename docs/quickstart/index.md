# Quickstart

The v0.1 quickstart will cover installation, a first fractional operator call,
and a first fixed-step Caputo FDE simulation.

## First Operator Call

```python
import jax.numpy as jnp
import hpfracc as hp

dt = 0.01
t = jnp.arange(101) * dt
x = t**2

dx = hp.ops.caputo(x, dt=dt, order=0.5)
```

Use `return_info=True` when a validation or research workflow needs method
metadata:

```python
result = hp.ops.caputo(x, dt=dt, order=0.5, return_info=True)
print(result.operator_info.to_dict())
```
