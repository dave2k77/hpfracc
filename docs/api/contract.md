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
