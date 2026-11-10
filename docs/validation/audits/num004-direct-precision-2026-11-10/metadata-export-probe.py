"""Reproduce strong-order metadata export without changing the numerical core."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp

import hpfracc as hp
import hpfracc.solvers.base as base


def model(t, state, params, *, rng_key=None, inputs=None):
    return -params * state


rows = []
with jax.enable_x64(False):
    for cls in (hp.solvers.PredictorCorrector, hp.solvers.ImplicitPredictorCorrector):
        for strong in (False, True):
            order = jnp.asarray(.5, dtype=jnp.float32) if strong else .5
            result = hp.solvers.simulate(
                model=model, ts=jnp.asarray([0., .5, 1.], dtype=jnp.float32),
                solver=cls(dt=.5, order=order),  # type: ignore[arg-type]
                initial_state=jnp.ones(1, dtype=jnp.float32), params=.5,
            )
            assert result.solver_info is not None
            info = result.solver_info.to_dict()
            error = None
            try:
                json.dumps(info, allow_nan=False)
            except TypeError as exc:
                error = str(exc)
            assert (error is not None) is strong
            rows.append({"solver": cls.__name__, "strong_array_order": strong,
                         "json_export_succeeded": error is None,
                         "exported_order_type": type(info["fractional_order"]).__name__,
                         "error": error})
print(json.dumps({
    "scope": "Metadata export observation, not numerical validation",
    "backend": jax.default_backend(),
    "source_file": str(Path(base.__file__).resolve()),
    "source_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
    "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "rows": rows,
}, indent=2))
