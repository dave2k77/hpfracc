"""Record a scoped stiff-case residual; no scientific acceptance assertion."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import hpfracc as hp
import hpfracc.solvers.predictor_corrector as pc
from benchmarks.numerical.implicit_validation import independent_residuals


def model(t, state, params, *, rng_key=None, inputs=None):
    return -params * state


with jax.enable_x64(False):
    n = 257
    times = jnp.linspace(0., 1., n, dtype=jnp.float32)
    rows = []
    for mode in ("direct", "stable"):
        solver = hp.solvers.ImplicitPredictorCorrector(
            # Strong array orders are supported; the config annotation is float.
            dt=1/(n-1), order=jnp.asarray(.5, jnp.float32),  # type: ignore[arg-type]
            newton_iterations=8, newton_rtol=1e-5, newton_atol=1e-7,
            weight_evaluation=mode,
        )
        result = hp.solvers.simulate(
            model=model, ts=times, solver=solver,
            initial_state=jnp.ones(1, dtype=jnp.float32),
            params=jnp.asarray(30., jnp.float32),
        )
        assert result.solver_info is not None
        assert result.latent_state.dtype == jnp.float32
        residuals = independent_residuals(
            "half_decay", np.asarray(times),
            np.asarray(result.latent_state, dtype=np.float64),
            30., .5, np.ones(1), 1e-5, 1e-7,
        )
        rows.append({"weight_evaluation": mode,
                     "independent_max_scaled_residual": float(np.max(residuals)),
                     "newton_converged": bool(
                         result.solver_info.diagnostics["newton_converged"]),
                     "effective_trajectory_dtype": str(result.latent_state.dtype)})
print(json.dumps({
    "scope": ("Exploratory stiff case, outside frozen scientific matrix; "
              "not continuous accuracy validation"),
    "backend": jax.default_backend(), "order": .5, "strength": 30.,
    "n_samples": n, "newton_iterations": 8, "rtol": 1e-5, "atol": 1e-7,
    "reference_state_dtype": "float64",
    "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "source_file": str(Path(pc.__file__).resolve()),
    "source_sha256": hashlib.sha256(Path(pc.__file__).read_bytes()).hexdigest(),
    "rows": rows,
}, indent=2))
