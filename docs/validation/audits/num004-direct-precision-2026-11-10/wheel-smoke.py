"""Exercise the installed wheel in an isolated interpreter, not the checkout."""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

import hpfracc as hp
import hpfracc.solvers.predictor_corrector as pc

assert "jax" not in sys.modules
assert Path(hp.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert hashlib.sha256(Path(pc.__file__).read_bytes()).hexdigest() == (
    "506e1d8f9ab6bafb9ee9f9c79b88459c819eb4f479610f13a4ee84b9e6c789bf"
)

import jax  # noqa: E402 -- first verify that package imports remain lazy
import jax.numpy as jnp  # noqa: E402


def model(t, state, params, *, rng_key=None, inputs=None):
    return -params * state


with jax.enable_x64(False):
    times = jnp.arange(257, dtype=jnp.float32) / 256

    @jax.jit
    def trajectory(order, rate):
        return hp.solvers.simulate(
            model=model, ts=times, initial_state=jnp.ones(1, jnp.float32),
            params=rate,
            solver=hp.solvers.ImplicitPredictorCorrector(
                dt=1/256, order=order, newton_iterations=8,
                newton_rtol=1e-5, newton_atol=1e-7,
            ),
        ).latent_state

    order, rate = jnp.asarray(.5, jnp.float32), jnp.asarray(1., jnp.float32)
    values = trajectory(order, rate)
    gradient = jax.jit(jax.grad(lambda a: trajectory(a, rate)[-1, 0]))(order)
    assert values.dtype == gradient.dtype == jnp.float32
    assert np.isfinite(float(gradient)) and abs(float(gradient)) > 1e-4

# Independent host float64 panel quadrature, with exact newest-panel moments.
state = np.asarray(values, dtype=np.float64)
n = len(state)
a = .5
scale = (1/(n-1))**a / math.gamma(a)
left, right = np.empty(n-1), np.empty(n-1)
left[0], right[0] = scale/(a+1), scale/(a*(a+1))
nodes, weights = np.polynomial.legendre.leggauss(32)
fraction = (nodes+1)/2
lags = np.arange(1, n-1)[:, None] + fraction
integrand = lags**(a-1) * (weights/2)
left[1:], right[1:] = scale*(integrand @ fraction), scale*(integrand @ (1-fraction))
quadrature = np.zeros((n, n))
for step in range(1, n):
    quadrature[step, :step] += left[step-1::-1]
    quadrature[step, 1:step+1] += right[step-1::-1]
diagonal = np.diag(quadrature).copy()
np.fill_diagonal(quadrature, 0.)
rhs = -state
known = state[0] + quadrature @ rhs
residual = state-known-diagonal[:, None]*rhs
threshold = 1e-7 + 1e-5*np.maximum(np.abs(state), np.abs(known))
maximum = float(np.max(np.abs(residual[1:])/threshold[1:]))
assert maximum <= 1.
assert abs(state[-1, 0]-math.exp(1)*math.erfc(1)) < 1e-3
assert jax.default_backend() == "cpu"
print(json.dumps({
    "scope": ("Installed-wheel CPU import/JIT/order-gradient "
              "and selected residual smoke"),
    "module_path": str(Path(hp.__file__).resolve()),
    "kernel_sha256": hashlib.sha256(Path(pc.__file__).read_bytes()).hexdigest(),
    "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "python": sys.version, "jax": jax.__version__, "backend": jax.default_backend(),
    "trajectory_dtype": str(values.dtype), "gradient_dtype": str(gradient.dtype),
    "order_gradient": float(gradient), "independent_max_scaled_residual": maximum,
    "endpoint": float(state[-1, 0]),
}, indent=2))
