"""Check the installed NUM-005 wheel, not an editable checkout."""

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

import hpfracc as hp
import hpfracc.solvers.predictor_corrector as pc

assert "jax" not in sys.modules
assert Path(hp.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
expected = sys.argv[1]
assert re.fullmatch(r"[0-9a-f]{64}", expected)
kernel_hash = hashlib.sha256(Path(pc.__file__).read_bytes()).hexdigest()
assert kernel_hash == expected

import jax  # noqa: E402 -- verify lazy package import first
import jax.numpy as jnp  # noqa: E402


def model(t, state, params, *, rng_key=None, inputs=None):
    return -params * state


def independent_maximum(values, rate):
    # Host float64 panel quadrature; newest-panel moments are exact analytically.
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
    left[1:] = scale*(integrand @ fraction)
    right[1:] = scale*(integrand @ (1-fraction))
    quadrature = np.zeros((n, n))
    for step in range(1, n):
        quadrature[step, :step] += left[step-1::-1]
        quadrature[step, 1:step+1] += right[step-1::-1]
    diagonal = np.diag(quadrature).copy()
    np.fill_diagonal(quadrature, 0.)
    rhs = -rate * state
    known = state[0] + quadrature @ rhs
    residual = state-known-diagonal[:, None]*rhs
    threshold = 1e-7 + 1e-5*np.maximum(np.abs(state), np.abs(known))
    return float(np.max(np.abs(residual[1:])/threshold[1:]))


rows = []
with jax.enable_x64(False):
    times = jnp.arange(257, dtype=jnp.float32)/256

    def trajectory(order, rate, mode):
        return hp.solvers.simulate(
            model=model, ts=times, initial_state=jnp.ones(1, jnp.float32),
            params=rate,
            solver=hp.solvers.ImplicitPredictorCorrector(
                dt=1/256, order=order, newton_iterations=8,
                newton_rtol=1e-5, newton_atol=1e-7, weight_evaluation=mode,
            ),
        ).latent_state

    compiled = jax.jit(trajectory, static_argnames=("mode",))
    order = jnp.asarray(.5, jnp.float32)
    for rate_value, mode in ((1., "direct"), (30., "direct"), (30., "stable")):
        rate = jnp.asarray(rate_value, jnp.float32)
        values = compiled(order, rate, mode)
        assert values.dtype == jnp.float32
        assert np.all(np.isfinite(np.asarray(values)))
        maximum = independent_maximum(values, rate_value)
        assert maximum <= 1.
        rows.append({
            "rate": rate_value, "weight_evaluation": mode,
            "trajectory_dtype": str(values.dtype),
            "independent_max_scaled_residual": maximum,
        })
        if rate_value == 1.:
            endpoint = np.asarray(values)[-1, 0]
            assert abs(endpoint-math.exp(1)*math.erfc(1)) < 1e-3
    weak_rate = jnp.asarray(1., jnp.float32)

    def objective(a):
        return jnp.sum(compiled(a, weak_rate, "direct")[-1])

    value, gradient = jax.jit(jax.value_and_grad(objective))(order)
    assert value.dtype == gradient.dtype == jnp.float32
    assert np.isfinite(float(gradient)) and abs(float(gradient)) > 1e-4
assert jax.default_backend() == "cpu"
print(json.dumps({
    "scope": "Installed-wheel CPU lazy-import/JIT/gradient and selected residuals",
    "module_path": str(Path(hp.__file__).resolve()),
    "kernel_sha256": kernel_hash,
    "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "python": sys.version, "jax": jax.__version__,
    "backend": jax.default_backend(), "cases": rows,
    "objective_dtype": str(value.dtype), "gradient_dtype": str(gradient.dtype),
    "order_gradient": float(gradient),
}, indent=2))
