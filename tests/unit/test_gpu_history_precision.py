"""Independent float32 history checks for reduced-precision device contractions."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hpfracc.ops import grunwald_letnikov, riemann_liouville
from hpfracc.ops.kernels import _least_squares_solve


@pytest.mark.parametrize("operator", [grunwald_letnikov, riemann_liouville])
@pytest.mark.parametrize("compiled", [False, True])
def test_full_history_multicomponent_float32_matches_discrete_reference(
    operator, compiled,
):
    dt, alpha, n = 0.02, 0.7, 101
    t = jnp.linspace(0, 1, n, dtype=jnp.float32)
    values = jnp.stack((t, t**2, t**3), axis=-1)
    actual_inputs = np.asarray(values, dtype=np.float64)
    weights = np.empty(n, dtype=np.float64)
    weights[0] = 1.0
    for k in range(1, n):
        weights[k] = weights[k - 1] * (k - 1 - alpha) / k
    expected = np.stack(
        [np.convolve(actual_inputs[:, i], weights)[:n] for i in range(3)], axis=-1,
    ) / math.pow(dt, alpha)
    fn = lambda x: operator(x, dt=dt, order=alpha, history="full")  # noqa: E731
    actual = (jax.jit(fn) if compiled else fn)(values)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0, atol=1e-5)


def test_float32_fit_normal_equations_preserve_input_precision():
    rng = np.random.default_rng(42)
    host_a = rng.normal(size=(128, 8)).astype(np.float32)
    host_b = rng.normal(size=128).astype(np.float32)
    a64, b64 = host_a.astype(np.float64), host_b.astype(np.float64)
    expected = np.linalg.solve(a64.T @ a64 + 1e-8 * np.eye(8), a64.T @ b64)
    actual = _least_squares_solve(jnp.asarray(host_a), jnp.asarray(host_b))
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0, atol=2e-6)
