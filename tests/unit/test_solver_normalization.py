"""Independent uniform-solver normalization and float32 residual checks."""

import math

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

from hpfracc.ops._special import _LOG_GAMMA1P_COEFFICIENTS, _rgamma1p
from hpfracc.solvers import ImplicitPredictorCorrector, simulate
from hpfracc.solvers.predictor_corrector import _setup


def _cubic(t, state, params, *, rng_key=None, inputs=None):
    del t, rng_key, inputs
    return params * state**3


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_uniform_normalization_against_independent_gamma(dtype, compiled, enable_x64):
    orders = jnp.asarray([.01, .1, .3, .49, .5, .51, .7, .9, .99], dtype=dtype)

    def scales(alpha):
        solver = ImplicitPredictorCorrector(dt=.5, order=alpha)
        setup = _setup(
            model=_cubic, times=jnp.asarray([0., .5], dtype=dtype), solver=solver,
            y0=jnp.asarray(1., dtype=dtype), params=jnp.asarray(-2., dtype=dtype),
            rng_key=None, inputs=None,
        )
        return jnp.stack(setup[2:4])

    evaluate = jax.vmap(scales)
    actual = np.asarray((jax.jit(evaluate) if compiled else evaluate)(orders))
    expected = np.asarray([
        [.5**float(a) / math.gamma(float(a) + k) for k in (1, 2)]
        for a in np.asarray(orders)
    ])
    tolerance = 2.5e-7 if dtype == jnp.float32 else 3e-15
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=0)
    assert actual.dtype == np.dtype(dtype)


@pytest.mark.parametrize("compiled", [False, True])
def test_float32_implicit_corrector_meets_independent_residual(compiled):
    # Keep all inputs strongly float32, independently of the process x64 setting.
    alpha = jnp.asarray(.7, dtype=jnp.float32)
    solver = ImplicitPredictorCorrector(dt=.5, order=alpha, newton_iterations=8)

    def run(order):
        result = simulate(
            model=_cubic, ts=jnp.asarray([0., .5], dtype=jnp.float32),
            solver=ImplicitPredictorCorrector(dt=.5, order=order, newton_iterations=8),
            initial_state=jnp.asarray(1., dtype=jnp.float32),
            params=jnp.asarray(-2., dtype=jnp.float32),
        )
        return result.latent_state, result.solver_info.diagnostics["newton_converged"]

    states, converged = (jax.jit(run) if compiled else run)(alpha)
    a, y = float(alpha), float(states[-1])
    scale = .5**a / math.gamma(a + 2)
    known = 1 - 2 * a * scale
    residual = y - known + 2 * scale * y**3
    tolerance = solver.newton_atol + solver.newton_rtol * max(abs(y), abs(known))
    assert bool(converged)
    assert abs(residual) / tolerance <= 1
    assert states.dtype == jnp.float32


def test_normalization_series_constants_and_remainder():
    with mp.workdps(80):
        for k, coefficient in enumerate(_LOG_GAMMA1P_COEFFICIENTS, start=2):
            expected = (-1)**k * (mp.zeta(k) - 1) / k
            assert coefficient == float(expected)
        # Alternating terms decrease for a in [0,1]. First omitted terms bound
        # truncation only; floating-point evaluation needs separate checks.
        assert (mp.zeta(25) - 1) / 25 < mp.mpf("1.2e-9")
        assert (mp.zeta(57) - 1) / 57 < mp.mpf("1.3e-19")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_reciprocal_gamma_values_and_order_gradients(dtype, enable_x64):
    orders = jnp.asarray(np.linspace(0, 1, 257), dtype=dtype)
    values = jax.jit(_rgamma1p)(orders)
    gradients = jax.jit(jax.vmap(jax.grad(_rgamma1p)))(orders)
    with mp.workdps(70):
        references = [(1 / mp.gamma(1 + float(a)),
                       -mp.digamma(1 + float(a)) / mp.gamma(1 + float(a)))
                      for a in np.asarray(orders)]
    expected = np.asarray(references, dtype=np.float64)
    tolerance = 2.5e-7 if dtype == jnp.float32 else 1.5e-15
    np.testing.assert_allclose(values, expected[:, 0], rtol=tolerance, atol=0)
    np.testing.assert_allclose(gradients, expected[:, 1], rtol=tolerance,
                               atol=tolerance)
    assert values.dtype == dtype and gradients.dtype == dtype
