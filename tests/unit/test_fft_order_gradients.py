"""Dedicated order-gradient evidence for uniform FFT operator histories."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hpfracc.ops import caputo, grunwald_letnikov, riemann_liouville


@pytest.fixture
def precision_for_dtype(dtype):
    # Scope precision outside JIT and restore the caller's setting on exit.
    with jax.enable_x64(dtype == jnp.float64):
        yield


@pytest.mark.parametrize("operator", [caputo, grunwald_letnikov, riemann_liouville])
@pytest.mark.parametrize("order", [.15, .7, .9, [.2, .8]])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_fft_order_gradients_match_finite_difference_and_full_history(
    operator, order, dtype, precision_for_dtype,
):
    assert jax.config.read("jax_enable_x64") == (dtype == jnp.float64)
    time = jnp.linspace(0., 1., 33, dtype=dtype)
    values = jnp.stack((time**2, jnp.sin(time)), axis=-1)
    alpha = jnp.asarray(order, dtype=dtype)

    def objective(a, history="fft"):
        result = operator(values, dt=1/32, order=a, history=history)
        return jnp.mean(result**2)

    compiled_forward = jax.jit(
        lambda a, history: operator(values, dt=1/32, order=a, history=history),
        static_argnames=("history",),
    )
    compiled_objective = jax.jit(objective, static_argnames=("history",))
    for history in ("fft", "full"):
        result = operator(values, dt=1/32, order=alpha, history=history)
        assert result.dtype == dtype, (history, "eager forward", result.dtype)
        result = compiled_forward(alpha, history=history)
        assert result.dtype == dtype, (history, "compiled forward", result.dtype)
        assert objective(alpha, history=history).dtype == dtype
        assert compiled_objective(alpha, history=history).dtype == dtype

    eager = jax.grad(objective)(alpha)
    compiled = jax.jit(jax.grad(objective))(alpha)
    reference = jax.grad(lambda a: objective(a, history="full"))(alpha)
    for gradient in (eager, compiled, reference):
        assert gradient.dtype == dtype
    step = 2e-3 if dtype == jnp.float32 else 1e-5
    directions = jnp.eye(alpha.size, dtype=dtype).reshape((alpha.size, *alpha.shape))
    differences = jnp.stack([
        (compiled_objective(alpha + step * direction)
         - compiled_objective(alpha - step * direction)) / (2 * step)
        for direction in directions
    ]).reshape(alpha.shape)
    tolerance = 2e-3 if dtype == jnp.float32 else 2e-7
    np.testing.assert_allclose(compiled, differences, rtol=tolerance, atol=tolerance)
    comparison_tolerance = 2e-5 if dtype == jnp.float32 else 2e-12
    np.testing.assert_allclose(compiled, eager, rtol=comparison_tolerance,
                               atol=comparison_tolerance)
    np.testing.assert_allclose(compiled, reference, rtol=comparison_tolerance,
                               atol=comparison_tolerance)
    assert compiled.dtype == dtype
    assert np.all(np.isfinite(np.asarray(compiled)))
