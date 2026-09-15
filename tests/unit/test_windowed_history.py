"""Windowed execution preserves the truncated sum and its gradient contract."""

from math import gamma

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hpfracc.ops import caputo, grunwald_letnikov, kernels, riemann_liouville
from hpfracc.ops import fractional as operators


def reference(values, alpha, dt, window, caputo_scheme):
    """Host recurrence and explicit sum; independent of JAX kernel helpers."""
    values = np.asarray(values)
    a = np.broadcast_to(np.asarray(alpha), values.shape[1:])
    signal = np.diff(values, axis=0) if caputo_scheme else values
    n = len(signal)
    w = max(1, min(int(window), n))
    weights = [np.ones_like(a)]
    for k in range(1, n):
        weights.append(((k + 1)**(1 - a) - k**(1 - a)) if caputo_scheme
                       else weights[-1] * (k - 1 - a) / k)
    out = np.zeros_like(signal)
    # Bound from the triangle inequality applied to discarded discrete terms.
    bound = np.zeros_like(signal.real)
    for t in range(n):
        for k in range(t + 1):
            term = weights[k] * signal[t - k]
            if k < w:
                out[t] += term
            else:
                bound[t] += np.abs(term)
    scale = dt**(-a)
    if caputo_scheme:
        scale = scale / np.vectorize(gamma)(2 - a)
        out = np.concatenate((np.zeros_like(values[:1]), out), axis=0)
        bound = np.concatenate((np.zeros_like(values[:1].real), bound), axis=0)
    return out * scale, bound * scale


@pytest.mark.parametrize("fn", [caputo, grunwald_letnikov, riemann_liouville])
@pytest.mark.parametrize("window", [0, 1, 5, 100])
@pytest.mark.parametrize("vector", [False, True])
def test_independent_truncated_sum_and_discarded_tail_bound(
    fn, window, vector, enable_x64,
):
    # Oscillatory, transient and complex components exercise cancellation and the
    # cutoff, rather than only smooth monotone signals with favorable errors.
    t = np.arange(19) * .05
    values = np.stack((np.sin(13 * t), (t >= .2) * np.exp(-5 * t), t**2), axis=-1)
    values = values.astype(np.complex128) * (1 + .2j)
    order = np.asarray([.25, .5, .8]) if vector else .5
    expected, bound = reference(values, order, .05, window, fn is caputo)
    actual = jax.jit(lambda x, a: fn(
        x, dt=.05, order=a, history="short_memory", window_steps=window,
    ))(jnp.asarray(values), jnp.asarray(order))
    full = fn(values, dt=.05, order=order)
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
    assert np.all(np.abs(np.asarray(full) - actual) <= bound + 3e-13)


@pytest.mark.parametrize("alpha", [.25, .5, .8])
def test_caputo_linear_signal_has_exact_window_cutoff(alpha, enable_x64):
    dt, window = .025, 7
    t = jnp.arange(41, dtype=jnp.float64) * dt
    actual = caputo(t, dt=dt, order=alpha,
                    history="short_memory", window_steps=window)
    # Constant increments telescope the retained L1 weights. After the window
    # fills, the result saturates rather than following the full Caputo derivative.
    expected = (jnp.minimum(jnp.arange(41), window) * dt)**(1 - alpha)
    expected /= gamma(2 - alpha)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(
        caputo(jnp.ones_like(t), dt=dt, order=alpha,
               history="short_memory", window_steps=window), jnp.zeros_like(t),
    )


@pytest.mark.parametrize("fn", [caputo, grunwald_letnikov])
def test_windowed_input_and_per_state_order_gradients(fn, enable_x64):
    values = jnp.arange(42., dtype=jnp.float64).reshape(7, 2, 3) / 42
    direction = jnp.cos(values * 4)
    alpha = jnp.asarray([.3, .5, .7])
    alpha_direction = jnp.asarray([.2, -.1, .3])

    def objective(x, a):
        out = fn(x, dt=.1, order=a, history="short_memory", window_steps=3)
        return jnp.mean(out**2)

    eager = jax.grad(objective, argnums=(0, 1))(values, alpha)
    compiled = jax.jit(jax.grad(objective, argnums=(0, 1)))(values, alpha)
    for actual, expected in zip(compiled, eager, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    epsilon = 1e-5
    fd_x = (objective(values + epsilon * direction, alpha)
            - objective(values - epsilon * direction, alpha)) / (2 * epsilon)
    fd_a = (objective(values, alpha + epsilon * alpha_direction)
            - objective(values, alpha - epsilon * alpha_direction)) / (2 * epsilon)
    np.testing.assert_allclose(jnp.sum(compiled[0] * direction), fd_x, rtol=1e-8)
    np.testing.assert_allclose(jnp.sum(compiled[1] * alpha_direction), fd_a, rtol=1e-8)


@pytest.mark.parametrize("fn,helper", [(caputo, "_l1_weights"),
                                     (grunwald_letnikov, "_gl_weights")])
def test_short_memory_builds_only_window_weights_without_dense_path(
    fn, helper, monkeypatch,
):
    original = getattr(operators, helper)
    sizes = []

    def weights(alpha, n):
        sizes.append(n)
        return original(alpha, n)

    def dense_forbidden(*args, **kwargs):
        raise AssertionError("short memory must not allocate a dense history matrix")

    monkeypatch.setattr(operators, helper, weights)
    monkeypatch.setattr(kernels, "_full_history_convolution", dense_forbidden)
    result = jax.jit(lambda x: fn(
        x, dt=.1, order=.5, history="short_memory", window_steps=8,
    ))(jnp.ones((1024, 2)))
    assert result.shape == (1024, 2) and sizes == [8]


def test_single_sample_gl_and_kernel_window_one():
    value = jnp.asarray([[1., 2.]])
    np.testing.assert_allclose(
        grunwald_letnikov(value, dt=.25, order=.5,
                         history="short_memory", window_steps=1), 2 * value,
    )
    np.testing.assert_array_equal(
        kernels._short_memory_convolution(value, jnp.ones(1), window_steps=1), value,
    )


def test_windowed_scan_supports_outer_batching_without_native_convolution(
    monkeypatch, enable_x64,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("native convolution must not be used by short memory")

    monkeypatch.setattr(jnp, "convolve", forbidden)
    values = jnp.sin(jnp.arange(2 * 31 * 3, dtype=jnp.float64)).reshape(2, 31, 3)
    weights = jnp.asarray([[1., .5, .25], [.3, -.2, .1], [.1, .2, -.1]])

    def apply(x, w):
        return kernels._short_memory_convolution(x, w, window_steps=3)

    out = jax.jit(jax.vmap(apply, in_axes=(0, None)))(values, weights)
    padded = jnp.pad(weights, ((0, 28), (0, 0)))
    expected = jax.vmap(kernels._full_history_convolution, in_axes=(0, None))(
        values, padded,
    )
    np.testing.assert_allclose(out, expected, rtol=2e-13, atol=2e-13)
    # The loop carry must promote real values when weights have higher precision.
    mixed = apply(values[0].astype(jnp.float32), weights)
    assert mixed.dtype == jnp.float64
