"""Tests for FFT-accelerated history convolution equivalence.

FFT convolution is an opt-in, named alternative to the dense full-history
reference.  These tests enforce the contract that both strategies compute the
same causal history sum on uniform grids to within floating-point accumulation
error, and that the FFT path remains JIT- and autodiff-compatible.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from hpfracc.ops import caputo, grunwald_letnikov, riemann_liouville


@pytest.mark.parametrize("history", ["full", "fft"])
def test_caputo_history_methods_are_jit_compatible(history: str) -> None:
    dt = 0.05
    t = jnp.linspace(0.0, 1.0, 41)
    x = t**2

    fn = jax.jit(lambda v: caputo(v, dt=dt, order=0.5, history=history))
    result = fn(x)

    expected = caputo(x, dt=dt, order=0.5, history="full")
    assert jnp.allclose(result, expected, atol=1e-5)


@pytest.mark.parametrize("history", ["full", "fft"])
def test_grunwald_letnikov_history_methods_are_jit_compatible(history: str) -> None:
    dt = 0.02
    t = jnp.linspace(0.0, 1.0, 101)
    x = jnp.stack([t, t**2, t**3], axis=-1)

    fn = jax.jit(lambda v: grunwald_letnikov(v, dt=dt, order=0.7, history=history))
    result = fn(x)

    expected = grunwald_letnikov(x, dt=dt, order=0.7, history="full")
    assert jnp.allclose(result, expected, atol=1e-5)


def test_fft_caputo_matches_full_history_float64() -> None:
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        dt = 0.0125
        t = jnp.linspace(0.0, 1.0, 161)
        x = t**2

        full = caputo(x, dt=dt, order=0.5, history="full")
        fft = caputo(x, dt=dt, order=0.5, history="fft")
        assert jnp.max(jnp.abs(full - fft)) < 1e-10
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_fft_grunwald_letnikov_matches_full_history_float64() -> None:
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        dt = 0.01
        t = jnp.linspace(0.0, 1.0, 201)
        x = t**2

        full = grunwald_letnikov(x, dt=dt, order=0.7, history="full")
        fft = grunwald_letnikov(x, dt=dt, order=0.7, history="fft")
        assert jnp.max(jnp.abs(full - fft)) < 1e-10
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.7, 0.9])
def test_fft_caputo_matches_full_history_power_law(alpha: float) -> None:
    dt = 0.0125
    t = jnp.linspace(0.0, 1.0, 161)
    x = t**2

    full = caputo(x, dt=dt, order=alpha, history="full")
    fft = caputo(x, dt=dt, order=alpha, history="fft")
    assert jnp.max(jnp.abs(full - fft)) < 1e-5


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
def test_fft_grunwald_letnikov_matches_full_history_power_law(alpha: float) -> None:
    dt = 0.01
    t = jnp.linspace(0.0, 1.0, 201)
    x = t**2

    full = grunwald_letnikov(x, dt=dt, order=alpha, history="full")
    fft = grunwald_letnikov(x, dt=dt, order=alpha, history="fft")
    assert jnp.max(jnp.abs(full - fft)) < 1e-5


@pytest.mark.parametrize("alpha", [0.4, 0.6, 0.8])
def test_fft_riemann_liouville_matches_full_history_constant(alpha: float) -> None:
    dt = 0.02
    t = jnp.linspace(0.0, 1.0, 101)
    x = jnp.ones_like(t)

    full = riemann_liouville(x, dt=dt, order=alpha, history="full")
    fft = riemann_liouville(x, dt=dt, order=alpha, history="fft")
    interior = t >= 0.5
    assert jnp.max(jnp.abs((full - fft)[interior])) < 1e-5


def test_caputo_fft_gradient_matches_full_history() -> None:
    dt = 0.05
    t = jnp.linspace(0.0, 1.0, 41)

    def objective(x: jax.Array, *, history: str) -> jax.Array:
        return jnp.sum(caputo(x, dt=dt, order=0.6, history=history) ** 2)

    x0 = t**2
    g_full = jax.grad(lambda v: objective(v, history="full"))(x0)
    g_fft = jax.grad(lambda v: objective(v, history="fft"))(x0)

    assert jnp.allclose(g_full, g_fft, atol=2e-5)


def test_caputo_fft_gradient_matches_full_history_float64() -> None:
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        dt = 0.05
        t = jnp.linspace(0.0, 1.0, 41)

        def objective(x: jax.Array, *, history: str) -> jax.Array:
            return jnp.sum(caputo(x, dt=dt, order=0.6, history=history) ** 2)

        x0 = t**2
        g_full = jax.grad(lambda v: objective(v, history="full"))(x0)
        g_fft = jax.grad(lambda v: objective(v, history="fft"))(x0)

        assert jnp.max(jnp.abs(g_full - g_fft)) < 1e-10
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_grunwald_letnikov_fft_gradient_matches_full_history() -> None:
    dt = 0.05
    t = jnp.linspace(0.0, 1.0, 41)

    def objective(x: jax.Array, *, history: str) -> jax.Array:
        return jnp.sum(grunwald_letnikov(x, dt=dt, order=0.6, history=history) ** 2)

    x0 = t**2
    g_full = jax.grad(lambda v: objective(v, history="full"))(x0)
    g_fft = jax.grad(lambda v: objective(v, history="fft"))(x0)

    assert jnp.allclose(g_full, g_fft, atol=2e-5)


def test_grunwald_letnikov_fft_gradient_matches_full_history_float64() -> None:
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        dt = 0.05
        t = jnp.linspace(0.0, 1.0, 41)

        def objective(x: jax.Array, *, history: str) -> jax.Array:
            return jnp.sum(grunwald_letnikov(x, dt=dt, order=0.6, history=history) ** 2)

        x0 = t**2
        g_full = jax.grad(lambda v: objective(v, history="full"))(x0)
        g_fft = jax.grad(lambda v: objective(v, history="fft"))(x0)

        assert jnp.max(jnp.abs(g_full - g_fft)) < 1e-10
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_operator_info_records_history_method() -> None:
    dt = 0.1
    t = jnp.linspace(0.0, 1.0, 21)
    result = caputo(t**2, dt=dt, order=0.5, history="fft", return_info=True)

    assert result.operator_info.history.value == "fft"
    assert "fft" in result.operator_info.method
    assert result.operator_info.diagnostics.get("history") == "fft"


@pytest.mark.parametrize("n", [1, 2, 5, 17, 64, 129])
def test_fft_matches_full_history_for_odd_and_power_of_two_lengths(n: int) -> None:
    dt = 1.0 / float(n - 1) if n > 1 else 1.0
    t = jnp.linspace(0.0, 1.0, n)
    x = jnp.sin(2.0 * jnp.pi * t)

    full = caputo(x, dt=dt, order=0.5, history="full")
    fft = caputo(x, dt=dt, order=0.5, history="fft")
    assert jnp.allclose(full, fft, atol=1e-5)
