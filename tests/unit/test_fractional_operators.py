from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from hpfracc.ops import (
    OperatorFamily,
    caputo,
    caputo_power_law,
    grunwald_letnikov,
    riemann_liouville,
)


def test_caputo_constant_is_zero() -> None:
    x = jnp.ones((8,))
    actual = caputo(x, dt=0.1, order=0.5)
    assert jnp.allclose(actual, jnp.zeros_like(x))


def test_caputo_linear_matches_l1_closed_form() -> None:
    dt = 0.1
    alpha = 0.5
    ts = jnp.arange(8) * dt
    actual = caputo(ts, dt=dt, order=alpha)
    expected = ts ** (1.0 - alpha) / math.gamma(2.0 - alpha)
    assert jnp.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_caputo_power_law_reference_matches_linear_case() -> None:
    dt = 0.1
    alpha = 0.5
    ts = jnp.arange(8) * dt
    actual = caputo_power_law(ts, power=1.0, order=alpha)
    expected = ts ** (1.0 - alpha) / math.gamma(2.0 - alpha)
    assert jnp.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_caputo_quadratic_refines_against_analytic_reference() -> None:
    alpha = 0.4
    coarse_t = jnp.linspace(0.0, 1.0, 21)
    fine_t = jnp.linspace(0.0, 1.0, 81)

    coarse = caputo(coarse_t**2, dt=float(coarse_t[1] - coarse_t[0]), order=alpha)
    fine = caputo(fine_t**2, dt=float(fine_t[1] - fine_t[0]), order=alpha)

    coarse_ref = caputo_power_law(coarse_t, power=2.0, order=alpha)
    fine_ref = caputo_power_law(fine_t, power=2.0, order=alpha)

    coarse_error = jnp.max(jnp.abs(coarse[1:] - coarse_ref[1:]))
    fine_error = jnp.max(jnp.abs(fine[1:] - fine_ref[1:]))
    assert fine_error < coarse_error


def test_operators_preserve_trailing_state_shape() -> None:
    x = jnp.arange(24, dtype=float).reshape(6, 2, 2)
    actual = caputo(x, dt=0.2, order=0.4)
    assert actual.shape == x.shape


def test_riemann_liouville_uses_gl_baseline_discretisation() -> None:
    x = jnp.linspace(0.0, 1.0, 8)
    gl = grunwald_letnikov(x, dt=0.1, order=0.5)
    rl = riemann_liouville(x, dt=0.1, order=0.5)
    assert jnp.allclose(rl, gl)


def test_return_info_provides_operator_metadata() -> None:
    result = caputo(jnp.linspace(0.0, 1.0, 5), dt=0.25, order=0.5, return_info=True)
    assert result.values.shape == (5,)
    assert result.operator_info.family is OperatorFamily.CAPUTO
    assert result.operator_info.method == "l1_full_history"
    assert result.operator_info.to_dict()["family"] == "caputo"


def test_riemann_liouville_return_info_records_baseline_warning() -> None:
    result = riemann_liouville(
        jnp.linspace(0.0, 1.0, 5),
        dt=0.25,
        order=0.5,
        return_info=True,
    )
    assert result.operator_info.family is OperatorFamily.RIEMANN_LIOUVILLE
    assert result.operator_info.warnings


def test_grunwald_letnikov_is_jittable() -> None:
    fn = jax.jit(lambda x: grunwald_letnikov(x, dt=0.1, order=0.5))
    x = jnp.linspace(0.0, 1.0, 8)
    actual = fn(x)
    assert actual.shape == x.shape


def test_caputo_supports_input_gradients() -> None:
    def objective(x: jax.Array) -> jax.Array:
        return jnp.sum(caputo(x, dt=0.1, order=0.5) ** 2)

    x = jnp.linspace(0.0, 1.0, 8)
    grad = jax.grad(objective)(x)
    assert grad.shape == x.shape
    assert jnp.all(jnp.isfinite(grad))


@given(
    a=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20, deadline=None)
def test_caputo_is_linear_in_input_samples(a: float, b: float) -> None:
    x = jnp.linspace(0.0, 1.0, 16)
    y = jnp.linspace(1.0, -0.5, 16) ** 2
    lhs = caputo(a * x + b * y, dt=0.1, order=0.5)
    rhs = a * caputo(x, dt=0.1, order=0.5) + b * caputo(y, dt=0.1, order=0.5)
    assert jnp.allclose(lhs, rhs, rtol=1e-5, atol=1e-5)


@given(
    a=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20, deadline=None)
def test_grunwald_letnikov_is_linear_in_input_samples(a: float, b: float) -> None:
    x = jnp.linspace(0.0, 1.0, 16)
    y = jnp.linspace(1.0, -0.5, 16) ** 2
    lhs = grunwald_letnikov(a * x + b * y, dt=0.1, order=0.5)
    rhs = a * grunwald_letnikov(x, dt=0.1, order=0.5) + b * grunwald_letnikov(
        y,
        dt=0.1,
        order=0.5,
    )
    assert jnp.allclose(lhs, rhs, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dt", [0.0, -0.1])
def test_operators_reject_nonpositive_dt(dt: float) -> None:
    with pytest.raises(ValueError, match="positive uniform timestep"):
        caputo(jnp.ones((4,)), dt=dt, order=0.5)
