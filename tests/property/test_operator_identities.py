"""Property tests for fractional-operator algebraic identities.

These assert structural invariants that hold for every admissible input rather
than checking one hand-picked case: linearity of the discrete operators and the
defining Caputo-of-a-constant identity. Later Phase A workstreams add their own
property tests alongside these.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

jnp = pytest.importorskip("jax.numpy")

from hpfracc.ops import caputo, grunwald_letnikov

pytestmark = pytest.mark.property

_orders = st.floats(min_value=0.05, max_value=0.95)
_dts = st.floats(min_value=1e-3, max_value=1.0)
_n_steps = st.integers(min_value=2, max_value=24)
_signal_values = st.floats(
    min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
)


def _signal(draw: st.DrawFn, n: int) -> jnp.ndarray:
    return jnp.asarray(draw(st.lists(_signal_values, min_size=n, max_size=n)))


@st.composite
def _two_signals(draw: st.DrawFn) -> tuple:
    n = draw(_n_steps)
    x = _signal(draw, n)
    y = _signal(draw, n)
    a = draw(_signal_values)
    b = draw(_signal_values)
    return x, y, a, b


# Run in float64: linearity holds exactly in real arithmetic, but the GL
# ``dt**(-alpha)`` factor (large for small dt / alpha near 1) amplifies float32
# cancellation past a 1e-5 tolerance for adversarial scalars. float64 measures the
# algebraic property rather than single-precision roundoff. The float64 fixture is
# set up once around all hypothesis examples, hence the function_scoped_fixture
# health-check suppression.
_PROPERTY_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


@pytest.mark.parametrize("operator", [caputo, grunwald_letnikov])
@_PROPERTY_SETTINGS
@given(order=_orders, dt=_dts, payload=_two_signals())
def test_operator_is_linear(operator, enable_x64, order, dt, payload) -> None:
    x, y, a, b = payload
    combined = operator(a * x + b * y, dt=dt, order=order)
    separate = a * operator(x, dt=dt, order=order) + b * operator(
        y, dt=dt, order=order
    )
    assert jnp.allclose(combined, separate, rtol=1e-5, atol=1e-5)


@_PROPERTY_SETTINGS
@given(
    order=_orders,
    dt=_dts,
    n=_n_steps,
    constant=_signal_values,
)
def test_caputo_of_constant_is_zero(enable_x64, order, dt, n, constant) -> None:
    x = jnp.full((n,), constant)
    result = caputo(x, dt=dt, order=order)
    assert jnp.allclose(result, jnp.zeros_like(x), atol=1e-6)
