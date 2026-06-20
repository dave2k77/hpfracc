"""Validation for vector / per-state fractional orders (WS-4).

These promote per-state orders from a future extension to a validated operator
feature. The load-bearing properties are:

* a uniform vector order reproduces the scalar result exactly, and
* a per-component vector order decouples -- component ``j`` is exactly the scalar
  operator with order ``alpha[j]`` applied to that component alone.

Plus the usual contract checks: open-interval validation, history-method
equivalence (fft / short_memory match full; soe rejects vector orders), and
differentiability with respect to the order *vector*.
"""

from __future__ import annotations

import json

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from hpfracc.ops import caputo, grunwald_letnikov, riemann_liouville

_OPERATORS = {
    "caputo": caputo,
    "grunwald_letnikov": grunwald_letnikov,
    "riemann_liouville": riemann_liouville,
}


def _signal(n: int = 48, n_features: int = 3):
    t = jnp.linspace(0.0, 1.0, n)
    dt = float(t[1] - t[0])
    columns = [jnp.sin(2.0 * t), t**2, jnp.cos(t)]
    x = jnp.stack(columns[:n_features], axis=1)
    return x, dt


@pytest.mark.parametrize("target", list(_OPERATORS))
def test_uniform_vector_order_matches_scalar(target, enable_x64) -> None:
    operator = _OPERATORS[target]
    x, dt = _signal()
    alpha = 0.6
    scalar = operator(x, dt=dt, order=alpha)
    vector = operator(x, dt=dt, order=jnp.full((x.shape[1],), alpha))
    assert jnp.allclose(scalar, vector, atol=1e-12)


@pytest.mark.parametrize("target", list(_OPERATORS))
def test_vector_order_decouples_per_component(target, enable_x64) -> None:
    operator = _OPERATORS[target]
    x, dt = _signal()
    alphas = jnp.array([0.3, 0.55, 0.8])
    vector = operator(x, dt=dt, order=alphas)
    for j in range(x.shape[1]):
        column = operator(x[:, j : j + 1], dt=dt, order=float(alphas[j]))
        assert jnp.allclose(vector[:, j : j + 1], column, atol=1e-12)


@pytest.mark.parametrize("target", list(_OPERATORS))
@pytest.mark.parametrize("bad", [[0.5, 1.0], [0.0, 0.5], [0.5, -0.1], [0.5, 1.2]])
def test_vector_order_rejects_values_outside_open_interval(target, bad) -> None:
    operator = _OPERATORS[target]
    x, dt = _signal(n_features=2)
    with pytest.raises(ValueError, match="open interval"):
        operator(x, dt=dt, order=jnp.array(bad))


@pytest.mark.parametrize("target", ["caputo", "grunwald_letnikov"])
@pytest.mark.parametrize("history", ["fft", "short_memory"])
def test_vector_order_history_matches_full(target, history, enable_x64) -> None:
    operator = _OPERATORS[target]
    x, dt = _signal()
    alphas = jnp.array([0.3, 0.55, 0.8])
    full = operator(x, dt=dt, order=alphas, history="full")
    other = operator(
        x, dt=dt, order=alphas, history=history, window_steps=x.shape[0]
    )
    assert jnp.allclose(full, other, atol=1e-8)


@pytest.mark.parametrize("target", ["caputo", "grunwald_letnikov"])
def test_vector_order_soe_raises(target) -> None:
    operator = _OPERATORS[target]
    x, dt = _signal()
    with pytest.raises(NotImplementedError, match="per-state"):
        operator(x, dt=dt, order=jnp.array([0.3, 0.55, 0.8]), history="soe")


@pytest.mark.parametrize("target", list(_OPERATORS))
def test_vector_order_gradient_matches_finite_difference(target, enable_x64) -> None:
    operator = _OPERATORS[target]
    x, dt = _signal()
    alphas = jnp.array([0.3, 0.55, 0.8])

    def loss(order):
        return jnp.sum(operator(x, dt=dt, order=order) ** 2)

    grad = jax.grad(loss)(alphas)
    assert jnp.all(jnp.isfinite(grad))

    eps = 1e-6
    fd = jnp.array(
        [
            (loss(alphas.at[j].add(eps)) - loss(alphas.at[j].add(-eps))) / (2.0 * eps)
            for j in range(alphas.shape[0])
        ]
    )
    assert jnp.allclose(grad, fd, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("target", list(_OPERATORS))
def test_vector_order_is_jit_compatible(target, enable_x64) -> None:
    operator = _OPERATORS[target]
    x, dt = _signal()
    alphas = jnp.array([0.3, 0.55, 0.8])
    eager = operator(x, dt=dt, order=alphas)
    jitted = jax.jit(lambda order: operator(x, dt=dt, order=order))(alphas)
    assert jnp.allclose(eager, jitted, atol=1e-12)


def test_vector_order_return_info_records_tuple(enable_x64) -> None:
    x, dt = _signal(n_features=2)
    result = caputo(x, dt=dt, order=jnp.array([0.3, 0.8]), return_info=True)
    order_field = result.operator_info.fractional_order
    assert isinstance(order_field, tuple)
    assert len(order_field) == 2
    payload = result.operator_info.to_dict()
    assert payload["fractional_order"] == list(order_field)
    # The export must stay JSON-serialisable.
    assert json.loads(json.dumps(payload["fractional_order"])) == list(order_field)


def test_scalar_order_return_info_stays_float(enable_x64) -> None:
    x, dt = _signal(n_features=2)
    result = caputo(x, dt=dt, order=0.5, return_info=True)
    assert isinstance(result.operator_info.fractional_order, float)
    assert result.operator_info.fractional_order == 0.5
