"""Validation for the non-uniform / graded-grid Caputo operator (WS-3).

These promote non-uniform grids from a deferred future extension to a validated
Caputo feature. The load-bearing properties are:

* the non-uniform L1 weights collapse **exactly** to the uniform ``b_k`` weights
  when the nodes are equispaced (consistency anchor), and
* on a mesh graded toward ``t = 0`` the operator matches the analytic Caputo
  derivative, and for a weakly-singular ``t^beta`` (``beta < 1``) the graded mesh
  beats a uniform mesh of the same node count.

Plus the contract guards (mutual exclusion of ``dt`` / ``t``, monotone / finite
nodes, full-history + scalar-order only, GL/RL rejection) and differentiability
with respect to the order on the non-uniform path.
"""

from __future__ import annotations

import json

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from hpfracc.ops import (
    caputo,
    caputo_power_law,
    grunwald_letnikov,
    riemann_liouville,
)


def _graded_nodes(n: int, rate: float = 3.0, t_end: float = 1.0):
    """Mesh graded toward ``t = 0``: ``t_j = t_end * (j / (n-1))**rate``."""

    j = jnp.arange(n, dtype=jnp.float64) / (n - 1)
    return t_end * j**rate


def test_nonuniform_reduces_to_uniform(enable_x64) -> None:
    n = 60
    t = jnp.linspace(0.0, 1.0, n)
    dt = float(t[1] - t[0])
    x = t**2
    uniform = caputo(x, dt=dt, order=0.5)
    via_nodes = caputo(x, t=t, order=0.5)
    assert jnp.allclose(uniform, via_nodes, atol=1e-12)


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
def test_graded_mesh_matches_analytic(alpha, enable_x64) -> None:
    t = _graded_nodes(160, rate=3.0)
    expected = caputo_power_law(t, power=2.0, order=alpha)
    actual = caputo(t**2, t=t, order=alpha)
    # Skip t=0 where the analytic power law has the trivial zero history.
    assert jnp.max(jnp.abs(actual[1:] - expected[1:])) < 1e-2


def test_graded_mesh_refines(enable_x64) -> None:
    alpha, beta, rate = 0.4, 0.6, 3.0
    errors = []
    for n in (40, 80, 160):
        t = _graded_nodes(n, rate=rate)
        expected = caputo_power_law(t, power=beta, order=alpha)
        actual = caputo(t**beta, t=t, order=alpha)
        errors.append(float(jnp.max(jnp.abs(actual[1:] - expected[1:]))))
    assert errors[0] > errors[1] > errors[2]


def test_graded_beats_uniform_for_singular_function(enable_x64) -> None:
    alpha, beta, n = 0.5, 0.7, 80
    t_graded = _graded_nodes(n, rate=3.0)
    graded_err = float(
        jnp.max(
            jnp.abs(
                caputo(t_graded**beta, t=t_graded, order=alpha)[1:]
                - caputo_power_law(t_graded, power=beta, order=alpha)[1:]
            )
        )
    )
    t_uniform = jnp.linspace(0.0, 1.0, n)
    dt = float(t_uniform[1] - t_uniform[0])
    uniform_err = float(
        jnp.max(
            jnp.abs(
                caputo(t_uniform**beta, dt=dt, order=alpha)[1:]
                - caputo_power_law(t_uniform, power=beta, order=alpha)[1:]
            )
        )
    )
    assert graded_err < uniform_err


# --- guards -----------------------------------------------------------------


def _signal(n: int = 24):
    t = jnp.linspace(0.0, 1.0, n)
    return t, t**2


def test_dt_and_t_are_mutually_exclusive() -> None:
    t, x = _signal()
    with pytest.raises(ValueError, match="exactly one"):
        caputo(x, dt=0.1, t=t, order=0.5)


def test_dt_or_t_required() -> None:
    _, x = _signal()
    with pytest.raises(ValueError, match="exactly one"):
        caputo(x, order=0.5)


def test_nodes_must_be_strictly_increasing() -> None:
    t, x = _signal()
    with pytest.raises(ValueError, match="strictly increasing"):
        caputo(x, t=t[::-1], order=0.5)


def test_nodes_length_must_match_samples() -> None:
    t, x = _signal(24)
    with pytest.raises(ValueError, match="one node per time sample"):
        caputo(x, t=jnp.linspace(0.0, 1.0, 25), order=0.5)


@pytest.mark.parametrize("history", ["fft", "short_memory", "soe"])
def test_nonuniform_rejects_non_full_history(history) -> None:
    t, x = _signal()
    with pytest.raises(NotImplementedError, match="history='full'"):
        caputo(x, t=t, order=0.5, history=history)


def test_nonuniform_rejects_vector_order() -> None:
    t = jnp.linspace(0.0, 1.0, 24)
    x = jnp.stack([t**2, t**3], axis=1)
    with pytest.raises(NotImplementedError, match="scalar"):
        caputo(x, t=t, order=jnp.array([0.3, 0.7]))


@pytest.mark.parametrize("operator", [grunwald_letnikov, riemann_liouville])
def test_gl_and_rl_reject_nonuniform_grid(operator) -> None:
    t, x = _signal()
    with pytest.raises(NotImplementedError, match="only for caputo"):
        operator(x, t=t, order=0.5)


# --- differentiability & info -----------------------------------------------


def test_nonuniform_order_gradient_matches_finite_difference(enable_x64) -> None:
    t = _graded_nodes(48, rate=2.0)

    def loss(order):
        return jnp.sum(caputo(t**2, t=t, order=order) ** 2)

    grad = jax.grad(loss)(0.5)
    assert jnp.isfinite(grad)
    eps = 1e-6
    fd = (loss(0.5 + eps) - loss(0.5 - eps)) / (2.0 * eps)
    assert jnp.allclose(grad, fd, rtol=1e-5, atol=1e-7)


def test_nonuniform_is_jit_compatible(enable_x64) -> None:
    t = _graded_nodes(40, rate=2.0)
    eager = caputo(t**2, t=t, order=0.5)
    jitted = jax.jit(lambda order: caputo(t**2, t=t, order=order))(0.5)
    assert jnp.allclose(eager, jitted, atol=1e-12)


def test_nonuniform_return_info_records_grid(enable_x64) -> None:
    t = _graded_nodes(20, rate=2.0)
    result = caputo(t**2, t=t, order=0.5, return_info=True)
    info = result.operator_info
    assert info.method == "l1_nonuniform"
    assert info.diagnostics["grid"] == "nonuniform"
    assert jnp.isfinite(info.dt)
    assert json.dumps(info.to_dict())  # stays JSON-serialisable


def test_uniform_return_info_records_grid(enable_x64) -> None:
    t = jnp.linspace(0.0, 1.0, 20)
    result = caputo(t**2, dt=float(t[1] - t[0]), order=0.5, return_info=True)
    assert result.operator_info.diagnostics["grid"] == "uniform"
