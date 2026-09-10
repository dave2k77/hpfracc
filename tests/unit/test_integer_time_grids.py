"""Integer time differences must not wrap at signed dtype boundaries."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hpfracc as hp


def _constant(t, state, params, *, rng_key=None, inputs=None):
    del t, params, rng_key, inputs
    return jnp.ones_like(state)


@pytest.mark.parametrize("kind", ("explicit", "implicit", "nonuniform", "adaptive"))
def test_large_signed_integer_span_has_correct_constant_solution(kind):
    ts = jnp.asarray([-2**31, 0], dtype=jnp.int32)
    span = float(2**31)
    order = 0.7
    if kind == "explicit":
        solver = hp.solvers.PredictorCorrector(dt=span, order=order)
    elif kind == "implicit":
        solver = hp.solvers.ImplicitPredictorCorrector(dt=span, order=order)
    elif kind == "nonuniform":
        solver = hp.solvers.NonUniformPredictorCorrector(order=order)
    else:
        solver = hp.solvers.AdaptivePredictorCorrector(order=order, max_steps=16)

    def run(nodes):
        return hp.solvers.simulate(
            model=_constant, ts=nodes, solver=solver,
            initial_state=jnp.asarray(0.0), params=None,
        ).latent_state[-1]

    expected = span**order / math.gamma(order + 1)
    assert float(run(ts)) == pytest.approx(expected, rel=1e-5)
    assert float(jax.jit(run)(ts)) == pytest.approx(expected, rel=1e-5)


def test_large_signed_integer_span_caputo_value_and_metadata():
    ts = jnp.asarray([-2**31, 0], dtype=jnp.int32)
    span = float(2**31)
    values = jnp.asarray([0.0, span])
    order = 0.7
    result = hp.ops.caputo(values, t=ts, order=order, return_info=True)
    expected = span ** (1 - order) / math.gamma(2 - order)
    assert float(result.values[-1]) == pytest.approx(expected, rel=1e-5)
    assert result.operator_info.dt == span
    compiled = jax.jit(lambda nodes: hp.ops.caputo(values, t=nodes, order=order))(ts)
    np.testing.assert_allclose(compiled, result.values, rtol=1e-5)
