"""Roundoff regressions against independent high-precision interval integrals."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

import hpfracc as hp
from hpfracc.ops._quadrature import _power_interval_moments
from hpfracc.solvers.predictor_corrector import _nonuniform_pece_weights


def _reference_weights(nodes, order):
    # Use the actual rounded input nodes, not a separately generated ideal mesh.
    with mp.workdps(70):
        ts = [mp.mpf(float(t)) for t in nodes]
        alpha = mp.mpf(float(order))
        predictor = []
        left = []
        right = []
        for start, end in zip(ts[:-1], ts[1:], strict=True):
            a, b, h = ts[-1] - start, ts[-1] - end, end - start
            zeroth = (a**alpha - b**alpha) / alpha
            first = (a ** (alpha + 1) - b ** (alpha + 1)) / (alpha + 1)
            predictor.append(zeroth / mp.gamma(alpha))
            left.append((first - b * zeroth) / (h * mp.gamma(alpha)))
            right.append((a * zeroth - first) / (h * mp.gamma(alpha)))
        corrector = [left[0]]
        corrector.extend(left[j] + right[j - 1] for j in range(1, len(left)))
        return (
            np.asarray(predictor, dtype=float),
            np.asarray(corrector, dtype=float),
            float(right[-1]),
        )


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
@pytest.mark.parametrize("order", (0.1, 0.4, 0.9))
def test_graded_weights_match_high_precision(enable_x64, dtype, order):
    ts = (jnp.arange(160, dtype=dtype) / 159) ** 4
    alpha = jnp.asarray(order, dtype=dtype)
    p, c, d = _nonuniform_pece_weights(159, ts, jnp.arange(160), alpha, dtype)
    p_ref, c_ref, d_ref = _reference_weights(np.asarray(ts), float(alpha))
    tolerance = 3e-5 if dtype == jnp.float32 else 3e-12
    # Tiny early weights must not vanish or become negative, even though their
    # lag endpoints may round to the same number in ordinary subtraction.
    assert np.all(np.asarray(p[:-1]) > 0)
    assert np.all(np.asarray(c[:-1]) > 0)
    np.testing.assert_allclose(p[:-1], p_ref, rtol=tolerance, atol=0)
    np.testing.assert_allclose(c[:-1], c_ref, rtol=tolerance, atol=0)
    assert float(d) == pytest.approx(d_ref, rel=tolerance)


@pytest.mark.parametrize("order", (0.1, 0.4, 0.9))
def test_float32_corrector_integrates_constant_and_affine_history(order):
    ts = (jnp.arange(160, dtype=jnp.float32) / 159) ** 4
    _, c, d = _nonuniform_pece_weights(159, ts, jnp.arange(160), order, ts.dtype)
    constant = jnp.sum(c[:-1]) + d
    affine = jnp.dot(c[:-1], ts[:-1]) + d * ts[-1]
    assert float(constant) == pytest.approx(1 / math.gamma(order + 1), rel=3e-6)
    assert float(affine) == pytest.approx(1 / math.gamma(order + 2), rel=3e-6)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
@pytest.mark.parametrize("order", (0.01, 0.4, 0.99))
def test_interval_branches_match_high_precision(enable_x64, dtype, order):
    # Cover both sides of the series switch, and a positive lower lag so small
    # that computing it as A*(1-h/A) would incorrectly round it to zero.
    grids = ([0, 0.24999, 1], [0, 0.25, 1], [0, 0.25001, 1], [-1, -1e-10, 0])
    for nodes in grids:
        ts = jnp.asarray(nodes, dtype=dtype)
        alpha = jnp.asarray(order, dtype=dtype)
        actual = _nonuniform_pece_weights(2, ts, jnp.arange(3), alpha, dtype)
        expected = _reference_weights(np.asarray(ts), float(alpha))
        tolerance = 3e-5 if dtype == jnp.float32 else 3e-12
        for values, reference in zip(
            (actual[0][:2], actual[1][:2], actual[2]), expected, strict=True
        ):
            np.testing.assert_allclose(values, reference, rtol=tolerance, atol=0)


def test_masked_interval_moments_have_finite_zero_gradient():
    def objective(order):
        moments = _power_interval_moments(
            jnp.asarray([0.0, -1.0, 0.0]),
            jnp.asarray([-1.0, -2.0, 0.0]),
            jnp.asarray([0.0, 1.0, 0.0]),
            order,
            # Completed adaptive scans can still evaluate a zero-width trial
            # interval with a true history mask before discarding the step.
            jnp.asarray([False, False, True]),
        )
        return sum(moment.sum() for moment in moments)

    assert float(jax.jit(objective)(0.4)) == 0.0
    assert float(jax.jit(jax.grad(objective))(0.4)) == 0.0


@pytest.mark.parametrize("order", (0.1, 0.4, 0.9))
def test_float32_caputo_retains_early_interval(order):
    ts = (jnp.arange(160, dtype=jnp.float32) / 159) ** 4
    # Only the first L1 increment is nonzero. This isolates the historical weight
    # that direct subtraction incorrectly rounds to zero on this graded mesh.
    signal = jnp.concatenate([jnp.zeros(1), jnp.ones(159)])
    result = hp.ops.caputo(signal, t=ts, order=order)
    with mp.workdps(70):
        alpha = mp.mpf(order)
        h = mp.mpf(float(ts[1]))
        expected = (1 - (1 - h) ** (1 - alpha)) / (h * mp.gamma(2 - alpha))
    assert float(result[-1]) == pytest.approx(float(expected), rel=3e-6)


def _constant_model(t, state, params, *, rng_key=None, inputs=None):
    del t, rng_key, inputs
    return jnp.ones_like(state) * params


def test_float32_graded_solver_matches_constant_forcing():
    order = 0.4
    ts = (jnp.arange(160, dtype=jnp.float32) / 159) ** 4
    result = hp.solvers.simulate(
        model=_constant_model,
        ts=ts,
        solver=hp.solvers.NonUniformPredictorCorrector(order=order),
        initial_state=jnp.asarray(1.0),
        params=jnp.asarray(1.0),
    )
    expected = 1 + ts**order / math.gamma(order + 1)
    np.testing.assert_allclose(result.latent_state, expected, rtol=3e-6, atol=0)


@pytest.mark.parametrize("target", ("solver", "operator"))
def test_graded_order_gradients_match_finite_difference(enable_x64, target):
    ts = (jnp.arange(32, dtype=jnp.float64) / 31) ** 4
    signal = jnp.concatenate([jnp.zeros(1), jnp.ones(31)])

    def objective(order):
        if target == "operator":
            return hp.ops.caputo(signal, t=ts, order=order)[-1]
        return hp.solvers.simulate(
            model=_constant_model,
            ts=ts,
            solver=hp.solvers.NonUniformPredictorCorrector(order=order),
            initial_state=jnp.asarray(1.0),
            params=jnp.asarray(1.0),
        ).latent_state[-1]

    order = 0.4
    actual = jax.jit(jax.grad(objective))(order)
    epsilon = 1e-5
    expected = (objective(order + epsilon) - objective(order - epsilon)) / (2 * epsilon)
    assert jnp.isfinite(actual)
    assert float(actual) == pytest.approx(float(expected), rel=1e-6, abs=1e-8)
    assert float(jax.jit(objective)(order)) == pytest.approx(
        float(objective(order)), rel=1e-12
    )
