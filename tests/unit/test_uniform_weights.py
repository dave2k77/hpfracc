"""Stable uniform PECE weights: independent precision and solver contracts."""

import json
from math import gamma

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from scipy.special import erfcx

import hpfracc as hp
from benchmarks.numerical.implicit_validation import independent_residuals
from hpfracc.solvers._weights import _stable_pece_weights
from hpfracc.solvers.predictor_corrector import _pece_weights


def reference_samples(step, positions, alpha):
    """70-digit interval integrals, not the solver's power-difference formulas."""
    with mp.workdps(70):
        a = mp.mpf(float(alpha))

        def moments(lag):
            upper, lower = mp.mpf(lag), mp.mpf(lag - 1)
            zero = (upper**a - lower**a) / a
            first = (upper**(a + 1) - lower**(a + 1)) / (a + 1)
            return zero, first - lower * zero, upper * zero - first

        predictor, corrector = [], []
        for index in positions:
            if index >= step:
                predictor.append(0.)
                corrector.append(0.)
                continue
            lag = step - index
            zero, left, _ = moments(lag)
            right = moments(lag + 1)[2] if index > 0 else 0
            predictor.append(float(a * zero))
            corrector.append(float(a * (a + 1) * (left + right)))
        return np.asarray(predictor), np.asarray(corrector)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("alpha", [.01, .1, .5, .9, .99])
@pytest.mark.parametrize("step", [1, 3, 256, 4096])
def test_stable_weights_against_high_precision(dtype, alpha, step, enable_x64):
    a = jnp.asarray(alpha, dtype=dtype)
    indices = jnp.arange(step + 3)
    p, c = jax.jit(lambda a: _stable_pece_weights(
        jnp.asarray(step), indices, a, dtype,
    ))(a)
    positions = sorted({0, min(1, step - 1), step // 2, max(0, step - 4),
                        step - 1, step, step + 1, step + 2})
    expected = reference_samples(step, positions, a)
    tolerance = 5e-6 if dtype == jnp.float32 else 3e-12
    for actual, reference in zip((p, c), expected, strict=True):
        assert actual.dtype == dtype
        np.testing.assert_allclose(np.asarray(actual)[positions], reference,
                                   rtol=tolerance, atol=0)
        assert np.all(np.asarray(actual[:step]) > 0)
        np.testing.assert_array_equal(actual[step:], 0)


@pytest.mark.parametrize("alpha", [.2, .5, .8])
def test_direct_formula_anchor_moments_and_weight_gradients(alpha, enable_x64):
    step = 16
    indices = jnp.arange(step + 1)
    a = jnp.asarray(alpha)
    p, c = _stable_pece_weights(jnp.asarray(step), indices, a, jnp.float64)
    direct = _pece_weights(jnp.asarray(step), indices, a, jnp.float64)
    for found, expected in zip((p, c), direct, strict=True):
        np.testing.assert_allclose(found, expected, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(jnp.sum(p), step**alpha, rtol=2e-13)
    np.testing.assert_allclose(jnp.sum(c) + 1, (alpha + 1) * step**alpha, rtol=2e-13)
    np.testing.assert_allclose(c @ indices + step, step**(alpha + 1), rtol=2e-13)

    def objective(order):
        pred, corr = _stable_pece_weights(jnp.asarray(step), indices,
                                          order, jnp.float64)
        return pred @ jnp.sin(indices) + .7 * (corr @ jnp.cos(indices))

    gradient = jax.jit(jax.grad(objective))(a)
    finite_difference = (objective(a + 1e-5) - objective(a - 1e-5)) / 2e-5
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-7, atol=1e-8)


def nonlinear_model(t, state, rate, *, rng_key=None, inputs=None):
    del rng_key, inputs
    return -rate * state**3 + .1 * state[::-1] + jnp.stack((t*t, jnp.sin(t)))


@pytest.mark.parametrize("solver_class", [hp.solvers.PredictorCorrector,
                                         hp.solvers.ImplicitPredictorCorrector])
@pytest.mark.parametrize("alpha", [.2, .5, .8])
def test_solver_compatibility_jit_and_gradients(solver_class, alpha, enable_x64):
    times = jnp.linspace(0., 1., 33)
    y0, rate, order = jnp.asarray([1., .5]), jnp.asarray(.3), jnp.asarray(alpha)

    def run(initial, parameter, a, evaluation="stable"):
        return hp.solvers.simulate(
            model=nonlinear_model, ts=times, initial_state=initial, params=parameter,
            solver=solver_class(dt=1/32, order=a, weight_evaluation=evaluation),
        ).latent_state

    values = run(y0, rate, order)
    compiled = jax.jit(run)(y0, rate, order)
    np.testing.assert_allclose(compiled, values, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(values, run(y0, rate, order, "direct"),
                               rtol=1e-10, atol=1e-11)

    def objective(initial, parameter, a):
        return jnp.sum(run(initial, parameter, a)[-1])

    gradients = jax.jit(jax.grad(objective, argnums=(0, 1, 2)))(y0, rate, order)
    direction, eps = jnp.asarray([.3, -.1]), 1e-5
    fd_y = (objective(y0 + eps * direction, rate, order)
            - objective(y0 - eps * direction, rate, order)) / (2 * eps)
    fd_rate = (objective(y0, rate + eps, order)
               - objective(y0, rate - eps, order)) / (2 * eps)
    fd_order = (objective(y0, rate, order + eps)
                - objective(y0, rate, order - eps)) / (2 * eps)
    np.testing.assert_allclose(gradients[0] @ direction, fd_y, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(gradients[1], fd_rate, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(gradients[2], fd_order, rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("solver_class", [hp.solvers.PredictorCorrector,
                                         hp.solvers.ImplicitPredictorCorrector])
def test_default_direct_path_and_metadata(solver_class):
    times = jnp.arange(9, dtype=jnp.float32) / 8

    def model(t, state, params, *, rng_key=None, inputs=None):
        return params * state

    def run(solver):
        return hp.solvers.simulate(model=model, ts=times, solver=solver,
                                   initial_state=jnp.ones((2, 3)), params=-.5)

    default = run(solver_class(dt=1/8, order=.5))
    explicit = run(solver_class(dt=1/8, order=.5, weight_evaluation="direct"))
    np.testing.assert_array_equal(default.latent_state, explicit.latent_state)
    stable = run(solver_class(dt=1/8, order=.5, weight_evaluation="stable"))
    assert stable.latent_state.shape == (9, 2, 3)
    payload = json.loads(json.dumps(stable.solver_info.to_dict(), allow_nan=False))
    assert payload["diagnostics"]["weight_evaluation"] == "stable"
    assert default.solver_info.diagnostics["weight_evaluation"] == "direct"


def test_explicit_matches_nonuniform_uniform_mesh_and_refines(enable_x64):
    errors = []

    def model(t, state, params, *, rng_key=None, inputs=None):
        return 2 * t**1.5 / gamma(2.5) - (state - t*t)

    for n in (33, 129):
        times = jnp.linspace(0., 1., n)
        def run(solver, times=times):
            return hp.solvers.simulate(model=model, ts=times, solver=solver,
                                       initial_state=jnp.asarray(0.), params=None)
        stable = run(hp.solvers.PredictorCorrector(
            dt=1/(n-1), order=.5, weight_evaluation="stable",
        ))
        nonuniform = run(hp.solvers.NonUniformPredictorCorrector(order=.5))
        np.testing.assert_allclose(stable.latent_state, nonuniform.latent_state,
                                   rtol=2e-12, atol=2e-12)
        errors.append(float(jnp.max(jnp.abs(stable.latent_state - times**2))))
    # Refinement concerns PECE discretization, not roundoff. On these grids the
    # unchanged one-pass method has order near 1+alpha and error about 3.3e-4.
    observed = np.log(errors[0] / errors[1]) / np.log(4)
    assert 1.4 < observed < 1.8 and errors[1] < 5e-4


def test_float32_implicit_residual_improvement():
    n = 257
    times = jnp.linspace(0., 1., n, dtype=jnp.float32)
    alpha = jnp.asarray(.5, dtype=jnp.float32)
    rate = jnp.asarray(30., dtype=jnp.float32)

    def model(t, state, params, *, rng_key=None, inputs=None):
        return -params * state

    independent = []
    for evaluation in ("direct", "stable"):
        solver = hp.solvers.ImplicitPredictorCorrector(
            dt=1/(n-1), order=alpha, newton_iterations=8,
            newton_rtol=1e-5, newton_atol=1e-7, weight_evaluation=evaluation,
        )
        result = hp.solvers.simulate(model=model, ts=times, solver=solver,
                                     initial_state=jnp.ones(1), params=rate)
        assert result.solver_info.diagnostics["newton_converged"]
        residuals = independent_residuals(
            "half_decay", np.asarray(times),
            np.asarray(result.latent_state, dtype=float),
            30., .5, np.ones(1), 1e-5, 1e-7,
        )
        independent.append(np.max(residuals))
    assert independent[1] < .05 * independent[0]


def test_stable_evaluation_does_not_repair_coarse_grid_positivity(enable_x64):
    solver = hp.solvers.ImplicitPredictorCorrector(
        dt=1/16, order=.5, newton_iterations=1, weight_evaluation="stable",
    )
    result = hp.solvers.simulate(
        model=lambda t, y, rate, **kw: -rate*y, ts=jnp.linspace(0., 1., 17),
        solver=solver, initial_state=jnp.asarray(1.), params=jnp.asarray(30.),
    )
    mu = 30 * (1/16)**.5 / gamma(2.5)
    np.testing.assert_allclose(result.latent_state[1], (1-.5*mu)/(1+mu), atol=1e-13)
    assert result.latent_state[1] < 0 < erfcx(30/4)
    assert result.solver_info.diagnostics["newton_converged"]


@pytest.mark.parametrize("solver_class", [hp.solvers.PredictorCorrector,
                                         hp.solvers.ImplicitPredictorCorrector])
def test_option_guards_singletons_and_bounded_graph(solver_class):
    with pytest.raises(ValueError, match="weight_evaluation"):
        solver_class(dt=.1, order=.5, weight_evaluation="unknown")

    def make(n):
        solver = solver_class(dt=.1, order=.5, weight_evaluation="stable")
        def run(y):
            return hp.solvers.simulate(
                model=lambda t, y, p, **kw: p*y, ts=jnp.arange(n)*.1,
                solver=solver, initial_state=y, params=-.5,
            ).latent_state
        return run

    assert len(jax.make_jaxpr(make(16))(1.).jaxpr.eqns) == len(
        jax.make_jaxpr(make(128))(1.).jaxpr.eqns,
    )
    np.testing.assert_array_equal(make(1)(jnp.asarray([1., 2.])), [[1., 2.]])


def test_integer_history_dtype_keeps_fractional_weights():
    pred, corr = _stable_pece_weights(jnp.asarray(2), jnp.arange(3), .5, jnp.int32)
    assert jnp.issubdtype(pred.dtype, jnp.floating)
    assert pred[0] > 0 and corr[0] > 0


def test_stable_weights_reject_vector_orders():
    with pytest.raises(NotImplementedError, match="scalar order"):
        _stable_pece_weights(jnp.asarray(2), jnp.arange(3),
                              jnp.asarray([.3, .7]), jnp.float32)


def test_stochastic_helper_rejects_unvalidated_weight_option():
    with pytest.raises(NotImplementedError, match="deterministic"):
        hp.prob.simulate_stochastic(
            model=lambda t, y, p, **kw: -y,
            diffusion=lambda t, y, p, **kw: jnp.zeros_like(y),
            ts=jnp.asarray([0., .1]), initial_state=jnp.asarray(1.), params=None,
            rng_key=jax.random.PRNGKey(0),
            solver=hp.solvers.PredictorCorrector(
                dt=.1, order=.7, weight_evaluation="stable",
            ),
        )
