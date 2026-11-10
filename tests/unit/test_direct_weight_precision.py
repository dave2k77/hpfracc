"""Direct uniform weights must meet independent working-precision criteria."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.numerical import implicit_validation as implicit
from benchmarks.numerical.uniform_weight_validation import reference_row
from hpfracc.solvers import ImplicitPredictorCorrector, PredictorCorrector, simulate
from hpfracc.solvers.predictor_corrector import _pece_weights


@pytest.fixture(params=[jnp.float32, jnp.float64], ids=["effective-float32", "float64"])
def working_dtype(request):
    """Exercise actual forward precision, restoring even an ambient x64 setting."""
    dtype = request.param
    previous = jax.config.read("jax_enable_x64")
    with jax.enable_x64(dtype == jnp.float64):
        assert jax.config.read("jax_enable_x64") == (dtype == jnp.float64)
        assert jnp.asarray(1.).dtype == dtype
        yield dtype
    assert jax.config.read("jax_enable_x64") == previous


def _orders(dtype):
    scalar = np.dtype(dtype).type
    return (.01, .5, .99, 1e-20, np.nextafter(scalar(1), scalar(0)))


def test_direct_float32_coefficients_match_independent_panel_integrals():
    step = 256
    with jax.enable_x64(False):
        alpha = jnp.asarray(.5, dtype=jnp.float32)
        indices = jnp.arange(step + 3, dtype=jnp.int32)
        actual = jax.jit(lambda a: _pece_weights(
            jnp.asarray(step, dtype=jnp.int32), indices, a, jnp.float32,
        ))(alpha)
        expected = reference_row(step, float(alpha))
        for values, truth in zip(actual, expected, strict=True):
            assert values.dtype == jnp.float32
            np.testing.assert_allclose(np.asarray(values[:step]), truth,
                                       rtol=5e-6, atol=0.)
            np.testing.assert_array_equal(values[step:], 0.)
            assert np.all(np.asarray(values[:step]) > 0.)


def test_default_direct_float32_decay_meets_independent_residual():
    with jax.enable_x64(False):
        row = implicit.run_case("half_decay", 257, 8, 1., .5, "float32")
    assert row.measurement_ok and row.newton_converged
    assert row.max_dimensionless_error is not None
    assert row.independent_max_scaled_residual is not None
    assert row.max_dimensionless_error <= 1e-3
    assert row.independent_max_scaled_residual <= 1.
    assert row.residual_flag_disagreements == 0


def test_dynamic_direct_coefficients_and_full_history_moments(working_dtype):
    """One compiled shape covers small/long histories and both open endpoints."""
    dtype = working_dtype
    indices = jnp.arange(4099, dtype=jnp.int32)
    tolerance = 5e-6 if dtype == jnp.float32 else 5e-12

    @jax.jit
    def weights(step, order):
        return _pece_weights(step, indices, order, dtype)

    for step in (1, 2, 3, 256, 4096):
        for value in _orders(dtype):
            order = jnp.asarray(value, dtype=dtype)
            assert 0. < float(order) < 1.
            actual = weights(jnp.asarray(step, dtype=jnp.int32), order)
            # This oracle factors out alpha: unlike subtracting powers with
            # exponent 1+alpha, its panels still resolve alpha=1e-20. No dense
            # high-precision reference or cancellation-prone tiny-order oracle.
            expected = reference_row(step, float(order))
            for found, truth in zip(actual, expected, strict=True):
                assert found.dtype == dtype
                host = np.asarray(found)
                assert np.all(np.isfinite(host))
                assert np.all(host[:step] > 0.)
                np.testing.assert_array_equal(host[step:], 0.)
                np.testing.assert_allclose(host[:step], truth,
                                           rtol=tolerance, atol=0.)

            # Sum the entire history in host float64 to isolate coefficient
            # accuracy from a separate float32 reduction error. The corrector's
            # current-stage diagonal has coefficient one and is applied outside
            # _pece_weights, so include it exactly once in both moments.
            predictor, corrector = (np.asarray(w, dtype=np.float64) for w in actual)
            alpha = float(order)
            np.testing.assert_allclose(predictor.sum(), step**alpha,
                                       rtol=tolerance, atol=0.)
            np.testing.assert_allclose(corrector.sum() + 1.,
                                       (alpha + 1.) * step**alpha,
                                       rtol=tolerance, atol=0.)
            np.testing.assert_allclose(corrector @ np.arange(indices.size) + step,
                                       step**(alpha + 1.),
                                       rtol=tolerance, atol=0.)


def test_direct_masked_jvp_vjp_and_first_step_derivative(working_dtype):
    dtype = working_dtype
    indices = jnp.arange(8, dtype=jnp.int32)

    @jax.jit
    def derivatives(step, order):
        def weights(a):
            return jnp.stack(_pece_weights(step, indices, a, dtype))

        values, tangent = jax.jvp(weights, (order,), (jnp.ones_like(order),))
        _, pullback = jax.vjp(weights, order)
        current = jnp.broadcast_to(indices == step, values.shape).astype(dtype)
        future = jnp.broadcast_to(indices > step, values.shape).astype(dtype)
        boundary = jnp.zeros_like(values).at[1, 0].set(1.)
        return (values, tangent, pullback(current)[0], pullback(future)[0],
                pullback(boundary)[0])

    for step in (0, 1, 3):
        for value in _orders(dtype):
            order = jnp.asarray(value, dtype=dtype)
            values, tangent, current_vjp, future_vjp, boundary_vjp = derivatives(
                jnp.asarray(step, dtype=jnp.int32), order,
            )
            for array in (values, tangent, current_vjp, future_vjp, boundary_vjp):
                assert array.dtype == dtype
                assert np.all(np.isfinite(np.asarray(array)))
            np.testing.assert_array_equal(values[:, step:], 0.)
            np.testing.assert_array_equal(tangent[:, step:], 0.)
            np.testing.assert_array_equal(current_vjp, 0.)
            np.testing.assert_array_equal(future_vjp, 0.)
            if step == 0:
                np.testing.assert_array_equal(boundary_vjp, 0.)
            elif step == 1:
                np.testing.assert_array_equal(values[:, 0], [1., float(order)])
                np.testing.assert_array_equal(tangent[:, 0], [0., 1.])
                np.testing.assert_array_equal(boundary_vjp, 1.)


def test_direct_integer_history_and_strong_order_promotion(working_dtype):
    dtype = working_dtype
    indices = jnp.arange(6, dtype=jnp.int32)

    @jax.jit
    def promoted_weights(step, order):
        return (_pece_weights(step, indices, order, jnp.int32),
                _pece_weights(step, indices, order, jnp.float32))

    order = jnp.asarray(.5, dtype=dtype)
    expected = reference_row(3, float(order))
    tolerance = 5e-6 if dtype == jnp.float32 else 5e-12
    for pair in promoted_weights(jnp.asarray(3, dtype=jnp.int32), order):
        for found, truth in zip(pair, expected, strict=True):
            assert found.dtype == dtype
            np.testing.assert_allclose(found[:3], truth, rtol=tolerance, atol=0.)
            np.testing.assert_array_equal(found[3:], 0.)
            assert np.all(np.asarray(found[:3]) > 0.)


def _nonlinear_model(t, state, params, *, rng_key=None, inputs=None):
    del rng_key, inputs
    return (-params[0] * state**3 + params[1] * state[::-1]
            + .1 * jnp.stack((t*t, jnp.sin(t))))


@pytest.mark.parametrize("solver_class", [PredictorCorrector,
                                         ImplicitPredictorCorrector])
def test_direct_nonlinear_solver_order_and_parameter_gradients(
    solver_class, working_dtype,
):
    """Differentiate the discrete nonlinear solve, not a continuous surrogate."""
    dtype = working_dtype
    times = jnp.arange(9, dtype=dtype) / 8
    initial = jnp.asarray([.7, .4], dtype=dtype)
    params = jnp.asarray([.3, .1], dtype=dtype)
    epsilon = 2e-3 if dtype == jnp.float32 else 1e-5
    rtol, atol = (2e-3, 1e-5) if dtype == jnp.float32 else (2e-6, 1e-8)

    def objective(order, parameters):
        trajectory = simulate(
            model=_nonlinear_model, ts=times, initial_state=initial,
            params=parameters, solver=solver_class(dt=1/8, order=order),
        ).latent_state
        loss = jnp.sum(trajectory[-1]**2) + .05 * jnp.mean(trajectory**2)
        return loss, trajectory

    # Reuse this executable for finite differences too: only two small solver
    # shapes per precision (explicit/implicit), not a compile for every order.
    evaluate = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))

    def value(order, parameters):
        return evaluate(order, parameters)[0][0]

    for alpha in (.2, .8):
        order = jnp.asarray(alpha, dtype=dtype)
        (loss, trajectory), (order_grad, params_grad) = evaluate(order, params)
        assert trajectory.shape == (9, 2)
        for array in (initial, times, order, params, trajectory, loss,
                      order_grad, params_grad):
            assert array.dtype == dtype
            assert np.all(np.isfinite(np.asarray(array)))
        # Require nontrivial sensitivities so a zero-gradient path cannot pass
        # merely through the absolute finite-difference allowance.
        assert abs(float(order_grad)) > 1e-4
        assert np.all(np.abs(np.asarray(params_grad)) > 1e-4)
        fd_order = (value(order + epsilon, params) - value(order - epsilon, params))
        fd_order = fd_order / (2 * epsilon)
        np.testing.assert_allclose(order_grad, fd_order, rtol=rtol, atol=atol)
        for index in range(params.size):
            delta = jnp.zeros_like(params).at[index].set(epsilon)
            fd_param = (value(order, params + delta) - value(order, params - delta))
            fd_param = fd_param / (2 * epsilon)
            np.testing.assert_allclose(params_grad[index], fd_param,
                                       rtol=rtol, atol=atol)


@pytest.mark.parametrize("solver_class", [PredictorCorrector,
                                         ImplicitPredictorCorrector])
def test_repaired_default_and_explicit_direct_selector_metadata(solver_class):
    previous = jax.config.read("jax_enable_x64")
    with jax.enable_x64(False):
        times = jnp.arange(9, dtype=jnp.float32) / 8
        order = jnp.asarray(.5, dtype=jnp.float32)

        def run(**options):
            return simulate(
                model=_nonlinear_model, ts=times,
                initial_state=jnp.asarray([.7, .4], dtype=jnp.float32),
                params=jnp.asarray([.3, .1], dtype=jnp.float32),
                solver=solver_class(dt=1/8, order=order, **options),
            )

        default, explicit = run(), run(weight_evaluation="direct")
        assert default.latent_state.dtype == explicit.latent_state.dtype == jnp.float32
        np.testing.assert_array_equal(default.latent_state, explicit.latent_state)
        np.testing.assert_array_equal(default.ts, explicit.ts)
        # Compare metadata without adding a serialization contract: a strongly
        # typed JAX order remains an array in SolverInfo.to_dict(). Both selectors
        # must identify the same algorithm, history and Newton diagnostics.
        assert default.solver_info is not None
        assert explicit.solver_info is not None
        default_info = default.solver_info.to_dict()
        explicit_info = explicit.solver_info.to_dict()
        assert default_info == explicit_info
        assert default_info["n_steps"] == 8
        assert default_info["step_size"] == 1/8
        assert default_info["fractional_order"] == .5
        assert default_info["diagnostics"]["history"] == "full"
        assert default_info["diagnostics"]["grid"] == "uniform"
        assert default_info["diagnostics"]["weight_evaluation"] == "direct"
        if solver_class is ImplicitPredictorCorrector:
            assert default_info["diagnostics"]["newton_converged"]
            assert default_info["diagnostics"]["newton_nonfinite_count"] == 0
    assert jax.config.read("jax_enable_x64") == previous
