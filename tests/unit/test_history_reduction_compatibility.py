"""Compatibility of the shared full-history reduction, not general IVP accuracy.

Complex checks exercise only the private bilinear helper; they do not establish
public complex implicit-solver support. All dtype flags are scoped outside JIT.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.numerical.implicit_validation import independent_residuals
from hpfracc.solvers import (
    AdaptivePredictorCorrector,
    ImplicitPredictorCorrector,
    NonUniformPredictorCorrector,
    PredictorCorrector,
    simulate,
)
from hpfracc.solvers.predictor_corrector import _history_dot

_DTYPES = [
    pytest.param(jnp.float32, False, id="float32"),
    pytest.param(jnp.float64, True, id="float64"),
    pytest.param(jnp.complex64, False, id="complex64"),
    pytest.param(jnp.complex128, True, id="complex128"),
]


def _host_dot(weights, history):
    """Independent host product and compensated real/imaginary component sums."""
    dtype = (
        np.complex128
        if np.iscomplexobj(weights) or np.iscomplexobj(history)
        else np.float64
    )
    weights, history = np.asarray(weights, dtype), np.asarray(history, dtype)
    terms = weights.reshape((-1,) + (1,) * (history.ndim - 1)) * history
    columns = terms.reshape(len(weights), -1).T
    if np.iscomplexobj(terms):
        sums = [complex(math.fsum(c.real), math.fsum(c.imag)) for c in columns]
    else:
        sums = [math.fsum(c) for c in columns]
    return np.asarray(sums, dtype=dtype).reshape(history.shape[1:])


def _decay(t, state, params, *, rng_key=None, inputs=None):
    return -params * state


@pytest.mark.parametrize("dtype,wide", _DTYPES)
@pytest.mark.parametrize("trailing", [(), (3,), (2, 1, 3, 2)])
def test_history_dot_signed_mixed_scale_shapes_and_dtypes(dtype, wide, trailing):
    with jax.enable_x64(wide):
        shape = (19, *trailing)
        indices = np.arange(math.prod(shape)).reshape(shape)
        history_host = ((indices % 11) - 5) * np.exp2((indices % 13) - 6)
        weights_host = ((np.arange(19) % 7) - 3) / 8
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            history_host = history_host + 1j * np.flip(history_host, axis=0) / 4
            weights_host = weights_host + 1j * np.flip(weights_host) / 2
        history = jnp.asarray(history_host, dtype=dtype)
        weights = jnp.asarray(weights_host, dtype=dtype)
        expected = _host_dot(weights, history)
        products = np.asarray(weights, dtype=expected.dtype).reshape(
            (-1,) + (1,) * len(trailing)
        ) * np.asarray(history, dtype=expected.dtype)
        eps = np.finfo(np.dtype(dtype)).eps
        # Case-specific roundoff allowance, scaled by the absolute term sum;
        # this is not an accuracy guarantee for arbitrary ill-conditioned sums.
        budget = 8 * eps * np.maximum(1, np.sum(np.abs(products), axis=0))
        for evaluate in (_history_dot, jax.jit(_history_dot)):
            actual = evaluate(weights, history)
            assert weights.dtype == history.dtype == actual.dtype == dtype
            assert actual.shape == trailing
            assert np.all(np.isfinite(np.asarray(actual)))
            assert np.all(np.abs(np.asarray(actual) - expected) <= budget)


@pytest.mark.parametrize("dtype,wide", _DTYPES)
def test_history_dot_jvp_vjp_match_independent_bilinear_derivatives(dtype, wide):
    with jax.enable_x64(wide):
        history_host = np.arange(30).reshape(5, 2, 3) / 8 - 1.5
        weights_host = np.asarray([-0.75, 0.125, 1.25, -2.0, 0.5])
        cotangent_host = np.asarray([[0.5, -0.25, 1.0], [0.125, -0.75, 0.375]])
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            history_host = history_host + 0.5j * np.flip(history_host, axis=0)
            weights_host = weights_host + 0.25j * np.flip(weights_host)
            cotangent_host = cotangent_host + 0.5j * np.flip(cotangent_host)
        weights = jnp.asarray(weights_host, dtype=dtype)
        history = jnp.asarray(history_host, dtype=dtype)
        dw = jnp.asarray(np.linspace(-0.3, 0.4, 5), dtype=dtype)
        dh = jnp.asarray((np.arange(30).reshape(5, 2, 3) % 5 - 2) / 10, dtype=dtype)
        cotangent = jnp.asarray(cotangent_host, dtype=dtype)

        value, tangent = jax.jit(
            lambda w, h, tw, th: jax.jvp(_history_dot, (w, h), (tw, th))
        )(weights, history, dw, dh)
        grad_weights, grad_history = jax.jit(
            lambda w, h, c: jax.vjp(_history_dot, w, h)[1](c)
        )(weights, history, cotangent)
        expected_value = _host_dot(weights, history)
        expected_tangent = _host_dot(dw, history) + _host_dot(weights, dh)
        host_history = np.asarray(history, dtype=expected_value.dtype)
        host_cotangent = np.asarray(cotangent, dtype=expected_value.dtype)
        # JAX's complex VJP uses the bilinear transpose, not a Hermitian
        # transpose. Neither the forward contraction nor this pullback conjugates.
        expected_weights = _host_dot(
            host_cotangent.ravel(), host_history.reshape(5, -1).T
        )
        expected_history = (
            np.asarray(weights, dtype=expected_value.dtype).reshape(5, 1, 1)
            * host_cotangent
        )
        real_dtype = np.finfo(np.dtype(dtype)).dtype
        objective = jnp.sum(jnp.abs(value) ** 2)
        assert objective.dtype == real_dtype
        assert np.isfinite(float(objective))
        eps = np.finfo(real_dtype).eps
        for actual, expected in (
            (value, expected_value),
            (tangent, expected_tangent),
            (grad_weights, expected_weights),
            (grad_history, expected_history),
        ):
            assert actual.dtype == dtype
            assert actual.shape == expected.shape
            np.testing.assert_allclose(actual, expected, rtol=16 * eps, atol=16 * eps)


def test_history_dot_objective_directional_derivative_matches_finite_difference():
    with jax.enable_x64(True):
        weights = jnp.asarray([-0.5, 0.25, 1.0, -0.125], dtype=jnp.float64)
        history = jnp.asarray(np.arange(16).reshape(4, 2, 2) / 8 - 1, jnp.float64)
        dw = jnp.asarray([0.125, -0.25, 0.5, 0.25], dtype=jnp.float64)
        dh = jnp.full_like(history, 0.125)

        @jax.jit
        def objective(w, h):
            return jnp.sum(_history_dot(w, h) ** 2)

        value, derivative = jax.jvp(objective, (weights, history), (dw, dh))
        step = 1e-5
        finite_difference = (
            objective(weights + step * dw, history + step * dh)
            - objective(weights - step * dw, history - step * dh)
        ) / (2 * step)
        expected = 2 * np.sum(
            _host_dot(weights, history)
            * (_host_dot(dw, history) + _host_dot(weights, dh))
        )
        assert value.dtype == derivative.dtype == finite_difference.dtype == jnp.float64
        np.testing.assert_allclose(derivative, expected, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(finite_difference, expected, rtol=2e-8, atol=1e-9)


@pytest.mark.parametrize("evaluation", ["stable", "direct"])
def test_dynamic_float32_order_jit_stiff_decay_independent_residual(
    evaluation, record_testsuite_property
):
    """Uninstrumented public solve: algebraic acceptance, not continuous accuracy."""
    with jax.enable_x64(False):
        times = jnp.linspace(0.0, 1.0, 257, dtype=jnp.float32)
        initial = jnp.ones(1, dtype=jnp.float32)
        order = jnp.asarray(0.5, dtype=jnp.float32)
        rate = jnp.asarray(30.0, dtype=jnp.float32)

        @jax.jit
        def run(alpha):
            result = simulate(
                model=_decay,
                ts=times,
                initial_state=initial,
                params=rate,
                solver=ImplicitPredictorCorrector(
                    dt=1 / 256,
                    order=alpha,
                    newton_iterations=8,
                    newton_rtol=1e-5,
                    newton_atol=1e-7,
                    weight_evaluation=evaluation,
                ),
            )
            assert result.solver_info is not None
            return result.latent_state, result.solver_info.diagnostics[
                "newton_converged"
            ]

        states, converged = run(order)
        assert not order.weak_type
        assert order.dtype == rate.dtype == states.dtype == jnp.float32
        assert states[-1].sum().dtype == jnp.float32
        assert bool(converged)
        assert np.all(np.isfinite(np.asarray(states)))
        # Widen the trajectory before evaluating the independent host RHS.
        host_states = np.asarray(states, dtype=np.float64)
        residuals = independent_residuals(
            "half_decay",
            np.asarray(times),
            host_states,
            30.0,
            0.5,
            np.ones(1, dtype=np.float64),
            1e-5,
            1e-7,
        )
        maximum = float(np.max(residuals))
        record_testsuite_property(
            f"{evaluation}_max_independent_scaled_residual", maximum
        )
        assert maximum <= 1.0


def _host_uniform_response(n_time, dt, alpha, rate, implicit):
    """Short scalar discrete recurrence, independent of solver weight helpers."""
    predictor_scale = dt**alpha / math.gamma(alpha + 1)
    corrector_scale = dt**alpha / math.gamma(alpha + 2)
    values = [1.0]
    for step in range(1, n_time):
        predictor = 1 - predictor_scale * math.fsum(
            ((step - j) ** alpha - (step - j - 1) ** alpha) * rate * y
            for j, y in enumerate(values)
        )
        corrector_weights = [
            (step - 1) ** (alpha + 1) - (step - 1 - alpha) * step**alpha
        ] + [
            (lag + 1) ** (alpha + 1) + (lag - 1) ** (alpha + 1) - 2 * lag ** (alpha + 1)
            for lag in range(step - 1, 0, -1)
        ]
        known = 1 - corrector_scale * math.fsum(
            w * rate * y for w, y in zip(corrector_weights, values, strict=True)
        )
        values.append(
            known / (1 + corrector_scale * rate)
            if implicit
            else known - corrector_scale * rate * predictor
        )
    return np.asarray(values)


@pytest.mark.parametrize("evaluation", ["direct", "stable"])
@pytest.mark.parametrize("implicit", [False, True], ids=["explicit", "implicit"])
def test_uniform_batched_paths_preserve_recurrence_and_state_gradient(
    evaluation, implicit
):
    with jax.enable_x64(False):
        times = jnp.linspace(0.0, 0.5, 9, dtype=jnp.float32)
        order = jnp.asarray(0.65, dtype=jnp.float32)
        rate = jnp.asarray(0.4, dtype=jnp.float32)
        initial = jnp.asarray([[1.0, -2.0, 0.03125], [4.0, 0.5, -0.125]], jnp.float32)
        solver_type = ImplicitPredictorCorrector if implicit else PredictorCorrector

        @jax.jit
        def run(alpha, y0, strength):
            return simulate(
                model=_decay,
                ts=times,
                initial_state=y0,
                params=strength,
                solver=solver_type(
                    dt=1 / 16, order=alpha, weight_evaluation=evaluation
                ),
            ).latent_state

        states = run(order, initial, rate)
        objective = states[-1].sum()
        gradient = jax.jit(jax.grad(lambda y0: run(order, y0, rate)[-1].sum()))(initial)
        response = _host_uniform_response(
            9, 1 / 16, float(order), float(rate), implicit
        )
        expected = response[:, None, None] * np.asarray(initial, dtype=np.float64)
        assert states.shape == (9, 2, 3)
        assert states.dtype == objective.dtype == gradient.dtype == jnp.float32
        np.testing.assert_allclose(states, expected, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(gradient, response[-1], rtol=2e-6, atol=2e-7)


def _affine_forcing(t, state, params, *, rng_key=None, inputs=None):
    return t * params


@pytest.mark.parametrize(
    "adaptive", [False, True], ids=["supplied-nonuniform", "adaptive"]
)
def test_variable_mesh_batched_forcing_moments_and_gradients(adaptive):
    """Affine forcing has exact interpolant moments on any realized mesh."""
    with jax.enable_x64(True):
        times = jnp.asarray(
            [0.0, 0.25] if adaptive else [0.0, 0.02, 0.08, 0.15, 0.25],
            dtype=jnp.float64,
        )
        order = jnp.asarray(0.6, dtype=jnp.float64)
        initial = jnp.asarray([[1.0, -2.0, 0.125], [0.5, 4.0, -0.25]], jnp.float64)
        forcing = jnp.asarray([[0.5, -1.0, 2.0], [-0.25, 0.125, -2.0]], jnp.float64)
        solver = (
            AdaptivePredictorCorrector(
                order=order,  # type: ignore[arg-type]
                rtol=1e-4,
                atol=1e-6,
                first_step=0.0625,
                max_steps=8,
            )
            if adaptive
            else NonUniformPredictorCorrector(order=order)  # type: ignore[arg-type]
        )

        @jax.jit
        def run(y0, amplitude):
            result = simulate(
                model=_affine_forcing,
                ts=times,
                solver=solver,
                initial_state=y0,
                params=amplitude,
            )
            assert result.solver_info is not None
            diagnostics = result.solver_info.diagnostics
            valid = (
                jnp.asarray(diagnostics["valid_mask"])
                if adaptive
                else jnp.ones(times.shape, dtype=bool)
            )
            success = jnp.asarray(diagnostics["success"] if adaptive else True)
            return result.ts, result.latent_state, valid, success

        mesh, states, valid, success = run(initial, forcing)
        assert bool(success)
        mask = np.asarray(valid)
        assert np.array_equal(mask, np.arange(len(mask)) < np.count_nonzero(mask))
        assert np.count_nonzero(mask) >= 3
        assert float(mesh[mask][-1]) == 0.25
        assert states.shape == (len(mesh), 2, 3)
        coefficients = np.asarray(mesh[mask]) ** (float(order) + 1) / math.gamma(
            float(order) + 2
        )
        expected = np.asarray(initial) + coefficients[:, None, None] * np.asarray(
            forcing
        )
        np.testing.assert_allclose(states[mask], expected, rtol=2e-12, atol=2e-12)

        def objective(y0, amplitude):
            return run(y0, amplitude)[1][-1].sum()

        grad_initial, grad_forcing = jax.jit(jax.grad(objective, argnums=(0, 1)))(
            initial, forcing
        )
        assert states.dtype == objective(initial, forcing).dtype == jnp.float64
        assert grad_initial.dtype == grad_forcing.dtype == jnp.float64
        np.testing.assert_allclose(grad_initial, 1.0, rtol=2e-12, atol=2e-12)
        # Adaptive sensitivities hold the realized mesh fixed; no controller AD.
        np.testing.assert_allclose(
            grad_forcing, coefficients[-1], rtol=2e-12, atol=2e-12
        )
