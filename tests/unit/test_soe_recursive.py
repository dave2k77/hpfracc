"""SOE recurrence, positive-tail fitting, analytic and gradient contracts."""

import json
from math import gamma

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gammaln

from hpfracc.ops import caputo, grunwald_letnikov, kernels, riemann_liouville
from hpfracc.ops._soe import _apply_soe, _fit_soe, _kernel_errors


def target_weights(alpha, n, kernel):
    """Host formulas independent of recursive fitter sufficient statistics."""
    k = np.arange(n, dtype=np.float64)
    if kernel == "gl_prefix":
        return np.exp(gammaln(k + 1 - alpha) - gammaln(1 - alpha) - gammaln(k + 1))
    if kernel == "l1":
        return np.r_[1., k[1:]**(1 - alpha)
                     * np.expm1((1 - alpha) * np.log1p(1 / k[1:]))]
    weights = np.ones(n)
    for i in range(1, n):
        weights[i] = weights[i - 1] * (i - 1 - alpha) / i
    return weights


def fitted_weights(fit):
    rho, c = np.asarray(fit.rhos), np.asarray(fit.coefficients)
    if fit.exact_head:
        return np.r_[1., (rho[None, :] ** np.arange(fit.n - 1)[:, None]) @ c]
    return (rho[None, :] ** np.arange(fit.n)[:, None]) @ c


def dense_fit_apply(values, fit):
    signal = np.asarray(values)
    if fit.kernel == "gl_prefix":
        signal = np.concatenate((signal[:1], np.diff(signal, axis=0)), axis=0)
    flat = signal.reshape(len(signal), -1)
    weights = fitted_weights(fit)
    out = np.stack([np.convolve(flat[:, i], weights)[:len(flat)]
                    for i in range(flat.shape[1])], axis=1)
    return out.reshape(signal.shape)


@pytest.mark.parametrize("kernel", ["l1", "gl"])
@pytest.mark.parametrize("n", [17, 129])
def test_streamed_fit_matches_independent_dense_normalized_fit(kernel, n, enable_x64):
    fit = _fit_soe(.5, n, n_poles=12, t_max=1., kernel=kernel, method="scaled_tail")
    rho = np.asarray(fit.rhos)
    design = rho[None, :] ** np.arange(n - 1)[:, None]
    target = target_weights(.5, n, fit.kernel)[1:]
    norms = np.linalg.norm(design, axis=0)
    normalized = design / norms
    gram, rhs = normalized.T @ normalized, normalized.T @ target
    bound = np.max(np.sum(np.abs(gram), axis=1))
    z = np.zeros(len(rho))
    initial_loss = np.sum(target**2)
    for _ in range(400):
        z = np.maximum(z - (gram @ z - rhs) / bound, 0.)
    np.testing.assert_allclose(fit.coefficients, z / norms, rtol=2e-9, atol=2e-11)
    assert np.sum((design @ np.asarray(fit.coefficients) - target)**2) < initial_loss
    assert np.all(np.asarray(fit.coefficients) >= 0)
    assert np.all((rho > 0) & (rho <= 1))
    # Metadata reports actual residuals, not the cancellation-prone Gram objective.
    error = fitted_weights(fit) - target_weights(.5, n, fit.kernel)
    maximum, relative, finite = _kernel_errors(fit, .5)
    assert finite
    np.testing.assert_allclose(maximum, np.max(np.abs(error)), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(relative, np.linalg.norm(error) / np.linalg.norm(
        target_weights(.5, n, fit.kernel)), rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("kernel", ["l1", "gl"])
@pytest.mark.parametrize("method", ["legacy", "scaled_tail"])
def test_recursive_sum_matches_dense_fitted_kernel(kernel, method, enable_x64):
    fit = _fit_soe(.6, 33, n_poles=8, t_max=2., kernel=kernel, method=method)
    values = jnp.sin(jnp.arange(198.).reshape(33, 2, 3)) * (1 + .25j)
    actual = jax.jit(lambda x: _apply_soe(x, fit))(values)
    np.testing.assert_allclose(actual, dense_fit_apply(values, fit),
                               rtol=1e-10, atol=3e-10)
    if method == "legacy":
        legacy_weights = kernels._soe_weights(
            .6, 33, n_poles=8, t_max=2., kernel=kernel,
        )
        legacy = kernels._full_history_convolution(values, legacy_weights)
        np.testing.assert_allclose(actual, legacy, rtol=1e-10, atol=3e-10)


@pytest.mark.parametrize("fn", [caputo, grunwald_letnikov])
@pytest.mark.parametrize("method", ["legacy", "scaled_tail"])
@pytest.mark.parametrize("alpha", [.3, .6, .85])
def test_input_and_order_gradients_against_finite_differences(
    fn, method, alpha, enable_x64,
):
    t = jnp.linspace(0., 1., 33)
    values = jnp.stack((jnp.sin(t) + .3 * t**2, .2 + t**2), axis=1)
    direction = jnp.cos(values)

    def objective(x, a):
        result = fn(x, dt=1 / 32, order=a, history="soe", soe_fit=method)
        return jnp.mean(result**2)

    a = jnp.asarray(alpha)
    eager = jax.grad(objective, argnums=(0, 1))(values, a)
    compiled = jax.jit(jax.grad(objective, argnums=(0, 1)))(values, a)
    for actual, expected in zip(compiled, eager, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-8)
    eps = 1e-5
    fd_input = (objective(values + eps * direction, a)
                - objective(values - eps * direction, a)) / (2 * eps)
    fd_alpha = (objective(values, a + eps) - objective(values, a - eps)) / (2 * eps)
    np.testing.assert_allclose(jnp.sum(compiled[0] * direction), fd_input,
                               rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(compiled[1], fd_alpha, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("fn,limit", [(caputo, 5e-4), (riemann_liouville, 3e-3)])
def test_selected_polynomial_refinement(fn, limit, enable_x64):
    errors = []
    for n in (64, 256):
        t = jnp.linspace(0., 1., n)
        actual = fn(t**2, dt=1 / (n - 1), order=.5, history="soe",
                    soe_fit="scaled_tail", soe_poles=12)
        expected = 2 * t**1.5 / gamma(2.5)
        errors.append(float(jnp.max(jnp.abs(actual - expected))))
    assert errors[1] < errors[0] and errors[1] < limit


def test_caputo_and_rl_constant_distinction(enable_x64):
    n = 257
    t = jnp.linspace(0., 1., n)
    kwargs = dict(dt=1 / (n - 1), order=.5, history="soe", soe_fit="scaled_tail",
                  soe_poles=12)
    np.testing.assert_array_equal(caputo(jnp.ones(n), **kwargs), jnp.zeros(n))
    actual = riemann_liouville(jnp.ones(n), **kwargs)
    truth = np.asarray(t[64:])**(-.5) / gamma(.5)
    np.testing.assert_allclose(actual[64:], truth, rtol=.01)
    assert np.all(np.asarray(actual) > 0)


@pytest.mark.parametrize("fn", [caputo, grunwald_letnikov, riemann_liouville])
@pytest.mark.parametrize("method", ["legacy", "scaled_tail"])
def test_public_application_is_recursive_and_metadata_is_json(fn, method, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Public SOE must not materialize a dense history")

    monkeypatch.setattr(kernels, "_full_history_convolution", forbidden)
    monkeypatch.setattr(kernels, "_soe_weights", forbidden)
    values = jnp.linspace(0., 1., 33)**2
    result = fn(values, dt=1 / 32, order=.5, history="soe",
                soe_fit=method, return_info=True)
    info = json.loads(json.dumps(result.operator_info.to_dict(), allow_nan=False))
    assert info["diagnostics"]["soe_application"] == "recursive"
    assert info["diagnostics"]["soe_fit"] == method
    assert info["diagnostics"]["soe_fit_finite"]
    if method == "scaled_tail" and fn is not caputo:
        assert info["diagnostics"]["soe_target_kernel"] == "gl_prefix"


@pytest.mark.parametrize("fn", [caputo, grunwald_letnikov, riemann_liouville])
@pytest.mark.parametrize("kwargs", [
    {"soe_fit": "unknown"}, {"soe_poles": 0}, {"soe_poles": 1.5},
    {"soe_poles": True}, {"soe_t_max": np.inf}, {"soe_t_max": np.nan},
])
def test_invalid_soe_configuration_including_singletons(fn, kwargs):
    for n in (1, 5):
        with pytest.raises(ValueError, match="soe_|n_poles|t_max"):
            fn(jnp.ones(n), dt=.1, order=.5, history="soe", **kwargs)


@pytest.mark.parametrize("fn", [caputo, grunwald_letnikov, riemann_liouville])
def test_vector_orders_rejected_even_for_singletons(fn):
    with pytest.raises(NotImplementedError, match="vector"):
        fn(jnp.ones((1, 2)), dt=.1, order=jnp.asarray([.3, .7]), history="soe",
           soe_fit="scaled_tail")


def test_empty_and_single_weight_histories():
    kwargs = dict(dt=.25, order=.5, history="soe", soe_fit="scaled_tail")
    np.testing.assert_array_equal(caputo(jnp.ones((1, 2)), **kwargs), jnp.zeros((1, 2)))
    np.testing.assert_allclose(grunwald_letnikov(jnp.ones((1, 2)), **kwargs), 2.)
    actual = caputo(jnp.asarray([1., 2.]), **kwargs)
    np.testing.assert_allclose(actual, [0., 2 / gamma(1.5)], rtol=2e-6)


def test_bounded_fit_graph_and_outer_vmap(enable_x64):
    def trace(n):
        def run(a):
            fit = _fit_soe(a, n, n_poles=8, t_max=1., kernel="gl", method="scaled_tail")
            return fit.rhos, fit.coefficients
        return jax.make_jaxpr(run)(jnp.asarray(.5))

    assert len(trace(32).jaxpr.eqns) == len(trace(256).jaxpr.eqns)

    def inspect_graph(graph):
        for equation in graph.eqns:
            for variable in equation.outvars:
                assert 256 not in getattr(variable.aval, "shape", ())
            for parameter in equation.params.values():
                if hasattr(parameter, "jaxpr"):
                    inspect_graph(parameter.jaxpr)

    # The fitter must not build a length-by-poles design matrix or target vector.
    inspect_graph(trace(256).jaxpr)
    values = jnp.sin(jnp.arange(66.).reshape(2, 33))
    def apply(x):
        return caputo(x, dt=.1, order=.5, history="soe", soe_fit="scaled_tail")

    actual = jax.jit(jax.vmap(apply))(values)
    np.testing.assert_allclose(actual, jnp.stack([apply(x) for x in values]),
                               rtol=1e-12, atol=1e-12)


def test_legacy_projected_fit_preserves_original_updates(enable_x64):
    fit = _fit_soe(.5, 33, n_poles=8, t_max=1., kernel="l1", method="legacy")
    design = np.asarray(fit.rhos)[None, :] ** np.arange(33)[:, None]
    target = target_weights(.5, 33, "l1")
    gram, rhs = design.T @ design, design.T @ target
    coefficients = np.zeros(8)
    for _ in range(400):
        coefficients = np.maximum(coefficients - .005 * (gram @ coefficients - rhs), 0.)
    np.testing.assert_allclose(fit.coefficients, coefficients, rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize("kernel", ["l1", "gl"])
@pytest.mark.parametrize("method", ["legacy", "scaled_tail"])
def test_signal_dependent_kernel_residual_bound(kernel, method, enable_x64):
    n, alpha = 65, .6
    fit = _fit_soe(alpha, n, n_poles=8, t_max=1., kernel=kernel, method=method)
    t = np.linspace(0., 1., n)
    values = .2 + np.sin(20 * t) + (t > .3)
    actual = np.asarray(_apply_soe(jnp.asarray(values), fit))
    # For GL this is the original signed derivative convolution, independent of
    # the positive-prefix/increment rewrite used by the scaled-tail application.
    full = np.convolve(values, target_weights(alpha, n, kernel))[:n]
    residual = fitted_weights(fit) - target_weights(alpha, n, fit.kernel)
    signal = np.r_[values[0], np.diff(values)] if fit.kernel == "gl_prefix" else values
    bound = np.max(np.abs(residual)) * np.cumsum(np.abs(signal))
    assert np.all(np.abs(actual - full) <= bound + 3e-12)


@pytest.mark.parametrize("n", [256, 1024, 4096])
def test_static_poles_and_long_float32_eager_jit_consistency(n):
    alpha = jnp.asarray(.5, dtype=jnp.float32)

    def fit(a):
        result = _fit_soe(a, n - 1, n_poles=8, t_max=1., kernel="l1",
                          method="scaled_tail")
        return result.rhos, result.coefficients

    eager_rho, eager_c = fit(alpha)
    compiled_rho, compiled_c = jax.jit(fit)(alpha)
    np.testing.assert_array_equal(compiled_rho, eager_rho)
    np.testing.assert_allclose(compiled_c, eager_c, rtol=2e-4, atol=3e-6)
    t = jnp.linspace(0., 1., n, dtype=jnp.float32)
    values = jnp.sin(3 * t[:, None]) * jnp.arange(1, 5)[None, :]

    def run(x, a):
        return caputo(x, dt=1 / (n - 1), order=a, history="soe", soe_fit="scaled_tail")

    np.testing.assert_allclose(jax.jit(run)(values, alpha), run(values, alpha),
                               rtol=2e-5, atol=2e-5)
