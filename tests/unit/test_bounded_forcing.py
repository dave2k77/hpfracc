"""Conditional full-history forcing bounds, compatibility, and JAX contracts."""

import json
from math import gamma

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

import hpfracc as hp


def quadratic(t, params, *, rng_key=None, inputs=None):
    return params * t*t


def cubic(t, params, *, rng_key=None, inputs=None):
    return params * (t*t - 2*t**3/3)


def solve(*, forcing=quadratic, curvature=2., order=.5, dtype=jnp.float64,
          atol=1e-4, rtol=0., first=1., budget=256, initial=0., parameter=1.,
          times=None):
    return hp.solvers.simulate(
        model=hp.solvers.BoundedForcing(forcing, curvature),
        ts=jnp.asarray([0., 1.], dtype=dtype) if times is None else times,
        solver=hp.solvers.AdaptiveForcingQuadrature(
            order=jnp.asarray(order, dtype=dtype), atol=atol, rtol=rtol,
            first_step=first, max_steps=budget,
        ), initial_state=jnp.asarray(initial, dtype=dtype),
        params=jnp.asarray(parameter, dtype=dtype),
    )


def independent_envelope(times, alpha, curvature):
    """70-digit integration of the supplied per-panel interpolation majorant."""
    with mp.workdps(70):
        a, bound = mp.mpf(float(alpha)), mp.mpf(curvature)
        nodes = [mp.mpf(float(t)) for t in times]
        total, latest = [], []
        for n, target in enumerate(nodes):
            terms = []
            for j in range(n):
                width = nodes[j+1] - nodes[j]
                mass = ((target-nodes[j])**a - (target-nodes[j+1])**a) / mp.gamma(a+1)
                terms.append(bound * width**2 * mass / 8)
            total.append(float(mp.fsum(terms)))
            latest.append(float(terms[-1]) if terms else 0.)
        return np.asarray(total), np.asarray(latest)


@pytest.mark.parametrize("alpha", [.3, .5, .8])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_global_envelope_encloses_quadratic_and_refines(alpha, dtype, enable_x64):
    errors = []
    for tolerance in (1e-3, 1e-4):
        result = solve(order=alpha, dtype=dtype, atol=tolerance)
        d = result.solver_info.diagnostics
        assert d["success"] and d["n_rejected"] > 0
        n = d["n_nodes"]
        t = np.asarray(result.ts[:n], dtype=np.float64)
        actual = np.asarray(result.latent_state[:n], dtype=np.float64)
        a = float(result.solver_info.fractional_order)
        truth = 2*t**(a+2)/gamma(a+3)
        error = np.abs(actual-truth)
        bound = np.asarray(d["interpolation_error_bound"][:n])
        slack = 2e-6 if dtype == jnp.float32 else 2e-12
        assert np.all(error <= bound + slack)
        assert np.all(bound <= tolerance * (1 + 2e-6))
        assert result.latent_state.dtype == result.ts.dtype == dtype
        errors.append(float(np.max(error)))
    assert errors[1] < errors[0]


def test_bound_accounts_for_old_history_not_only_latest_panel(enable_x64):
    result = solve()
    d = result.solver_info.diagnostics
    n = d["n_nodes"]
    t = np.asarray(result.ts[:n])
    expected, local = independent_envelope(t, .5, 2.)
    np.testing.assert_allclose(d["interpolation_error_bound"][:n], expected,
                               rtol=2e-12, atol=1e-14)
    actual_error = abs(float(result.latent_state[-1]) - 2/gamma(3.5))
    assert local[-1] < actual_error <= expected[-1]
    assert expected[-1] > 3*local[-1]


def test_cubic_pair_blind_spot_rejected_and_controlled(enable_x64):
    result = solve(forcing=cubic, atol=1e-4)
    d = result.solver_info.diagnostics
    assert d["success"] and d["n_rejected"] > 0
    n = d["n_nodes"]
    t = np.asarray(result.ts[:n])
    expected = 2*t**2.5/gamma(3.5) - 4*t**3.5/gamma(4.5)
    error = np.abs(np.asarray(result.latent_state[:n])-expected)
    assert np.max(error) < 1e-4
    assert np.all(error <= np.asarray(d["interpolation_error_bound"][:n]) + 2e-12)


def test_componentwise_curvature_affine_control_and_relative_scale(enable_x64):
    def forcing(t, params, *, rng_key=None, inputs=None):
        return jnp.stack((1+2*t, params*t*t))

    result = solve(forcing=forcing, curvature=(0., 2.), initial=(1., .5),
                   atol=1e-5, rtol=1e-3)
    d = result.solver_info.diagnostics
    assert d["success"]
    n = d["n_nodes"]
    t = np.asarray(result.ts[:n])
    truth = np.stack((1+t**.5/gamma(1.5)+2*t**1.5/gamma(2.5),
                      .5+2*t**2.5/gamma(3.5)), axis=-1)
    error = np.abs(np.asarray(result.latent_state[:n])-truth)
    envelope = np.asarray(d["interpolation_error_bound"][:n])
    assert result.latent_state.shape == (258, 2)
    np.testing.assert_array_equal(envelope[:, 0], 0.)
    assert np.all(error <= envelope + 3e-12)
    assert np.all(error / (1e-5+1e-3*np.abs(truth)) < 1)
    json.dumps(result.solver_info.to_dict(), allow_nan=False)


def test_nonuniform_consistency_and_frozen_mesh_gradients(enable_x64):
    model = hp.solvers.BoundedForcing(quadratic, 2.)

    def adaptive(y0, parameter, order):
        result = hp.solvers.simulate(
            model=model, ts=jnp.asarray([0., 1.]), initial_state=y0, params=parameter,
            solver=hp.solvers.AdaptiveForcingQuadrature(
                order=order, atol=1e-3, first_step=1., max_steps=128,
            ),
        )
        d = result.solver_info.diagnostics
        return result.latent_state, result.ts, d["success"], d["n_nodes"]

    args = (jnp.asarray(.2), jnp.asarray(.7), jnp.asarray(.5))
    values, times, success, n = adaptive(*args)
    compiled = jax.jit(adaptive)(*args)
    assert success and compiled[2]
    np.testing.assert_allclose(compiled[0], values, rtol=1e-11, atol=1e-12)
    mesh = times[:int(n)]

    def frozen(y0, parameter, order):
        return hp.solvers.simulate(
            model=model, ts=mesh, initial_state=y0, params=parameter,
            solver=hp.solvers.NonUniformPredictorCorrector(order=order),
        ).latent_state

    np.testing.assert_allclose(values[:int(n)], frozen(*args), rtol=1e-12, atol=1e-12)
    gradient = jax.jit(jax.grad(lambda y, p, a: adaptive(y, p, a)[0][-1],
                               argnums=(0, 1, 2)))(*args)
    np.testing.assert_allclose(gradient[0], 1., atol=1e-12)
    for index in range(3):
        plus, minus = list(args), list(args)
        plus[index], minus[index] = args[index]+1e-5, args[index]-1e-5
        fd = (frozen(*plus)[-1] - frozen(*minus)[-1]) / 2e-5
        np.testing.assert_allclose(gradient[index], fd, rtol=1e-7, atol=1e-8)


def test_tensor_state_shifted_origin_and_explicit_context(enable_x64):
    amplitudes = jnp.arange(1., 7.).reshape(2, 3) / 6
    key = jax.random.key(7)
    inputs = {"origin": 2., "offset": jnp.asarray([.1, .2, .3])}

    def forcing(t, params, *, rng_key=None, inputs=None):
        # The same explicit key gives a time-independent offset, so q remains C2.
        return (params["amplitude"] * (t-inputs["origin"])**2
                + inputs["offset"] + jax.random.uniform(rng_key, ()))

    model = hp.solvers.BoundedForcing(forcing, 2*np.asarray(amplitudes))

    @jax.jit
    def run(params):
        result = hp.solvers.simulate(
            model=model, ts=jnp.asarray([2., 3.]), initial_state=jnp.ones((2, 3)),
            params=params, rng_key=key, inputs=inputs,
            solver=hp.solvers.AdaptiveForcingQuadrature(
                order=.5, atol=1e-3, first_step=1., max_steps=128,
            ),
        )
        d = result.solver_info.diagnostics
        return (result.ts, result.latent_state, d["interpolation_error_bound"],
                d["n_nodes"], d["success"])

    times, values, envelope, n, success = run({"amplitude": amplitudes})
    assert success and values.shape == envelope.shape == (130, 2, 3)
    lag = np.asarray(times[:int(n)])[:, None, None] - 2
    offset = np.asarray(inputs["offset"]) + float(jax.random.uniform(key, ()))
    truth = (1 + offset*lag**.5/gamma(1.5)
             + np.asarray(amplitudes)*2*lag**2.5/gamma(3.5))
    error = np.abs(np.asarray(values[:int(n)]) - truth)
    assert np.all(error <= np.asarray(envelope[:int(n)]) + 3e-12)
    assert np.max(error) < 1e-3


def test_budget_failure_padding_and_last_allowed_slot(enable_x64):
    failed = solve(atol=1e-8, budget=1)
    d = failed.solver_info.diagnostics
    assert not d["success"] and d["max_steps_reached"]
    assert d["n_rejected"] == d["n_attempted"] == 1 and d["n_accepted"] == 0
    np.testing.assert_array_equal(d["valid_mask"], [True, False, False])
    np.testing.assert_array_equal(failed.latent_state, 0.)

    def constant(t, params, *, rng_key=None, inputs=None):
        return params

    complete = solve(forcing=constant, curvature=0., budget=1)
    d = complete.solver_info.diagnostics
    assert d["success"] and not d["max_steps_reached"]
    np.testing.assert_array_equal(d["valid_mask"], [True, True, False])
    np.testing.assert_allclose(complete.latent_state[-1], 1/gamma(1.5), atol=1e-12)


def test_nonfinite_trial_preserves_prefix_under_jit(enable_x64):
    def bad(t, params, *, rng_key=None, inputs=None):
        return jnp.where(t > .2, jnp.nan, params)

    def run(parameter):
        result = solve(forcing=bad, curvature=0., first=.1, parameter=parameter)
        d = result.solver_info.diagnostics
        return result.ts, result.latent_state, d["nonfinite_detected"], d["n_nodes"]

    times, states, bad_flag, n = jax.jit(run)(jnp.asarray(1.))
    assert bad_flag and int(n) == 2
    np.testing.assert_array_equal(times[1:], .1)
    assert np.all(np.isfinite(states))
    np.testing.assert_array_equal(states[2:], states[1])


def test_singleton_and_time_resolution():
    singleton = solve(dtype=jnp.float32, times=jnp.asarray([2.]), initial=1.)
    d = singleton.solver_info.diagnostics
    assert d["success"] and d["n_attempted"] == 0
    start = jnp.asarray(1., dtype=jnp.float32)
    end = jnp.nextafter(start, jnp.asarray(2., dtype=jnp.float32))
    failed = solve(dtype=jnp.float32, times=jnp.stack((start, end)), first=1e-20)
    assert failed.solver_info.diagnostics["step_size_underflow"]
    assert failed.solver_info.diagnostics["n_accepted"] == 0


@pytest.mark.parametrize("bound", [-1., np.nan, np.inf, 1j, [], (1., -2.)])
def test_invalid_curvature_rejected(bound):
    with pytest.raises(ValueError, match="curvature_bound"):
        hp.solvers.BoundedForcing(quadratic, bound)


def test_plain_dynamics_integer_inputs_and_unrepresentable_bounds_rejected():
    solver = hp.solvers.AdaptiveForcingQuadrature(order=.5)
    with pytest.raises(TypeError, match="BoundedForcing"):
        hp.solvers.simulate(model=lambda t,y,p,**kw:p*y, ts=jnp.asarray([0., 1.]),
                            initial_state=jnp.asarray(1.), params=1., solver=solver)
    with pytest.raises(ValueError, match="float32/float64"):
        solve(dtype=jnp.float32, times=jnp.asarray([0, 1]))
    with pytest.raises(ValueError, match="representable"):
        solve(dtype=jnp.float32, curvature=1e100)
    with pytest.raises(ValueError, match="broadcast"):
        solve(dtype=jnp.float32, curvature=(1., 2.), initial=(0., 0., 0.))


def test_finite_curvature_is_an_assumption_not_automatically_verified(enable_x64):
    # Deliberately false M=0 for q=t^2: demonstrate why the condition matters.
    result = solve(curvature=0., atol=1e-8, budget=1)
    assert result.solver_info.diagnostics["success"]
    np.testing.assert_array_equal(result.solver_info.diagnostics[
        "interpolation_error_bound"], 0.)
    assert abs(float(result.latent_state[-1])-2/gamma(3.5)) > .01


def test_roundoff_is_explicitly_outside_the_bound():
    def constant(t, params, *, rng_key=None, inputs=None):
        return params

    result = solve(dtype=jnp.float32, forcing=constant, curvature=0.,
                   atol=1e-12, budget=1)
    d = result.solver_info.diagnostics
    assert d["success"] and d["roundoff_included"] is False
    assert d["bound_scope"] == "conditional_nodal_interpolation_error"
    assert abs(float(result.latent_state[-1])-1/gamma(1.5)) > 1e-12


def test_graph_size_is_bounded_by_scan_not_python_unrolling():
    def size(budget):
        return len(jax.make_jaxpr(lambda p: solve(
            dtype=jnp.float32, budget=budget, parameter=p,
        ).latent_state)(jnp.asarray(.7, dtype=jnp.float32)).jaxpr.eqns)
    assert size(16) == size(256)


@pytest.mark.parametrize("kwargs", [
    {"atol": 0.}, {"rtol": -1.}, {"max_steps": 0}, {"max_steps": 1.5},
    {"safety": 0.}, {"min_factor": 1.}, {"max_factor": 1.}, {"first_step": -1.},
])
def test_invalid_solver_configuration(kwargs):
    with pytest.raises(ValueError):
        hp.solvers.AdaptiveForcingQuadrature(order=.5, **kwargs)
