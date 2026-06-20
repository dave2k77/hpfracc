"""Validation for the a-posteriori adaptive-step Caputo solver (WS-2 part 2b).

The adaptive solver chooses its own mesh from a **step-doubling (Richardson)**
local-error estimate -- one step of ``h`` against two of ``h/2`` -- and accepts /
shrinks / grows the step to track ``rtol``/``atol``. The load-bearing properties:

* **tolerance control** -- the error against a trusted reference decreases as the
  tolerance is tightened;
* **earns its keep** -- on a problem with a weak ``t**alpha`` origin singularity it
  clusters nodes near ``t=0`` and reaches an accuracy the uniform fixed-step solver
  cannot reach even with many more nodes (the uniform global error stalls);
* **clean differentiable contract** -- advancing with the single full step (no local
  extrapolation) makes the realized trajectory *identical* to the non-uniform solver
  run on the realized mesh, so the gradient is the exact frozen-mesh sensitivity and
  is finite-difference checkable; the mesh / controller themselves are not
  differentiated (the diffrax discretize-then-optimize pattern); and the autodiff
  graph is bounded by ``max_steps`` independently of the problem.
"""

from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import hpfracc as hp


def linear_model(t, state, params, *, rng_key=None, inputs=None):
    del t, rng_key, inputs
    return params * state


def mittag_leffler(alpha: float, z, *, terms: int = 160):
    total = jnp.zeros_like(z)
    for k in range(terms):
        total = total + (z**k) / math.gamma(alpha * k + 1.0)
    return total


def _adaptive(order, *, rtol, atol, max_steps=4096, first_step=None):
    return hp.solvers.AdaptivePredictorCorrector(
        order=order,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        first_step=first_step,
    )


def _run(solver, rate, *, t_end=1.0, y0=None):
    if y0 is None:
        y0 = jnp.asarray(1.0)
    return hp.solvers.simulate(
        model=linear_model,
        ts=jnp.asarray([0.0, t_end]),
        solver=solver,
        initial_state=y0,
        params=jnp.asarray(rate),
    )


def _nodes(result):
    """The realized (un-padded) nodes / states up to the accepted count."""
    n = result.solver_info.diagnostics["n_nodes"]
    return result.ts[:n], result.latent_state[:n]


def test_tolerance_control_decreases_error(enable_x64) -> None:
    order, rate = 0.7, -0.8
    prev = None
    for rtol in (1e-3, 1e-4, 1e-5):
        result = _run(_adaptive(order, rtol=rtol, atol=rtol * 1e-2), rate)
        t, y = _nodes(result)
        expected = mittag_leffler(order, rate * t**order)
        err = float(jnp.max(jnp.abs(y - expected)))
        if prev is not None:
            assert err < prev
        prev = err


def test_smooth_problem_matches_reference(enable_x64) -> None:
    order, rate = 0.6, -0.5
    result = _run(_adaptive(order, rtol=1e-6, atol=1e-8, max_steps=4096), rate)
    t, y = _nodes(result)
    expected = mittag_leffler(order, rate * t**order)
    assert float(jnp.max(jnp.abs(y - expected))) < 1e-3
    assert result.solver_info.diagnostics["reached_t_end"] is True


def test_earns_its_keep_on_origin_singularity(enable_x64) -> None:
    # Small alpha => strong t**alpha origin singularity. The uniform fixed-step
    # solver's global error stalls (it cannot resolve the origin layer); the
    # adaptive solver clusters nodes near t=0 and reaches an accuracy the uniform
    # grid does not, with far fewer nodes.
    order, rate = 0.4, -1.0
    adaptive = _run(
        _adaptive(order, rtol=3e-5, atol=3e-7, max_steps=4096, first_step=1e-3),
        rate,
    )
    t, y = _nodes(adaptive)
    n_adaptive = int(t.shape[0])
    expected = mittag_leffler(order, rate * t**order)
    adaptive_err = float(jnp.max(jnp.abs(y - expected)))

    # Nodes cluster near the origin: the smallest steps are at the start.
    spacings = jnp.diff(t)
    assert int(jnp.argmin(spacings)) < n_adaptive // 4
    assert float(spacings[0]) < 0.1 * float(jnp.max(spacings))

    # A uniform grid with several times as many nodes still cannot match it.
    n_uniform = 4 * n_adaptive
    tu = jnp.linspace(0.0, 1.0, n_uniform)
    uniform = hp.solvers.simulate(
        model=linear_model,
        ts=tu,
        solver=hp.solvers.PredictorCorrector(dt=float(tu[1] - tu[0]), order=order),
        initial_state=jnp.asarray(1.0),
        params=jnp.asarray(rate),
    )
    uniform_err = float(
        jnp.max(jnp.abs(uniform.latent_state - mittag_leffler(order, rate * tu**order)))
    )
    assert adaptive_err < uniform_err


def test_gradient_matches_nonuniform_solver_on_realized_mesh(enable_x64) -> None:
    # No local extrapolation => the adaptive trajectory is *exactly* the non-uniform
    # solver on the realized mesh, so its frozen-mesh gradient equals that solver's
    # gradient there, which in turn matches finite differences (fixed mesh).
    order, base_rate = 0.6, -0.8

    def adaptive_obj(y0, rate):
        return _run(
            _adaptive(order, rtol=1e-4, atol=1e-6, max_steps=512), rate, y0=y0
        ).latent_state[-1].sum()

    mesh, _ = _nodes(
        _run(_adaptive(order, rtol=1e-4, atol=1e-6, max_steps=512), base_rate)
    )

    def nonuniform_obj(y0, rate):
        return hp.solvers.simulate(
            model=linear_model,
            ts=mesh,
            solver=hp.solvers.NonUniformPredictorCorrector(order=order),
            initial_state=y0,
            params=rate,
        ).latent_state[-1].sum()

    y0 = jnp.asarray(1.0)
    rate = jnp.asarray(base_rate)
    grad_adaptive = jax.grad(adaptive_obj, argnums=1)(y0, rate)
    grad_nonuniform = jax.grad(nonuniform_obj, argnums=1)(y0, rate)
    assert jnp.isfinite(grad_adaptive)
    # The two solvers compute the identical trajectory on this mesh.
    assert jnp.allclose(grad_adaptive, grad_nonuniform, rtol=1e-7, atol=1e-10)

    eps = 1e-6
    fd = (nonuniform_obj(y0, rate + eps) - nonuniform_obj(y0, rate - eps)) / (2 * eps)
    assert jnp.allclose(grad_nonuniform, fd, rtol=1e-4, atol=1e-6)


def test_adaptive_is_jit_consistent(enable_x64) -> None:
    order = 0.6
    solver = _adaptive(order, rtol=1e-4, atol=1e-6, max_steps=256)

    def run(y0):
        return _run(solver, -0.5, y0=y0).latent_state

    assert jnp.allclose(jax.jit(run)(jnp.asarray(1.0)), run(jnp.asarray(1.0)))


def test_graph_size_is_independent_of_problem() -> None:
    # The bounded masked scan depends only on max_steps, not on the dynamics.
    def make(rate: float) -> int:
        solver = _adaptive(0.6, rtol=1e-4, atol=1e-6, max_steps=64)

        def run(y0):
            return _run(solver, rate, y0=y0).latent_state

        return len(jax.make_jaxpr(run)(jnp.asarray(1.0)).jaxpr.eqns)

    assert make(-0.5) == make(-5.0)


def test_preserves_trailing_state_shape(enable_x64) -> None:
    solver = _adaptive(0.6, rtol=1e-3, atol=1e-5, max_steps=128)
    result = _run(solver, -0.5, y0=jnp.ones((2, 3)))
    n = result.solver_info.diagnostics["n_nodes"]
    assert result.latent_state.shape == (128 + 2, 2, 3)
    assert result.latent_state[:n].shape == (n, 2, 3)
    assert result.solver_info.method == "caputo_adaptive_pece"
    assert result.solver_info.diagnostics["grid"] == "adaptive"
    assert result.solver_info.diagnostics["estimator"] == "step_doubling"
    assert result.solver_info.step_size is None


def test_monotone_padded_grid(enable_x64) -> None:
    solver = _adaptive(0.6, rtol=1e-3, atol=1e-5, max_steps=128)
    result = _run(solver, -0.5)
    # Padded tail keeps ts monotone non-decreasing and ends at t_end.
    assert bool(jnp.all(jnp.diff(result.ts) >= -1e-12))
    assert float(result.ts[-1]) == pytest.approx(1.0, abs=1e-9)


def test_too_small_cap_does_not_crash_and_reports_not_reached(enable_x64) -> None:
    # A cap too small to reach t_end must not error; it reports reached_t_end False.
    solver = _adaptive(0.4, rtol=1e-8, atol=1e-10, max_steps=3, first_step=1e-4)
    result = _run(solver, -1.0)
    assert result.solver_info.diagnostics["reached_t_end"] is False
    assert jnp.all(jnp.isfinite(result.latent_state))


def test_invalid_configuration_raises() -> None:
    with pytest.raises(ValueError, match="rtol"):
        hp.solvers.AdaptivePredictorCorrector(order=0.5, rtol=0.0)
    with pytest.raises(ValueError, match="max_steps"):
        hp.solvers.AdaptivePredictorCorrector(order=0.5, max_steps=1)
    with pytest.raises(ValueError, match="first_step"):
        hp.solvers.AdaptivePredictorCorrector(order=0.5, first_step=-1.0)
    with pytest.raises(ValueError, match="safety"):
        hp.solvers.AdaptivePredictorCorrector(order=0.5, safety=2.0)
    with pytest.raises(ValueError, match="min_factor"):
        hp.solvers.AdaptivePredictorCorrector(order=0.5, min_factor=1.5)
