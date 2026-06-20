"""Validation for the non-uniform / graded-mesh Caputo solver (WS-2 part 2a).

The non-uniform solver uses the variable-step fractional Adams-Bashforth-Moulton
weights built from the actual node spacings. The load-bearing properties are:

* on an equispaced grid it reduces **exactly** to the uniform ``PredictorCorrector``
  (the validated fixed-step weights), and
* on a mesh graded toward ``t=0`` it restores the convergence the uniform grid
  loses to the solution's weak ``t**alpha`` origin singularity -- the graded error
  keeps decreasing under refinement while the uniform global error stalls, and the
  graded mesh beats a uniform mesh of equal node count once the grid is fine enough.
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


def mittag_leffler(alpha: float, z, *, terms: int = 140):
    total = jnp.zeros_like(z)
    for k in range(terms):
        total = total + (z**k) / math.gamma(alpha * k + 1.0)
    return total


def _run(solver, ts, rate, y0=None):
    if y0 is None:
        y0 = jnp.asarray(1.0)
    return hp.solvers.simulate(
        model=linear_model,
        ts=ts,
        solver=solver,
        initial_state=y0,
        params=jnp.asarray(rate),
    )


def _graded(n: int, rate: float, t_end: float = 1.0):
    # Use the active default float dtype (float64 under the enable_x64 fixture,
    # float32 otherwise) so the graph-size test does not request an unavailable dtype.
    return t_end * (jnp.arange(n) / (n - 1)) ** rate


def test_nonuniform_reduces_to_uniform(enable_x64) -> None:
    order, rate = 0.7, -0.8
    ts = jnp.linspace(0.0, 1.0, 30)
    nonuniform = _run(hp.solvers.NonUniformPredictorCorrector(order=order), ts, rate)
    uniform = _run(
        hp.solvers.PredictorCorrector(dt=float(ts[1] - ts[0]), order=order), ts, rate
    )
    assert jnp.allclose(
        nonuniform.latent_state, uniform.latent_state, atol=1e-12
    )


def test_graded_mesh_matches_mittag_leffler_and_refines(enable_x64) -> None:
    order, rate = 0.7, -0.8
    errors = []
    for n in (40, 80):
        t = _graded(n, rate=2.0)
        result = _run(hp.solvers.NonUniformPredictorCorrector(order=order), t, rate)
        expected = mittag_leffler(order, rate * t**order)
        errors.append(float(jnp.max(jnp.abs(result.latent_state - expected))))
    assert errors[0] < 5e-3
    assert errors[1] < errors[0]


def test_graded_mesh_restores_order_lost_on_uniform_grid(enable_x64) -> None:
    # Small alpha => stronger t**alpha origin singularity. Grading r = (2-a)/a is the
    # standard choice. The graded error converges cleanly while the uniform global
    # max-norm error stalls; graded wins outright once the grid is fine enough.
    order, rate = 0.4, -1.0
    grade = (2.0 - order) / order
    graded_errors = []
    uniform_finest = None
    graded_finest = None
    for n in (40, 80, 160):
        tg = _graded(n, rate=grade)
        eg = float(
            jnp.max(
                jnp.abs(
                    _run(hp.solvers.NonUniformPredictorCorrector(order=order), tg, rate)
                    .latent_state
                    - mittag_leffler(order, rate * tg**order)
                )
            )
        )
        graded_errors.append(eg)
        tu = jnp.linspace(0.0, 1.0, n)
        eu = float(
            jnp.max(
                jnp.abs(
                    _run(
                        hp.solvers.PredictorCorrector(
                            dt=float(tu[1] - tu[0]), order=order
                        ),
                        tu,
                        rate,
                    ).latent_state
                    - mittag_leffler(order, rate * tu**order)
                )
            )
        )
        uniform_finest, graded_finest = eu, eg
    # Graded mesh converges monotonically (restored order)...
    assert graded_errors[0] > graded_errors[1] > graded_errors[2]
    # ...and beats the uniform grid of equal node count at the finest level.
    assert graded_finest < uniform_finest


def test_nonuniform_supports_state_and_parameter_gradients(enable_x64) -> None:
    order = 0.6
    t = _graded(20, rate=2.0)

    def objective(initial_state, rate):
        return _run(
            hp.solvers.NonUniformPredictorCorrector(order=order),
            t,
            rate,
            y0=initial_state,
        ).latent_state[-1].sum()

    grad_y0, grad_rate = jax.grad(objective, argnums=(0, 1))(
        jnp.asarray(1.0), jnp.asarray(-0.8)
    )
    assert jnp.isfinite(grad_y0) and jnp.isfinite(grad_rate)
    eps = 1e-6
    fd = (
        objective(jnp.asarray(1.0), jnp.asarray(-0.8 + eps))
        - objective(jnp.asarray(1.0), jnp.asarray(-0.8 - eps))
    ) / (2.0 * eps)
    assert jnp.allclose(grad_rate, fd, rtol=1e-4, atol=1e-6)


def test_nonuniform_is_jit_consistent(enable_x64) -> None:
    order = 0.6
    t = _graded(12, rate=2.0)
    solver = hp.solvers.NonUniformPredictorCorrector(order=order)

    def run(y0):
        return _run(solver, t, -0.5, y0=y0).latent_state

    assert jnp.allclose(jax.jit(run)(jnp.asarray(1.0)), run(jnp.asarray(1.0)))


def test_nonuniform_graph_size_is_independent_of_n() -> None:
    def make(n: int) -> int:
        t = _graded(n, rate=2.0)
        solver = hp.solvers.NonUniformPredictorCorrector(order=0.7)

        def run(y0):
            return _run(solver, t, -0.8, y0=y0).latent_state

        return len(jax.make_jaxpr(run)(jnp.asarray(1.0)).jaxpr.eqns)

    assert make(16) == make(256)


def test_nonuniform_preserves_trailing_state_shape(enable_x64) -> None:
    solver = hp.solvers.NonUniformPredictorCorrector(order=0.6)
    t = jnp.array([0.0, 0.1, 0.25, 0.45, 0.7, 1.0])
    result = _run(solver, t, -0.5, y0=jnp.ones((2, 3)))
    assert result.latent_state.shape == (6, 2, 3)
    assert result.solver_info is not None
    assert result.solver_info.method == "caputo_pece_nonuniform"
    assert result.solver_info.diagnostics["grid"] == "nonuniform"
    assert result.solver_info.step_size is None


def test_nonuniform_rejects_non_increasing_grid() -> None:
    solver = hp.solvers.NonUniformPredictorCorrector(order=0.7)
    ts = jnp.array([0.0, 0.2, 0.2, 0.5])
    with pytest.raises(ValueError, match="strictly increasing"):
        _run(solver, ts, -1.0)
