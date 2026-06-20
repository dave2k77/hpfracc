"""Validation for the implicit Caputo solver (WS-2, part 1).

The implicit solver solves the fractional Adams-Moulton corrector equation with a
fixed-iteration Newton step instead of the explicit one-pass PECE correction. The
load-bearing property is **stability**: on a stiff linear problem where the
explicit predictor-corrector diverges, the implicit method stays bounded and
tracks the decaying solution. Accuracy in the moderately-stiff regime is checked
against a fine-grid self-refinement reference (the truncated Mittag-Leffler series
is unreliable for large negative arguments), and the mild-rate accuracy against
the analytic Mittag-Leffler solution.
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


def mittag_leffler(alpha: float, z, *, terms: int = 100):
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


def _implicit(dt, order, **kw):
    return hp.solvers.ImplicitPredictorCorrector(dt=dt, order=order, **kw)


def _explicit(dt, order):
    return hp.solvers.PredictorCorrector(dt=dt, order=order)


def test_implicit_matches_mittag_leffler_and_refines(enable_x64) -> None:
    order, rate = 0.7, -0.8
    errors = []
    for n in (21, 81):
        ts = jnp.linspace(0.0, 1.0, n)
        result = _run(_implicit(float(ts[1] - ts[0]), order), ts, rate)
        expected = mittag_leffler(order, rate * ts**order)
        errors.append(float(jnp.max(jnp.abs(result.latent_state - expected))))
    assert errors[0] < 5e-3
    assert errors[1] < errors[0]


def test_implicit_is_stable_where_explicit_diverges(enable_x64) -> None:
    # Stiff linear decay on a coarse grid. The true solution decays monotonically
    # from y0 = 1, so a stable solver never exceeds ~1; the explicit PECE leaves
    # its (finite) stability region and blows up.
    order, rate = 0.7, -30.0
    ts = jnp.linspace(0.0, 1.0, 30)
    dt = float(ts[1] - ts[0])
    explicit = _run(_explicit(dt, order), ts, rate).latent_state
    implicit = _run(_implicit(dt, order, newton_iterations=5), ts, rate).latent_state

    assert float(jnp.max(jnp.abs(implicit))) < 1.0 + 1e-2  # bounded, tracks decay
    assert float(jnp.max(jnp.abs(explicit))) > 1e2  # explicit diverges
    assert jnp.all(jnp.isfinite(implicit))


def test_implicit_is_more_accurate_when_moderately_stiff(enable_x64) -> None:
    # Reference: same implicit scheme on a very fine grid (self-refinement), since
    # the truncated Mittag-Leffler series is unreliable at this argument size.
    order, rate = 0.7, -12.0
    fine_ts = jnp.linspace(0.0, 1.0, 2000)
    truth_end = float(
        _run(
            _implicit(float(fine_ts[1] - fine_ts[0]), order, newton_iterations=6),
            fine_ts,
            rate,
        ).latent_state[-1]
    )

    ts = jnp.linspace(0.0, 1.0, 25)
    dt = float(ts[1] - ts[0])
    explicit_end = float(_run(_explicit(dt, order), ts, rate).latent_state[-1])
    implicit_end = float(
        _run(_implicit(dt, order, newton_iterations=5), ts, rate).latent_state[-1]
    )
    assert abs(implicit_end - truth_end) < abs(explicit_end - truth_end)


def test_newton_iterations_irrelevant_for_linear_dynamics(enable_x64) -> None:
    # The corrector is linear in y for f = lambda*y, so one Newton step is exact.
    order, rate = 0.6, -2.0
    ts = jnp.linspace(0.0, 1.0, 20)
    dt = float(ts[1] - ts[0])
    one = _run(_implicit(dt, order, newton_iterations=1), ts, rate).latent_state
    five = _run(_implicit(dt, order, newton_iterations=5), ts, rate).latent_state
    assert jnp.allclose(one, five, atol=1e-12)


def test_implicit_graph_size_is_independent_of_n() -> None:
    def make(n: int) -> int:
        ts = jnp.linspace(0.0, 1.0, n)
        solver = _implicit(float(ts[1] - ts[0]), 0.7)

        def run(y0):
            return _run(solver, ts, -0.8, y0=y0).latent_state

        return len(jax.make_jaxpr(run)(jnp.asarray(1.0)).jaxpr.eqns)

    assert make(16) == make(256)


def test_implicit_is_jit_consistent(enable_x64) -> None:
    order = 0.6
    ts = jnp.linspace(0.0, 0.5, 11)
    solver = _implicit(float(ts[1] - ts[0]), order)

    def run(y0):
        return _run(solver, ts, -0.25, y0=y0).latent_state

    assert jnp.allclose(jax.jit(run)(jnp.asarray(1.0)), run(jnp.asarray(1.0)))


def test_implicit_supports_state_and_parameter_gradients(enable_x64) -> None:
    ts = jnp.linspace(0.0, 0.5, 11)
    solver = _implicit(float(ts[1] - ts[0]), 0.6)

    def objective(initial_state, rate):
        return _run(solver, ts, rate, y0=initial_state).latent_state[-1].sum()

    grad_y0, grad_rate = jax.grad(objective, argnums=(0, 1))(
        jnp.asarray(1.0), jnp.asarray(-0.25)
    )
    assert jnp.isfinite(grad_y0) and jnp.isfinite(grad_rate)

    eps = 1e-6
    fd = (
        objective(jnp.asarray(1.0), jnp.asarray(-0.25 + eps))
        - objective(jnp.asarray(1.0), jnp.asarray(-0.25 - eps))
    ) / (2.0 * eps)
    assert jnp.allclose(grad_rate, fd, rtol=1e-4, atol=1e-6)


def test_implicit_preserves_trailing_state_shape(enable_x64) -> None:
    solver = _implicit(0.1, 0.6)
    ts = jnp.arange(6) * 0.1
    result = _run(solver, ts, -0.5, y0=jnp.ones((2, 3)))
    assert result.latent_state.shape == (6, 2, 3)
    assert result.solver_info is not None
    assert result.solver_info.method == "caputo_implicit_adams_moulton"
    assert result.solver_info.diagnostics["scheme"] == "implicit"


def test_implicit_rejects_nonuniform_grid() -> None:
    solver = _implicit(0.1, 0.7)
    ts = jnp.asarray([0.0, 0.1, 0.25])
    with pytest.raises(ValueError, match="uniform time grid"):
        _run(solver, ts, -1.0)


def test_implicit_config_rejects_nonpositive_newton_iterations() -> None:
    with pytest.raises(ValueError, match="newton_iterations"):
        hp.solvers.ImplicitPredictorCorrector(dt=0.1, order=0.5, newton_iterations=0)


def test_implicit_agrees_with_explicit_when_non_stiff(enable_x64) -> None:
    order, rate = 0.6, -0.5
    ts = jnp.linspace(0.0, 1.0, 200)
    dt = float(ts[1] - ts[0])
    explicit = _run(_explicit(dt, order), ts, rate).latent_state
    implicit = _run(_implicit(dt, order), ts, rate).latent_state
    assert jnp.allclose(explicit, implicit, atol=1e-3)
