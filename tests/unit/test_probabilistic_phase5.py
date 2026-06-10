from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import hpfracc as hp


def _mittag_leffler_two(alpha: float, beta: float, z: jax.Array, terms: int = 140):
    total = jnp.zeros_like(z)
    for k in range(terms):
        total = total + z**k / math.gamma(alpha * k + beta)
    return total


def _analytic_fsde_variance(
    final_time: float, alpha: float, rate: float, sigma: float, n: int = 4000
) -> float:
    # Var(y(T)) = sigma^2 * integral_0^T [tau^(alpha-1) E_{alpha,alpha}(rate tau^alpha)]^2 dtau.
    # The substitution tau = u^2 removes the mild tau^(2 alpha - 2) endpoint
    # singularity (integrable for alpha > 1/2), so a midpoint rule is accurate.
    root_t = final_time**0.5
    du = root_t / n
    u = (jnp.arange(n) + 0.5) * du
    tau = u**2
    kernel = tau ** (alpha - 1.0) * _mittag_leffler_two(alpha, alpha, rate * tau**alpha)
    integrand = (kernel**2) * 2.0 * u
    return float(sigma**2 * jnp.sum(integrand) * du)


def linear_dynamics(
    t: jax.Array,
    state: jax.Array,
    params: dict[str, jax.Array],
    *,
    rng_key: jax.Array | None = None,
    inputs: jax.Array | None = None,
) -> jax.Array:
    del t, rng_key, inputs
    return params["rate"] * state


def constant_diffusion(
    t: jax.Array,
    state: jax.Array,
    params: dict[str, jax.Array],
    *,
    rng_key: jax.Array | None = None,
    inputs: jax.Array | None = None,
) -> jax.Array:
    del t, state, rng_key, inputs
    return jnp.asarray(params["noise"])


def make_model() -> hp.nn.NeuralFODE:
    ts = jnp.linspace(0.0, 0.5, 11)
    solver = hp.solvers.PredictorCorrector(dt=float(ts[1] - ts[0]), order=0.7)
    return hp.nn.NeuralFODE(dynamics=linear_dynamics, solver=solver)


def test_stochastic_simulation_is_reproducible_for_same_key() -> None:
    ts = jnp.linspace(0.0, 0.5, 11)
    solver = hp.solvers.PredictorCorrector(dt=float(ts[1] - ts[0]), order=0.7)
    params = {"rate": jnp.asarray(-0.5), "noise": jnp.asarray(0.1)}
    key = jax.random.PRNGKey(0)

    first = hp.prob.simulate_stochastic(
        model=linear_dynamics,
        diffusion=constant_diffusion,
        ts=ts,
        solver=solver,
        initial_state=jnp.asarray(1.0),
        params=params,
        rng_key=key,
    )
    second = hp.prob.simulate_stochastic(
        model=linear_dynamics,
        diffusion=constant_diffusion,
        ts=ts,
        solver=solver,
        initial_state=jnp.asarray(1.0),
        params=params,
        rng_key=key,
    )

    assert jnp.allclose(first.latent_state, second.latent_state)
    assert first.solver_info is not None
    assert first.solver_info.warnings


def test_stochastic_simulation_changes_with_key() -> None:
    ts = jnp.linspace(0.0, 0.5, 11)
    solver = hp.solvers.PredictorCorrector(dt=float(ts[1] - ts[0]), order=0.7)
    params = {"rate": jnp.asarray(-0.5), "noise": jnp.asarray(0.1)}

    first = hp.prob.simulate_stochastic(
        model=linear_dynamics,
        diffusion=constant_diffusion,
        ts=ts,
        solver=solver,
        initial_state=jnp.asarray(1.0),
        params=params,
        rng_key=jax.random.PRNGKey(0),
    )
    second = hp.prob.simulate_stochastic(
        model=linear_dynamics,
        diffusion=constant_diffusion,
        ts=ts,
        solver=solver,
        initial_state=jnp.asarray(1.0),
        params=params,
        rng_key=jax.random.PRNGKey(1),
    )

    assert not jnp.allclose(first.latent_state, second.latent_state)


def test_stochastic_fsde_variance_matches_analytic() -> None:
    # The headline validity check: the simulated variance must track the analytic
    # variance of the linear additive FSDE. The old dt**0.5 scheme was wrong by
    # ~50x, so a generous tolerance still discriminates strongly.
    alpha, rate, sigma, y0, final_time = 0.8, -0.8, 0.5, 1.0, 1.0
    ts = jnp.linspace(0.0, final_time, 41)
    solver = hp.solvers.PredictorCorrector(dt=float(ts[1] - ts[0]), order=alpha)
    params = {"rate": jnp.asarray(rate), "noise": jnp.asarray(sigma)}

    def one(key: jax.Array) -> jax.Array:
        return hp.prob.simulate_stochastic(
            model=linear_dynamics,
            diffusion=constant_diffusion,
            ts=ts,
            solver=solver,
            initial_state=jnp.asarray(y0),
            params=params,
            rng_key=key,
        ).latent_state

    keys = jax.random.split(jax.random.PRNGKey(0), 4000)
    paths = jax.jit(jax.vmap(one))(keys)
    mc_variance = float(jnp.var(paths[:, -1]))
    analytic = _analytic_fsde_variance(final_time, alpha, rate, sigma)

    assert 0.8 < mc_variance / analytic < 1.3, (mc_variance, analytic)


def test_stochastic_fsde_mean_matches_deterministic_solution() -> None:
    # The noise is mean-zero, so the sample mean must reproduce the deterministic
    # Mittag-Leffler trajectory from the deterministic solver.
    alpha, rate, sigma, y0 = 0.8, -0.8, 0.5, 1.0
    ts = jnp.linspace(0.0, 1.0, 41)
    solver = hp.solvers.PredictorCorrector(dt=float(ts[1] - ts[0]), order=alpha)
    params = {"rate": jnp.asarray(rate), "noise": jnp.asarray(sigma)}

    def one(key: jax.Array) -> jax.Array:
        return hp.prob.simulate_stochastic(
            model=linear_dynamics,
            diffusion=constant_diffusion,
            ts=ts,
            solver=solver,
            initial_state=jnp.asarray(y0),
            params=params,
            rng_key=key,
        ).latent_state

    keys = jax.random.split(jax.random.PRNGKey(1), 4000)
    mc_mean_final = float(jnp.mean(jax.jit(jax.vmap(one))(keys)[:, -1]))

    deterministic = hp.solvers.simulate(
        model=linear_dynamics,
        ts=ts,
        solver=solver,
        initial_state=jnp.asarray(y0),
        params={"rate": jnp.asarray(rate)},
    ).latent_state
    assert abs(mc_mean_final - float(deterministic[-1])) < 0.03


@pytest.mark.parametrize("order", [0.5, 0.3])
def test_stochastic_fsde_rejects_order_at_or_below_half(order: float) -> None:
    ts = jnp.linspace(0.0, 1.0, 11)
    solver = hp.solvers.PredictorCorrector(dt=float(ts[1] - ts[0]), order=order)
    params = {"rate": jnp.asarray(-0.5), "noise": jnp.asarray(0.1)}
    with pytest.raises(ValueError, match="order > 0.5"):
        hp.prob.simulate_stochastic(
            model=linear_dynamics,
            diffusion=constant_diffusion,
            ts=ts,
            solver=solver,
            initial_state=jnp.asarray(1.0),
            params=params,
            rng_key=jax.random.PRNGKey(0),
        )


def test_gaussian_log_likelihood_prefers_closer_predictions() -> None:
    observed = jnp.asarray([1.0, 0.8, 0.6])
    close = jnp.asarray([1.0, 0.81, 0.59])
    far = jnp.asarray([1.3, 1.1, 0.9])

    assert hp.prob.gaussian_log_likelihood(
        close,
        observed,
        noise_scale=0.1,
    ) > hp.prob.gaussian_log_likelihood(far, observed, noise_scale=0.1)


def test_grid_calibration_recovers_nearby_scalar_parameter() -> None:
    model = make_model()
    ts = jnp.linspace(0.0, 0.5, 11)
    observed = model(
        ts=ts,
        initial_state=jnp.asarray(1.0),
        params={"rate": jnp.asarray(-0.8)},
    ).latent_state
    grid = jnp.asarray([-1.2, -0.8, -0.4, 0.0])

    result = hp.prob.grid_calibrate_scalar(
        model,
        ts=ts,
        initial_state=jnp.asarray(1.0),
        observations=observed,
        parameter_name="rate",
        parameter_grid=grid,
        noise_scale=0.05,
    )

    assert result.best_index == 1
    assert jnp.isclose(result.best_params["rate"], -0.8)
    assert jnp.isclose(jnp.sum(result.posterior_weights), 1.0)


def test_posterior_predictive_returns_weighted_summary() -> None:
    model = make_model()
    ts = jnp.linspace(0.0, 0.5, 11)
    grid = jnp.asarray([-1.2, -0.8, -0.4])
    weights = jnp.asarray([0.1, 0.8, 0.1])

    result = hp.prob.posterior_predictive(
        model,
        ts=ts,
        initial_state=jnp.asarray(1.0),
        parameter_name="rate",
        parameter_grid=grid,
        weights=weights,
    )

    assert result.trajectories.shape == (3, 11)
    assert result.mean.shape == (11,)
    assert result.lower.shape == (11,)
    assert result.upper.shape == (11,)
    assert jnp.isclose(jnp.sum(result.weights), 1.0)


def test_weighted_quantile_matches_known_discrete_cases() -> None:
    two = jnp.asarray([0.0, 10.0])
    # The weighted median follows the mass: 0.9 on the low point -> low point.
    assert float(hp.prob.weighted_quantile(two, jnp.asarray([0.9, 0.1]), 0.5)) == 0.0
    assert float(hp.prob.weighted_quantile(two, jnp.asarray([0.1, 0.9]), 0.5)) == 10.0

    four = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    uniform = jnp.ones(4)
    assert float(hp.prob.weighted_quantile(four, uniform, 0.05)) == 0.0
    assert float(hp.prob.weighted_quantile(four, uniform, 0.5)) == 1.0
    assert float(hp.prob.weighted_quantile(four, uniform, 0.95)) == 3.0


def test_weighted_quantile_is_vectorized_over_columns() -> None:
    # Leading axis is the weighted (grid) axis; each column resolved independently.
    stacked = jnp.asarray([[0.0, 0.0], [1.0, 5.0], [2.0, 9.0]])
    lower = hp.prob.weighted_quantile(stacked, jnp.ones(3), 0.05)
    upper = hp.prob.weighted_quantile(stacked, jnp.ones(3), 0.95)
    assert jnp.allclose(lower, jnp.asarray([0.0, 0.0]))
    assert jnp.allclose(upper, jnp.asarray([2.0, 9.0]))


def test_weighted_quantile_rejects_out_of_range_q() -> None:
    with pytest.raises(ValueError, match="quantile level"):
        hp.prob.weighted_quantile(jnp.asarray([0.0, 1.0]), jnp.ones(2), 1.5)


def _final_band(weights: jax.Array) -> tuple[float, float, float]:
    model = make_model()
    ts = jnp.linspace(0.0, 0.5, 11)
    grid = jnp.asarray([-1.2, -0.8, -0.4])
    result = hp.prob.posterior_predictive(
        model,
        ts=ts,
        initial_state=jnp.asarray(1.0),
        parameter_name="rate",
        parameter_grid=grid,
        weights=weights,
    )
    return float(result.lower[-1]), float(result.mean[-1]), float(result.upper[-1])


def test_posterior_predictive_interval_respects_weights() -> None:
    # Regression guard for the original bug, where lower/upper used an unweighted
    # quantile and were identical across different posteriors. The grid is
    # [-1.2, -0.8, -0.4]; faster decay (-1.2) gives the lowest final value.
    low_lower, _, low_upper = _final_band(jnp.asarray([0.98, 0.01, 0.01]))
    high_lower, _, high_upper = _final_band(jnp.asarray([0.01, 0.01, 0.98]))

    # Different posteriors must yield different bands (the bug made them equal).
    assert not jnp.isclose(low_lower, high_lower)
    assert not jnp.isclose(low_upper, high_upper)
    # A posterior concentrated on fast decay sits entirely below one concentrated
    # on slow decay.
    assert low_upper < high_lower


def test_posterior_predictive_band_stays_within_trajectory_support() -> None:
    model = make_model()
    ts = jnp.linspace(0.0, 0.5, 11)
    grid = jnp.asarray([-1.2, -0.8, -0.4])
    weights = jnp.asarray([0.2, 0.5, 0.3])
    result = hp.prob.posterior_predictive(
        model,
        ts=ts,
        initial_state=jnp.asarray(1.0),
        parameter_name="rate",
        parameter_grid=grid,
        weights=weights,
    )
    per_time_min = jnp.min(result.trajectories, axis=0)
    per_time_max = jnp.max(result.trajectories, axis=0)
    assert jnp.all(result.lower >= per_time_min - 1e-6)
    assert jnp.all(result.upper <= per_time_max + 1e-6)
    assert jnp.all(result.upper >= result.lower)


def test_posterior_predictive_concentrated_weight_collapses_band() -> None:
    # 90% interval of a posterior with ~all mass on one grid point is that point.
    lower, _, upper = _final_band(jnp.asarray([0.001, 0.998, 0.001]))
    assert jnp.isclose(lower, upper, atol=1e-6)


def test_posterior_predictive_uniform_weight_brackets_mean() -> None:
    # For a symmetric uniform grid weighting the mean lies inside the band.
    lower, mean, upper = _final_band(jnp.asarray([1.0, 1.0, 1.0]))
    assert lower <= mean <= upper
