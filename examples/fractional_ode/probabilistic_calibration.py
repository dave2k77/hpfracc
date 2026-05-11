"""Small scalar-grid probabilistic calibration example."""

from __future__ import annotations

import jax.numpy as jnp

import hpfracc as hp


def linear_dynamics(
    t: jnp.ndarray,
    state: jnp.ndarray,
    params: dict[str, jnp.ndarray],
    *,
    rng_key: jnp.ndarray | None = None,
    inputs: jnp.ndarray | None = None,
) -> jnp.ndarray:
    del t, rng_key, inputs
    return params["rate"] * state


def main() -> None:
    ts = jnp.linspace(0.0, 0.5, 11)
    solver = hp.solvers.PredictorCorrector(dt=float(ts[1] - ts[0]), order=0.7)
    model = hp.nn.NeuralFODE(dynamics=linear_dynamics, solver=solver)

    observations = model(
        ts=ts,
        initial_state=jnp.asarray(1.0),
        params={"rate": jnp.asarray(-0.8)},
    ).latent_state

    grid = jnp.linspace(-1.2, -0.2, 11)
    calibration = hp.prob.grid_calibrate_scalar(
        model,
        ts=ts,
        initial_state=jnp.asarray(1.0),
        observations=observations,
        parameter_name="rate",
        parameter_grid=grid,
        noise_scale=0.05,
    )
    predictive = hp.prob.posterior_predictive(
        model,
        ts=ts,
        initial_state=jnp.asarray(1.0),
        parameter_name="rate",
        parameter_grid=calibration.parameter_grid,
        weights=calibration.posterior_weights,
    )

    print(
        {
            "best_rate": float(calibration.best_params["rate"]),
            "posterior_mean_final": float(predictive.mean[-1]),
        }
    )


if __name__ == "__main__":
    main()
