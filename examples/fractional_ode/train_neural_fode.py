"""Recover a scalar fractional ODE rate parameter from synthetic data."""

from __future__ import annotations

import jax
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

    target = model(
        ts=ts,
        initial_state=jnp.asarray(1.0),
        params={"rate": jnp.asarray(-0.8)},
    ).latent_state

    def loss(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        return hp.nn.trajectory_mse(
            model,
            ts=ts,
            initial_state=jnp.asarray(1.0),
            params=params,
            target=target,
        )

    params = {"rate": jnp.asarray(0.1)}
    for _ in range(25):
        grads = jax.grad(loss)(params)
        params = hp.nn.sgd_step(params, grads, learning_rate=0.5)

    print({"rate": float(params["rate"]), "loss": float(loss(params))})


if __name__ == "__main__":
    main()
