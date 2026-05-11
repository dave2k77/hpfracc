"""Minimal Caputo FDE solver example."""

from __future__ import annotations

import jax.numpy as jnp

import hpfracc as hp


def linear_model(
    t: jnp.ndarray,
    state: jnp.ndarray,
    params: jnp.ndarray,
    *,
    rng_key: jnp.ndarray | None = None,
    inputs: jnp.ndarray | None = None,
) -> jnp.ndarray:
    del t, rng_key, inputs
    return params * state


def main() -> None:
    dt = 0.01
    ts = jnp.arange(101) * dt
    solver = hp.solvers.PredictorCorrector(dt=dt, order=0.7)
    result = hp.solvers.simulate(
        model=linear_model,
        ts=ts,
        solver=solver,
        initial_state=jnp.asarray(1.0),
        params=jnp.asarray(-0.8),
    )

    print(result.latent_state[-1])
    print(result.solver_info.to_dict())


if __name__ == "__main__":
    main()
