"""Minimal Caputo operator example."""

from __future__ import annotations

import jax.numpy as jnp

import hpfracc as hp


def main() -> None:
    dt = 0.01
    t = jnp.arange(101) * dt
    x = t**2
    result = hp.ops.caputo(x, dt=dt, order=0.5, return_info=True)

    print(result.values[-1])
    print(result.operator_info.to_dict())


if __name__ == "__main__":
    main()

