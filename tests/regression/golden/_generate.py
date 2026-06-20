"""Regenerate the committed operator golden artifact.

Run from the repo root with::

    uv run python tests/regression/golden/_generate.py

Regenerate (and review the diff) only when an operator's numerics intentionally
change. An unexplained diff here is a regression, not a reason to regenerate.
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from hpfracc.ops import caputo, grunwald_letnikov

# Fixed, deterministic case. Keep these in sync with test_golden_operators.py.
ALPHA = 0.5
DT = 0.05
N = 32
GOLDEN = Path(__file__).with_name("operator_outputs.csv")


def _signal() -> jnp.ndarray:
    ts = jnp.arange(N, dtype=jnp.float64) * DT
    return ts**2


def main() -> None:
    x = _signal()
    cap = caputo(x, dt=DT, order=ALPHA)
    gl = grunwald_letnikov(x, dt=DT, order=ALPHA)
    with GOLDEN.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "caputo", "grunwald_letnikov"])
        for i in range(N):
            writer.writerow([i, f"{float(cap[i]):.17e}", f"{float(gl[i]):.17e}"])
    print(f"wrote {GOLDEN}")


if __name__ == "__main__":
    main()
