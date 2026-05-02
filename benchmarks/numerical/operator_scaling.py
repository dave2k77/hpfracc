"""Small operator runtime scaling benchmark.

This script records baseline wall-clock timings for the explicit full-history
operator implementations. It is intentionally lightweight so it can be run on a
developer CPU without special hardware.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from hpfracc.ops import caputo, grunwald_letnikov


@dataclass(frozen=True, slots=True)
class ScalingRow:
    """One row in the operator scaling report."""

    operator: str
    n_steps: int
    state_dim: int
    order: float
    seconds: float
    output_shape: tuple[int, ...]


def benchmark_operator(
    fn: Callable[..., Any],
    *,
    operator: str,
    n_steps: int,
    state_dim: int,
    order: float,
    repeats: int,
) -> ScalingRow:
    """Time one operator/shape combination."""

    dt = 1.0 / float(n_steps - 1)
    t = jnp.linspace(0.0, 1.0, n_steps)
    x = jnp.stack([(idx + 1) * t**2 for idx in range(state_dim)], axis=-1)

    # Warm up JAX dispatch before timing.
    warmed = fn(x, dt=dt, order=order)
    jax.block_until_ready(warmed)

    start = time.perf_counter()
    for _ in range(repeats):
        y = fn(x, dt=dt, order=order)
        jax.block_until_ready(y)
    elapsed = (time.perf_counter() - start) / float(repeats)

    return ScalingRow(
        operator=operator,
        n_steps=n_steps,
        state_dim=state_dim,
        order=order,
        seconds=elapsed,
        output_shape=tuple(int(dim) for dim in warmed.shape),
    )


def generate_rows(
    *,
    n_steps_values: Sequence[int] = (32, 64, 128),
    state_dims: Sequence[int] = (1, 4),
    order: float = 0.5,
    repeats: int = 3,
) -> list[ScalingRow]:
    """Generate scaling rows for v0.1 baseline operators."""

    rows: list[ScalingRow] = []
    operators: tuple[tuple[str, Callable[..., Any]], ...] = (
        ("caputo", caputo),
        ("grunwald_letnikov", grunwald_letnikov),
    )
    for operator, fn in operators:
        for n_steps in n_steps_values:
            for state_dim in state_dims:
                rows.append(
                    benchmark_operator(
                        fn,
                        operator=operator,
                        n_steps=n_steps,
                        state_dim=state_dim,
                        order=order,
                        repeats=repeats,
                    )
                )
    return rows


def write_csv(rows: Sequence[ScalingRow], stream: object = sys.stdout) -> None:
    """Write scaling rows as CSV."""

    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "operator",
            "n_steps",
            "state_dim",
            "order",
            "seconds",
            "output_shape",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "operator": row.operator,
                "n_steps": row.n_steps,
                "state_dim": row.state_dim,
                "order": row.order,
                "seconds": row.seconds,
                "output_shape": "x".join(str(dim) for dim in row.output_shape),
            }
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-steps", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--state-dims", type=int, nargs="+", default=[1, 4])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = generate_rows(
        n_steps_values=tuple(args.n_steps),
        state_dims=tuple(args.state_dims),
        order=args.order,
        repeats=args.repeats,
    )
    write_csv(rows)


if __name__ == "__main__":
    main()

