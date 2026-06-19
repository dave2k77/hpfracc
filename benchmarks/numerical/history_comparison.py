"""Benchmark comparing full-history and FFT-accelerated operator kernels.

This script measures wall-clock time for the same operator call under the
``history="full"`` and ``history="fft"`` strategies.  It is intended as a
validation artifact that proves the FFT path is available and that its runtime
cost is sub-quadratic, not as a publication-grade benchmark.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from hpfracc.ops import caputo, grunwald_letnikov


@dataclass(frozen=True, slots=True)
class HistoryComparisonRow:
    """One full-vs-FFT benchmark row."""

    operator: str
    n_steps: int
    state_dim: int
    order: float
    history: str
    repeats: int
    seconds: float
    backend: str


def _benchmark_operator(
    fn: object,
    *,
    operator: str,
    n_steps: int,
    state_dim: int,
    order: float,
    history: str,
    repeats: int,
) -> HistoryComparisonRow:
    dt = 1.0 / float(n_steps - 1)
    t = jnp.linspace(0.0, 1.0, n_steps)
    x = jnp.stack([(idx + 1) * t**2 for idx in range(state_dim)], axis=-1)

    warmed = fn(x, dt=dt, order=order, history=history)
    jax.block_until_ready(warmed)

    start = time.perf_counter()
    for _ in range(repeats):
        y = fn(x, dt=dt, order=order, history=history)
        jax.block_until_ready(y)
    seconds = (time.perf_counter() - start) / float(repeats)

    return HistoryComparisonRow(
        operator=operator,
        n_steps=n_steps,
        state_dim=state_dim,
        order=order,
        history=history,
        repeats=repeats,
        seconds=seconds,
        backend=jax.default_backend(),
    )


def generate_rows(
    *,
    n_steps_values: Sequence[int] = (64, 128, 256, 512),
    state_dims: Sequence[int] = (1, 4),
    order: float = 0.7,
    repeats: int = 3,
) -> list[HistoryComparisonRow]:
    """Generate full-vs-FFT comparison rows for Caputo and GL operators."""

    rows: list[HistoryComparisonRow] = []
    operators: tuple[tuple[str, object], ...] = (
        ("caputo", caputo),
        ("grunwald_letnikov", grunwald_letnikov),
    )
    for operator, fn in operators:
        for n_steps in n_steps_values:
            for state_dim in state_dims:
                for history in ("full", "fft", "short_memory", "soe"):
                    rows.append(
                        _benchmark_operator(
                            fn,
                            operator=operator,
                            n_steps=n_steps,
                            state_dim=state_dim,
                            order=order,
                            history=history,
                            repeats=repeats,
                        )
                    )
    return rows


def write_csv(
    rows: Sequence[HistoryComparisonRow], stream: object = sys.stdout
) -> None:
    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "operator",
            "n_steps",
            "state_dim",
            "order",
            "history",
            "repeats",
            "seconds",
            "backend",
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
                "history": row.history,
                "repeats": row.repeats,
                "seconds": row.seconds,
                "backend": row.backend,
            }
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=float, default=0.7)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-steps", type=int, nargs="+", default=[64, 128, 256, 512])
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
