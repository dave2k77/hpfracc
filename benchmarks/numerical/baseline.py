"""CPU-oriented baseline benchmark rows for Phase 3."""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from hpfracc.ops import caputo, grunwald_letnikov
from hpfracc.solvers import PredictorCorrector, simulate


@dataclass(frozen=True, slots=True)
class BaselineRow:
    """One baseline benchmark row."""

    target: str
    case: str
    n_steps: int
    state_dim: int
    order: float
    repeats: int
    seconds: float
    backend: str
    platform: str


def linear_model(
    t: object,
    state: object,
    params: object,
    *,
    rng_key: object | None = None,
    inputs: object | None = None,
) -> object:
    del t, rng_key, inputs
    return params * state


def benchmark_operator(
    fn: object,
    *,
    target: str,
    n_steps: int,
    state_dim: int,
    order: float,
    repeats: int,
) -> BaselineRow:
    """Benchmark one operator on a deterministic input."""

    dt = 1.0 / float(n_steps - 1)
    t = jnp.linspace(0.0, 1.0, n_steps)
    x = jnp.stack([(idx + 1) * t**2 for idx in range(state_dim)], axis=-1)
    warmed = fn(x, dt=dt, order=order)
    jax.block_until_ready(warmed)

    start = time.perf_counter()
    for _ in range(repeats):
        y = fn(x, dt=dt, order=order)
        jax.block_until_ready(y)
    seconds = (time.perf_counter() - start) / float(repeats)
    return _baseline_row(
        target=target,
        case="full_history_operator",
        n_steps=n_steps,
        state_dim=state_dim,
        order=order,
        repeats=repeats,
        seconds=seconds,
    )


def benchmark_solver(
    *,
    n_steps: int,
    state_dim: int,
    order: float,
    repeats: int,
) -> BaselineRow:
    """Benchmark the fixed-step Caputo predictor-corrector solver."""

    ts = jnp.linspace(0.0, 1.0, n_steps)
    solver = PredictorCorrector(dt=float(ts[1] - ts[0]), order=order)
    y0 = jnp.ones((state_dim,))

    warmed = simulate(
        model=linear_model,
        ts=ts,
        solver=solver,
        initial_state=y0,
        params=jnp.asarray(-0.8),
    ).latent_state
    jax.block_until_ready(warmed)

    start = time.perf_counter()
    for _ in range(repeats):
        y = simulate(
            model=linear_model,
            ts=ts,
            solver=solver,
            initial_state=y0,
            params=jnp.asarray(-0.8),
        ).latent_state
        jax.block_until_ready(y)
    seconds = (time.perf_counter() - start) / float(repeats)
    return _baseline_row(
        target="predictor_corrector",
        case="linear_caputo_fde",
        n_steps=n_steps,
        state_dim=state_dim,
        order=order,
        repeats=repeats,
        seconds=seconds,
    )


def generate_rows(
    *,
    n_steps_values: Sequence[int] = (32, 64),
    state_dims: Sequence[int] = (1, 4),
    order: float = 0.7,
    repeats: int = 3,
) -> list[BaselineRow]:
    """Generate CPU-oriented baseline benchmark rows."""

    rows: list[BaselineRow] = []
    operators: tuple[tuple[str, Any], ...] = (
        ("caputo", caputo),
        ("grunwald_letnikov", grunwald_letnikov),
    )
    for n_steps in n_steps_values:
        for state_dim in state_dims:
            for target, fn in operators:
                rows.append(
                    benchmark_operator(
                        fn,
                        target=target,
                        n_steps=n_steps,
                        state_dim=state_dim,
                        order=order,
                        repeats=repeats,
                    )
                )
            rows.append(
                benchmark_solver(
                    n_steps=n_steps,
                    state_dim=state_dim,
                    order=order,
                    repeats=repeats,
                )
            )
    return rows


def write_csv(rows: Iterable[BaselineRow], stream: object = sys.stdout) -> None:
    """Write baseline benchmark rows as CSV."""

    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "target",
            "case",
            "n_steps",
            "state_dim",
            "order",
            "repeats",
            "seconds",
            "backend",
            "platform",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "target": row.target,
                "case": row.case,
                "n_steps": row.n_steps,
                "state_dim": row.state_dim,
                "order": row.order,
                "repeats": row.repeats,
                "seconds": row.seconds,
                "backend": row.backend,
                "platform": row.platform,
            }
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=float, default=0.7)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-steps", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--state-dims", type=int, nargs="+", default=[1, 4])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    write_csv(
        generate_rows(
            n_steps_values=tuple(args.n_steps),
            state_dims=tuple(args.state_dims),
            order=args.order,
            repeats=args.repeats,
        )
    )


def _baseline_row(
    *,
    target: str,
    case: str,
    n_steps: int,
    state_dim: int,
    order: float,
    repeats: int,
    seconds: float,
) -> BaselineRow:
    return BaselineRow(
        target=target,
        case=case,
        n_steps=n_steps,
        state_dim=state_dim,
        order=order,
        repeats=repeats,
        seconds=seconds,
        backend=jax.default_backend(),
        platform=platform.platform(),
    )


if __name__ == "__main__":
    main()
