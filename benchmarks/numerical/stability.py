"""Small deterministic stability checks for Phase 3 validation."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import jax.numpy as jnp

from hpfracc.ops import caputo
from hpfracc.solvers import PredictorCorrector, simulate


@dataclass(frozen=True, slots=True)
class StabilityRow:
    """One stability check row."""

    target: str
    case: str
    order: float
    n_steps: int
    metric: str
    value: float
    passed: bool


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


def caputo_constant_stability_row(
    *,
    order: float = 0.5,
    n_steps: int = 64,
) -> StabilityRow:
    """Check that the Caputo derivative of a constant remains zero."""

    values = jnp.ones((n_steps,))
    actual = caputo(values, dt=1.0 / float(n_steps - 1), order=order)
    metric_value = float(jnp.max(jnp.abs(actual)))
    return StabilityRow(
        target="caputo_operator",
        case="constant_zero_response",
        order=order,
        n_steps=n_steps,
        metric="max_abs_value",
        value=metric_value,
        passed=metric_value <= 1e-7,
    )


def linear_decay_solver_row(
    *,
    order: float = 0.7,
    n_steps: int = 81,
    rate: float = -0.8,
) -> StabilityRow:
    """Check that scalar linear decay remains finite and non-amplifying."""

    ts = jnp.linspace(0.0, 1.0, n_steps)
    solver = PredictorCorrector(dt=float(ts[1] - ts[0]), order=order)
    result = simulate(
        model=linear_model,
        ts=ts,
        solver=solver,
        initial_state=jnp.asarray(1.0),
        params=jnp.asarray(rate),
    )
    max_abs = float(jnp.max(jnp.abs(result.latent_state)))
    is_finite = bool(jnp.all(jnp.isfinite(result.latent_state)))
    return StabilityRow(
        target="caputo_solver",
        case="linear_decay_non_amplifying",
        order=order,
        n_steps=n_steps,
        metric="max_abs_state",
        value=max_abs,
        passed=is_finite and max_abs <= 1.0 + 1e-5,
    )


def generate_rows(
    *,
    order: float = 0.7,
    n_steps: int = 81,
) -> list[StabilityRow]:
    """Generate all stability check rows."""

    return [
        caputo_constant_stability_row(order=order, n_steps=n_steps),
        linear_decay_solver_row(order=order, n_steps=n_steps),
    ]


def write_csv(rows: Iterable[StabilityRow], stream: object = sys.stdout) -> None:
    """Write stability check rows as CSV."""

    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "target",
            "case",
            "order",
            "n_steps",
            "metric",
            "value",
            "passed",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "target": row.target,
                "case": row.case,
                "order": row.order,
                "n_steps": row.n_steps,
                "metric": row.metric,
                "value": row.value,
                "passed": row.passed,
            }
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=float, default=0.7)
    parser.add_argument("--n-steps", type=int, default=81)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    write_csv(generate_rows(order=args.order, n_steps=args.n_steps))


if __name__ == "__main__":
    main()
