"""Generate Phase 2 Caputo FDE solver validation tables."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import gamma

import jax.numpy as jnp

from hpfracc.solvers import PredictorCorrector, simulate


@dataclass(frozen=True, slots=True)
class SolverValidationRow:
    """One row in the Caputo FDE solver validation report."""

    solver: str
    case: str
    order: float
    rate: float
    n_steps: int
    dt: float
    max_abs_error: float
    reference: str


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


def mittag_leffler(alpha: float, z: object, *, terms: int = 80) -> object:
    """Approximate ``E_alpha(z)`` with a truncated power series."""

    values = jnp.asarray(z)
    total = jnp.zeros_like(values)
    for k in range(terms):
        total = total + (values**k) / gamma(alpha * k + 1.0)
    return total


def linear_fde_rows(
    *,
    order: float = 0.7,
    rate: float = -0.8,
    n_steps_values: Sequence[int] = (21, 41, 81),
) -> list[SolverValidationRow]:
    """Compute scalar linear FDE errors against a Mittag-Leffler reference."""

    rows: list[SolverValidationRow] = []
    for n_steps in n_steps_values:
        ts = jnp.linspace(0.0, 1.0, n_steps)
        dt = float(ts[1] - ts[0])
        solver = PredictorCorrector(dt=dt, order=order)
        result = simulate(
            model=linear_model,
            ts=ts,
            solver=solver,
            initial_state=jnp.asarray(1.0),
            params=jnp.asarray(rate),
        )
        expected = mittag_leffler(order, rate * ts**order)
        error = float(jnp.max(jnp.abs(result.latent_state - expected)))
        rows.append(
            SolverValidationRow(
                solver="predictor_corrector",
                case="linear_caputo_fde",
                order=order,
                rate=rate,
                n_steps=n_steps,
                dt=dt,
                max_abs_error=error,
                reference="truncated_mittag_leffler",
            )
        )
    return rows


def generate_rows(
    *,
    order: float = 0.7,
    rate: float = -0.8,
    n_steps_values: Sequence[int] = (21, 41, 81),
) -> list[SolverValidationRow]:
    """Generate the complete Phase 2 solver validation table."""

    return linear_fde_rows(
        order=order,
        rate=rate,
        n_steps_values=n_steps_values,
    )


def write_csv(rows: Iterable[SolverValidationRow], stream: object = sys.stdout) -> None:
    """Write validation rows as CSV."""

    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "solver",
            "case",
            "order",
            "rate",
            "n_steps",
            "dt",
            "max_abs_error",
            "reference",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "solver": row.solver,
                "case": row.case,
                "order": row.order,
                "rate": row.rate,
                "n_steps": row.n_steps,
                "dt": row.dt,
                "max_abs_error": row.max_abs_error,
                "reference": row.reference,
            }
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=float, default=0.7)
    parser.add_argument("--rate", type=float, default=-0.8)
    parser.add_argument(
        "--n-steps",
        type=int,
        nargs="+",
        default=[21, 41, 81],
        help="Time-grid sizes to evaluate.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = generate_rows(
        order=args.order,
        rate=args.rate,
        n_steps_values=tuple(args.n_steps),
    )
    write_csv(rows)


if __name__ == "__main__":
    main()
