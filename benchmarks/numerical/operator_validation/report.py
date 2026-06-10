"""Generate Phase 1 fractional-operator validation tables.

The report is intentionally small and deterministic. It is a validation
artifact, not a performance benchmark.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import jax.numpy as jnp

from hpfracc.ops import (
    caputo,
    caputo_power_law,
    riemann_liouville,
    riemann_liouville_power_law,
)


@dataclass(frozen=True, slots=True)
class ValidationRow:
    """One row in the operator validation report."""

    operator: str
    case: str
    order: float
    power: float | None
    n_steps: int
    dt: float
    max_abs_error: float
    reference: str


def caputo_power_law_rows(
    *,
    order: float = 0.5,
    power: float = 2.0,
    n_steps_values: Sequence[int] = (21, 41, 81),
) -> list[ValidationRow]:
    """Compute Caputo L1 errors against an analytic power-law reference."""

    rows: list[ValidationRow] = []
    for n_steps in n_steps_values:
        t = jnp.linspace(0.0, 1.0, n_steps)
        dt = float(t[1] - t[0])
        x = t**power
        actual = caputo(x, dt=dt, order=order)
        expected = caputo_power_law(t, power=power, order=order)
        error = float(jnp.max(jnp.abs(actual[1:] - expected[1:])))
        rows.append(
            ValidationRow(
                operator="caputo",
                case="power_law",
                order=order,
                power=power,
                n_steps=n_steps,
                dt=dt,
                max_abs_error=error,
                reference="analytic_caputo_power_law",
            )
        )
    return rows


def caputo_constant_rows(
    *,
    order: float = 0.5,
    n_steps_values: Sequence[int] = (21, 41, 81),
) -> list[ValidationRow]:
    """Compute Caputo L1 errors for the derivative of a constant."""

    rows: list[ValidationRow] = []
    for n_steps in n_steps_values:
        dt = 1.0 / float(n_steps - 1)
        x = jnp.ones((n_steps,))
        actual = caputo(x, dt=dt, order=order)
        error = float(jnp.max(jnp.abs(actual)))
        rows.append(
            ValidationRow(
                operator="caputo",
                case="constant",
                order=order,
                power=0.0,
                n_steps=n_steps,
                dt=dt,
                max_abs_error=error,
                reference="zero",
            )
        )
    return rows


def riemann_liouville_power_law_rows(
    *,
    order: float = 0.5,
    power: float = 2.0,
    n_steps_values: Sequence[int] = (21, 41, 81),
) -> list[ValidationRow]:
    """Compute RL errors against the analytic ``t**power`` reference.

    This validates the RL derivative against analytic ground truth rather than
    against the identical GL code path, so the error is a genuine discretisation
    error (not a tautological zero).
    """

    rows: list[ValidationRow] = []
    for n_steps in n_steps_values:
        t = jnp.linspace(0.0, 1.0, n_steps)
        dt = float(t[1] - t[0])
        x = t**power
        actual = riemann_liouville(x, dt=dt, order=order)
        expected = riemann_liouville_power_law(t, power=power, order=order)
        error = float(jnp.max(jnp.abs(actual[1:] - expected[1:])))
        rows.append(
            ValidationRow(
                operator="riemann_liouville",
                case="rl_power_law",
                order=order,
                power=power,
                n_steps=n_steps,
                dt=dt,
                max_abs_error=error,
                reference="analytic_riemann_liouville_power_law",
            )
        )
    return rows


def riemann_liouville_constant_rows(
    *,
    order: float = 0.5,
    n_steps_values: Sequence[int] = (21, 41, 81),
) -> list[ValidationRow]:
    """Compute RL errors for a constant against ``t**(-alpha) / Gamma(1 - alpha)``.

    The decisive RL-vs-Caputo case: the RL derivative of a constant is nonzero.
    The reference is singular at ``t = 0``, so the error is measured over the
    interior region ``t >= 0.5`` where the GL discretisation is in its smooth,
    convergent regime.
    """

    rows: list[ValidationRow] = []
    for n_steps in n_steps_values:
        t = jnp.linspace(0.0, 1.0, n_steps)
        dt = float(t[1] - t[0])
        actual = riemann_liouville(jnp.ones((n_steps,)), dt=dt, order=order)
        expected = riemann_liouville_power_law(t, power=0.0, order=order)
        interior = t >= 0.5
        error = float(jnp.max(jnp.abs((actual - expected)[interior])))
        rows.append(
            ValidationRow(
                operator="riemann_liouville",
                case="rl_constant",
                order=order,
                power=0.0,
                n_steps=n_steps,
                dt=dt,
                max_abs_error=error,
                reference="analytic_riemann_liouville_constant",
            )
        )
    return rows


def generate_rows(
    *,
    order: float = 0.5,
    power: float = 2.0,
    n_steps_values: Sequence[int] = (21, 41, 81),
) -> list[ValidationRow]:
    """Generate the complete Phase 1 operator validation table."""

    return [
        *caputo_constant_rows(order=order, n_steps_values=n_steps_values),
        *caputo_power_law_rows(
            order=order,
            power=power,
            n_steps_values=n_steps_values,
        ),
        *riemann_liouville_power_law_rows(
            order=order,
            power=power,
            n_steps_values=n_steps_values,
        ),
        *riemann_liouville_constant_rows(order=order, n_steps_values=n_steps_values),
    ]


def write_csv(rows: Iterable[ValidationRow], stream: object = sys.stdout) -> None:
    """Write validation rows as CSV."""

    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "operator",
            "case",
            "order",
            "power",
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
                "operator": row.operator,
                "case": row.case,
                "order": row.order,
                "power": "" if row.power is None else row.power,
                "n_steps": row.n_steps,
                "dt": row.dt,
                "max_abs_error": row.max_abs_error,
                "reference": row.reference,
            }
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=float, default=0.5)
    parser.add_argument("--power", type=float, default=2.0)
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
        power=args.power,
        n_steps_values=tuple(args.n_steps),
    )
    write_csv(rows)


if __name__ == "__main__":
    main()

