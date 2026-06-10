"""Aggregate Phase 3 numerical validation summary."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from benchmarks.numerical.convergence import (
    caputo_operator_order_row,
    row_passed,
    solver_endpoint_order_row,
)
from benchmarks.numerical.gradient_checks import generate_rows as gradient_rows
from benchmarks.numerical.operator_validation.report import (
    generate_rows as operator_rows,
)
from benchmarks.numerical.solver_validation.report import generate_rows as solver_rows
from benchmarks.numerical.stability import generate_rows as stability_rows


@dataclass(frozen=True, slots=True)
class ValidationSummaryRow:
    """One aggregate validation summary row."""

    area: str
    case: str
    metric: str
    value: float
    passed: bool
    details: str


def generate_rows(
    *,
    order: float = 0.7,
    n_steps_values: Sequence[int] = (21, 41, 81),
    gradient_tolerance: float = 2e-3,
) -> list[ValidationSummaryRow]:
    """Generate repeatable aggregate validation summary rows."""

    op_rows = operator_rows(order=order, power=2.0, n_steps_values=n_steps_values)
    power_law = [row for row in op_rows if row.case == "power_law"]
    caputo_constant = [row for row in op_rows if row.case == "constant"]

    sol_rows = solver_rows(order=order, rate=-0.8, n_steps_values=n_steps_values)
    grad_rows = gradient_rows()
    stable_rows = stability_rows(order=order, n_steps=n_steps_values[-1])

    # Convergence-order checks use their own dedicated refinement grids (not the
    # summary ``n_steps_values``, which are tuned for the other checks) so the
    # log-log slope is estimated in an asymptotic, float64-clean regime.
    operator_order = caputo_operator_order_row(
        alpha=order, n_steps_values=(41, 81, 161)
    )
    solver_order = solver_endpoint_order_row(
        alpha=order, n_steps_values=(21, 41, 81, 161)
    )

    return [
        ValidationSummaryRow(
            area="operator",
            case="caputo_constant",
            metric="max_abs_error",
            value=max(row.max_abs_error for row in caputo_constant),
            passed=all(row.max_abs_error <= 1e-7 for row in caputo_constant),
            details=f"n_steps={_join(row.n_steps for row in caputo_constant)}",
        ),
        ValidationSummaryRow(
            area="operator",
            case="caputo_power_law_refinement",
            metric="last_max_abs_error",
            value=power_law[-1].max_abs_error,
            passed=power_law[-1].max_abs_error < power_law[0].max_abs_error,
            details=(
                f"first={power_law[0].max_abs_error}; "
                f"last={power_law[-1].max_abs_error}"
            ),
        ),
        ValidationSummaryRow(
            area="solver",
            case="linear_caputo_fde_refinement",
            metric="last_max_abs_error",
            value=sol_rows[-1].max_abs_error,
            passed=sol_rows[-1].max_abs_error < sol_rows[0].max_abs_error,
            details=(
                f"first={sol_rows[0].max_abs_error}; "
                f"last={sol_rows[-1].max_abs_error}"
            ),
        ),
        ValidationSummaryRow(
            area="convergence",
            case="caputo_operator_order",
            metric="estimated_order",
            value=operator_order.estimated_order,
            passed=row_passed(operator_order),
            details=(
                f"expected~={operator_order.expected_order:.3f}; "
                f"alpha={order}; two_sided"
            ),
        ),
        ValidationSummaryRow(
            area="convergence",
            case="solver_endpoint_order",
            metric="estimated_order",
            value=solver_order.estimated_order,
            passed=row_passed(solver_order),
            details=(
                f"expected>={solver_order.expected_order:.3f}; "
                f"alpha={order}; lower_bound"
            ),
        ),
        ValidationSummaryRow(
            area="gradient",
            case="finite_difference_checks",
            metric="max_abs_error",
            value=max(row.abs_error for row in grad_rows),
            passed=all(row.abs_error <= gradient_tolerance for row in grad_rows),
            details=f"tolerance={gradient_tolerance}",
        ),
        ValidationSummaryRow(
            area="stability",
            case="basic_stability_checks",
            metric="passed_count",
            value=float(sum(row.passed for row in stable_rows)),
            passed=all(row.passed for row in stable_rows),
            details=f"total={len(stable_rows)}",
        ),
    ]


def write_csv(
    rows: Iterable[ValidationSummaryRow],
    stream: object = sys.stdout,
) -> None:
    """Write aggregate validation summary rows as CSV."""

    writer = csv.DictWriter(
        stream,
        fieldnames=["area", "case", "metric", "value", "passed", "details"],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "area": row.area,
                "case": row.case,
                "metric": row.metric,
                "value": row.value,
                "passed": row.passed,
                "details": row.details,
            }
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=float, default=0.7)
    parser.add_argument("--n-steps", type=int, nargs="+", default=[21, 41, 81])
    parser.add_argument("--gradient-tolerance", type=float, default=2e-3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    write_csv(
        generate_rows(
            order=args.order,
            n_steps_values=tuple(args.n_steps),
            gradient_tolerance=args.gradient_tolerance,
        )
    )


def _join(values: Iterable[int]) -> str:
    return "|".join(str(value) for value in values)


if __name__ == "__main__":
    main()
