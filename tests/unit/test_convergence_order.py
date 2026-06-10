"""Observed-convergence-order tests for the v0.1 operator and solver.

These tests upgrade the earlier "error decreased under refinement" checks into
assertions that the error decreases *at the theoretically required rate*:

- Caputo L1 operator: max-norm order ``2 - alpha``.
- Caputo PECE solver: endpoint order ``1 + alpha``.

Tolerance bands are grounded in float64 measurements on the exact grids used
here (see ``benchmarks/numerical/convergence.py``). The operator order sits
just below ``2 - alpha`` and converges up to it, so it is checked two-sided.
The solver order sits at or slightly above ``1 + alpha`` (a benign
super-convergence for the smooth linear FDE), so it is checked as a lower bound
plus a physical upper cap of 2 -- this guards against an order *regression*
while tolerating the observed over-performance.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

from benchmarks.numerical.convergence import (
    OPERATOR_ORDER_TOLERANCE,
    SOLVER_ORDER_LOWER_MARGIN,
    SOLVER_ORDER_UPPER_CAP,
    caputo_operator_order_row,
    estimate_order,
    riemann_liouville_order_row,
    row_passed,
    solver_endpoint_order_row,
)


def _is_strictly_decreasing(values: tuple[float, ...]) -> bool:
    # values[1:] is intentionally one shorter, so strict=False is correct here.
    return all(
        later < earlier for earlier, later in zip(values, values[1:], strict=False)
    )


def test_estimate_order_recovers_known_slope() -> None:
    # error = C * h^2 should estimate order 2 exactly.
    step_sizes = (0.1, 0.05, 0.025, 0.0125)
    errors = tuple(3.0 * h**2 for h in step_sizes)
    assert estimate_order(step_sizes, errors) == pytest.approx(2.0, abs=1e-9)


def test_estimate_order_rejects_nonpositive_errors() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        estimate_order((0.1, 0.05), (1e-3, 0.0))


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.7, 0.9])
def test_caputo_operator_matches_two_minus_alpha_order(alpha: float) -> None:
    row = caputo_operator_order_row(alpha=alpha)

    assert _is_strictly_decreasing(row.errors), row.errors
    assert row.expected_order == pytest.approx(2.0 - alpha)
    assert row.estimated_order == pytest.approx(
        row.expected_order, abs=OPERATOR_ORDER_TOLERANCE
    ), (
        f"alpha={alpha}: estimated order {row.estimated_order:.3f} "
        f"deviates from expected {row.expected_order:.3f} by more than "
        f"{OPERATOR_ORDER_TOLERANCE}"
    )
    assert row_passed(row)


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
def test_riemann_liouville_matches_first_order(alpha: float) -> None:
    row = riemann_liouville_order_row(alpha=alpha)

    assert _is_strictly_decreasing(row.errors), row.errors
    assert row.expected_order == 1.0
    assert row.estimated_order == pytest.approx(1.0, abs=OPERATOR_ORDER_TOLERANCE), (
        f"alpha={alpha}: RL/GL order {row.estimated_order:.3f} is not first order"
    )
    assert row_passed(row)


@pytest.mark.parametrize("alpha", [0.5, 0.7, 0.9])
def test_predictor_corrector_meets_one_plus_alpha_endpoint_order(alpha: float) -> None:
    row = solver_endpoint_order_row(alpha=alpha)

    assert _is_strictly_decreasing(row.errors), row.errors
    assert row.expected_order == pytest.approx(1.0 + alpha)
    assert row.estimated_order >= row.expected_order - SOLVER_ORDER_LOWER_MARGIN, (
        f"alpha={alpha}: endpoint order {row.estimated_order:.3f} dropped below "
        f"the expected {row.expected_order:.3f} (margin {SOLVER_ORDER_LOWER_MARGIN})"
    )
    assert row.estimated_order <= SOLVER_ORDER_UPPER_CAP, (
        f"alpha={alpha}: endpoint order {row.estimated_order:.3f} exceeds the "
        f"physical cap {SOLVER_ORDER_UPPER_CAP}; check the reference solution"
    )
    assert row_passed(row)
