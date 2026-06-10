"""Empirical convergence-order validation for the v0.1 operator and solver.

This module estimates *observed* convergence orders by grid refinement and
compares them to the theoretically expected rates, rather than only checking
that the error decreases under refinement.

Expected rates:

- Caputo L1 operator: max-norm error scales as ``O(h^(2 - alpha))``.
- Caputo predictor-corrector (PECE) solver: final-time (endpoint) error scales
  as ``O(h^(1 + alpha))`` for ``0 < alpha < 1``.

Two deliberate choices make the estimate trustworthy:

1. Measurements run in float64. At fine grids the error approaches float32
   roundoff (~1e-7), which would corrupt a log-log slope. float64 keeps the
   asymptotic regime clean. The precision change is scoped to the measurement
   and restored afterwards.
2. The solver uses the endpoint error, not the max-norm. The linear-FDE
   reference solution ``E_alpha(lambda t^alpha)`` has a weak ``t^alpha``
   singularity at the origin that degrades the *global* max-norm rate; the
   endpoint rate is the theoretically clean, stable metric.

The report is a validation artifact, not a performance benchmark.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from math import gamma

import jax
import jax.numpy as jnp
import numpy as np

from hpfracc.ops import caputo, caputo_power_law
from hpfracc.solvers import PredictorCorrector, simulate

# Pass criteria, grounded in float64 measurements on the grids used here.
# The operator order sits just below ``2 - alpha`` and is checked two-sided.
# The solver endpoint order sits at or slightly above ``1 + alpha`` (a benign
# super-convergence for the smooth linear FDE), so it is checked as a lower
# bound plus a physical upper cap -- this catches an order *regression* while
# tolerating the observed over-performance.
OPERATOR_ORDER_TOLERANCE = 0.15
SOLVER_ORDER_LOWER_MARGIN = 0.15
SOLVER_ORDER_UPPER_CAP = 2.2


@dataclass(frozen=True, slots=True)
class ConvergenceRow:
    """One estimated convergence-order row."""

    target: str
    case: str
    alpha: float
    metric: str
    n_steps: tuple[int, ...]
    step_sizes: tuple[float, ...]
    errors: tuple[float, ...]
    expected_order: float
    estimated_order: float
    reference: str


@contextmanager
def _enabled_x64():
    """Enable JAX float64 for the duration of a measurement, then restore."""

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def estimate_order(step_sizes: Sequence[float], errors: Sequence[float]) -> float:
    """Estimate the convergence order as the log-log least-squares slope.

    The slope of ``log(error)`` against ``log(step_size)`` is the empirical
    order ``p`` in ``error ~ C * h^p``.
    """

    if len(step_sizes) != len(errors):
        msg = "step_sizes and errors must have equal length."
        raise ValueError(msg)
    if len(step_sizes) < 2:
        msg = "Need at least two refinement levels to estimate an order."
        raise ValueError(msg)
    if any(error <= 0.0 for error in errors):
        msg = "Convergence-order estimation requires strictly positive errors."
        raise ValueError(msg)

    log_h = np.log(np.asarray(step_sizes, dtype=float))
    log_e = np.log(np.asarray(errors, dtype=float))
    slope, _intercept = np.polyfit(log_h, log_e, 1)
    return float(slope)


def caputo_operator_order_row(
    *,
    alpha: float,
    power: float = 2.0,
    n_steps_values: Sequence[int] = (41, 81, 161, 321),
) -> ConvergenceRow:
    """Estimate the Caputo L1 operator order against an analytic power law."""

    step_sizes: list[float] = []
    errors: list[float] = []
    with _enabled_x64():
        for n_steps in n_steps_values:
            t = jnp.linspace(0.0, 1.0, n_steps)
            dt = float(t[1] - t[0])
            actual = caputo(t**power, dt=dt, order=alpha)
            expected = caputo_power_law(t, power=power, order=alpha)
            error = float(jnp.max(jnp.abs(actual[1:] - expected[1:])))
            step_sizes.append(dt)
            errors.append(error)

    return ConvergenceRow(
        target="caputo_operator",
        case="power_law_max_norm",
        alpha=alpha,
        metric="max_abs_error",
        n_steps=tuple(n_steps_values),
        step_sizes=tuple(step_sizes),
        errors=tuple(errors),
        expected_order=2.0 - alpha,
        estimated_order=estimate_order(step_sizes, errors),
        reference="analytic_caputo_power_law",
    )


def _linear_model(
    t: object,
    state: object,
    params: object,
    *,
    rng_key: object | None = None,
    inputs: object | None = None,
) -> object:
    del t, rng_key, inputs
    return params * state


def _mittag_leffler(alpha: float, z: object, *, terms: int = 120) -> object:
    """Approximate ``E_alpha(z)`` with a truncated power series."""

    values = jnp.asarray(z)
    total = jnp.zeros_like(values)
    for k in range(terms):
        total = total + (values**k) / gamma(alpha * k + 1.0)
    return total


def solver_endpoint_order_row(
    *,
    alpha: float,
    rate: float = -0.8,
    n_steps_values: Sequence[int] = (21, 41, 81, 161),
) -> ConvergenceRow:
    """Estimate the PECE solver endpoint order against a Mittag-Leffler reference."""

    step_sizes: list[float] = []
    errors: list[float] = []
    with _enabled_x64():
        for n_steps in n_steps_values:
            ts = jnp.linspace(0.0, 1.0, n_steps)
            dt = float(ts[1] - ts[0])
            result = simulate(
                model=_linear_model,
                ts=ts,
                solver=PredictorCorrector(dt=dt, order=alpha),
                initial_state=jnp.asarray(1.0),
                params=jnp.asarray(rate),
            )
            expected = _mittag_leffler(alpha, rate * ts**alpha)
            error = float(jnp.abs(result.latent_state[-1] - expected[-1]))
            step_sizes.append(dt)
            errors.append(error)

    return ConvergenceRow(
        target="predictor_corrector",
        case="linear_caputo_fde_endpoint",
        alpha=alpha,
        metric="endpoint_abs_error",
        n_steps=tuple(n_steps_values),
        step_sizes=tuple(step_sizes),
        errors=tuple(errors),
        expected_order=1.0 + alpha,
        estimated_order=estimate_order(step_sizes, errors),
        reference="truncated_mittag_leffler",
    )


def row_passed(row: ConvergenceRow) -> bool:
    """Apply the documented convergence-order pass criterion for a row.

    Operator orders are checked two-sided against ``2 - alpha``; solver endpoint
    orders are checked as a lower bound against ``1 + alpha`` (they may benignly
    super-converge) with a physical upper cap.
    """

    if row.target == "caputo_operator":
        return abs(row.estimated_order - row.expected_order) <= OPERATOR_ORDER_TOLERANCE
    if row.target == "predictor_corrector":
        return (
            row.estimated_order >= row.expected_order - SOLVER_ORDER_LOWER_MARGIN
            and row.estimated_order <= SOLVER_ORDER_UPPER_CAP
        )
    msg = f"Unknown convergence target: {row.target!r}"
    raise ValueError(msg)


def generate_rows(
    *,
    operator_alphas: Sequence[float] = (0.25, 0.5, 0.7, 0.9),
    solver_alphas: Sequence[float] = (0.5, 0.7, 0.9),
    operator_n_steps: Sequence[int] = (41, 81, 161, 321),
    solver_n_steps: Sequence[int] = (21, 41, 81, 161),
) -> list[ConvergenceRow]:
    """Generate the complete convergence-order validation table."""

    rows = [
        caputo_operator_order_row(alpha=alpha, n_steps_values=operator_n_steps)
        for alpha in operator_alphas
    ]
    rows.extend(
        solver_endpoint_order_row(alpha=alpha, n_steps_values=solver_n_steps)
        for alpha in solver_alphas
    )
    return rows


def write_csv(rows: Iterable[ConvergenceRow], stream: object = sys.stdout) -> None:
    """Write convergence-order rows as CSV."""

    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "target",
            "case",
            "alpha",
            "metric",
            "n_steps",
            "step_sizes",
            "errors",
            "expected_order",
            "estimated_order",
            "reference",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "target": row.target,
                "case": row.case,
                "alpha": row.alpha,
                "metric": row.metric,
                "n_steps": "|".join(str(n) for n in row.n_steps),
                "step_sizes": "|".join(f"{h:.6g}" for h in row.step_sizes),
                "errors": "|".join(f"{e:.6e}" for e in row.errors),
                "expected_order": f"{row.expected_order:.4f}",
                "estimated_order": f"{row.estimated_order:.4f}",
                "reference": row.reference,
            }
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operator-alphas",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.7, 0.9],
    )
    parser.add_argument(
        "--solver-alphas",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.9],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = generate_rows(
        operator_alphas=tuple(args.operator_alphas),
        solver_alphas=tuple(args.solver_alphas),
    )
    write_csv(rows)


if __name__ == "__main__":
    main()
