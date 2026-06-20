"""Finite-difference gradient checks for differentiable numerical surfaces."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from hpfracc.ops import caputo, grunwald_letnikov, riemann_liouville
from hpfracc.solvers import PredictorCorrector, simulate


@contextmanager
def _enabled_x64():
    """Enable JAX float64 for the duration of a measurement, then restore.

    The fractional-order finite differences need double precision: in single
    precision the float32 roundoff floor (~1e-7) is comparable to the gradient
    perturbation and corrupts the check.
    """

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@dataclass(frozen=True, slots=True)
class GradientCheckRow:
    """One finite-difference gradient check row."""

    target: str
    parameter: str
    autodiff: float
    finite_difference: float
    abs_error: float
    step: float


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


def central_difference(fn: object, x: float, step: float) -> float:
    """Compute a scalar central finite difference."""

    return float((fn(x + step) - fn(x - step)) / (2.0 * step))


def operator_input_row(*, step: float = 1e-3) -> GradientCheckRow:
    """Check an operator gradient with respect to one input sample."""

    dt = 0.1

    def objective(sample: object) -> object:
        x = jnp.linspace(0.0, 1.0, 8).at[3].set(sample)
        return jnp.sum(caputo(x, dt=dt, order=0.5) ** 2)

    autodiff = float(jax.grad(objective)(jnp.asarray(0.3)))
    finite_difference = central_difference(objective, 0.3, step)
    return _row(
        target="caputo_operator",
        parameter="input_sample",
        autodiff=autodiff,
        finite_difference=finite_difference,
        step=step,
    )


def solver_initial_state_row(*, step: float = 1e-3) -> GradientCheckRow:
    """Check a solver gradient with respect to the initial state."""

    ts = jnp.linspace(0.0, 0.5, 11)
    solver = PredictorCorrector(dt=float(ts[1] - ts[0]), order=0.6)

    def objective(initial_state: object) -> object:
        result = simulate(
            model=linear_model,
            ts=ts,
            solver=solver,
            initial_state=jnp.asarray(initial_state),
            params=jnp.asarray(-0.25),
        )
        return result.latent_state[-1]

    autodiff = float(jax.grad(objective)(jnp.asarray(1.0)))
    finite_difference = central_difference(objective, 1.0, step)
    return _row(
        target="caputo_solver",
        parameter="initial_state",
        autodiff=autodiff,
        finite_difference=finite_difference,
        step=step,
    )


def solver_parameter_row(*, step: float = 1e-3) -> GradientCheckRow:
    """Check a solver gradient with respect to a scalar model parameter."""

    ts = jnp.linspace(0.0, 0.5, 11)
    solver = PredictorCorrector(dt=float(ts[1] - ts[0]), order=0.6)

    def objective(rate: object) -> object:
        result = simulate(
            model=linear_model,
            ts=ts,
            solver=solver,
            initial_state=jnp.asarray(1.0),
            params=jnp.asarray(rate),
        )
        return result.latent_state[-1]

    autodiff = float(jax.grad(objective)(jnp.asarray(-0.25)))
    finite_difference = central_difference(objective, -0.25, step)
    return _row(
        target="caputo_solver",
        parameter="rate",
        autodiff=autodiff,
        finite_difference=finite_difference,
        step=step,
    )


_OPERATORS = {
    "caputo": caputo,
    "grunwald_letnikov": grunwald_letnikov,
    "riemann_liouville": riemann_liouville,
}


def operator_alpha_row(
    target: str, *, alpha: float = 0.6, step: float = 1e-5
) -> GradientCheckRow:
    """Check an operator gradient with respect to the fractional order alpha."""

    operator = _OPERATORS[target]
    dt = 0.05

    with _enabled_x64():
        ts = jnp.arange(16, dtype=jnp.float64) * dt
        x = ts**2

        def objective(order: object) -> object:
            return jnp.sum(operator(x, dt=dt, order=order) ** 2)

        autodiff = float(jax.grad(objective)(jnp.asarray(alpha, dtype=jnp.float64)))
        finite_difference = central_difference(objective, alpha, step)
    return _row(
        target=f"{target}_operator",
        parameter="alpha",
        autodiff=autodiff,
        finite_difference=finite_difference,
        step=step,
    )


def solver_alpha_row(*, alpha: float = 0.7, step: float = 1e-5) -> GradientCheckRow:
    """Check a solver gradient with respect to the fractional order alpha."""

    dt = 0.05
    with _enabled_x64():
        ts = jnp.arange(40, dtype=jnp.float64) * dt

        def objective(order: object) -> object:
            solver = PredictorCorrector(dt=dt, order=order)
            result = simulate(
                model=linear_model,
                ts=ts,
                solver=solver,
                initial_state=jnp.asarray(1.0, dtype=jnp.float64),
                params=jnp.asarray(-0.5, dtype=jnp.float64),
            )
            return result.latent_state[-1]

        autodiff = float(jax.grad(objective)(jnp.asarray(alpha, dtype=jnp.float64)))
        finite_difference = central_difference(objective, alpha, step)
    return _row(
        target="caputo_solver",
        parameter="alpha",
        autodiff=autodiff,
        finite_difference=finite_difference,
        step=step,
    )


def generate_rows(*, step: float = 1e-3) -> list[GradientCheckRow]:
    """Generate all finite-difference gradient check rows."""

    return [
        operator_input_row(step=step),
        solver_initial_state_row(step=step),
        solver_parameter_row(step=step),
        operator_alpha_row("caputo"),
        operator_alpha_row("grunwald_letnikov"),
        operator_alpha_row("riemann_liouville"),
        solver_alpha_row(),
    ]


def write_csv(rows: Iterable[GradientCheckRow], stream: object = sys.stdout) -> None:
    """Write gradient check rows as CSV."""

    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "target",
            "parameter",
            "autodiff",
            "finite_difference",
            "abs_error",
            "step",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "target": row.target,
                "parameter": row.parameter,
                "autodiff": row.autodiff,
                "finite_difference": row.finite_difference,
                "abs_error": row.abs_error,
                "step": row.step,
            }
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    write_csv(generate_rows(step=args.step))


def _row(
    *,
    target: str,
    parameter: str,
    autodiff: float,
    finite_difference: float,
    step: float,
) -> GradientCheckRow:
    return GradientCheckRow(
        target=target,
        parameter=parameter,
        autodiff=autodiff,
        finite_difference=finite_difference,
        abs_error=abs(autodiff - finite_difference),
        step=step,
    )


if __name__ == "__main__":
    main()
