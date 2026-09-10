"""Fixed Phase A precision and solver-diagnostic validation cases.

All numerical inputs are explicitly float32 except the deliberately integer time
bounds. Reference arithmetic is host float64 or 70-digit mpmath; this module
never changes JAX's global precision setting. These are case-specific correctness
checks, not performance benchmarks or general error guarantees.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from math import fsum, gamma, isclose, isfinite
from typing import Any

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np

from hpfracc.ops import caputo
from hpfracc.solvers import (
    AdaptivePredictorCorrector,
    ImplicitPredictorCorrector,
    NonUniformPredictorCorrector,
    simulate,
)
from hpfracc.solvers.predictor_corrector import _nonuniform_pece_weights


@dataclass(frozen=True, slots=True)
class HardeningRow:
    """One measurement with the aggregate report's established CSV schema."""

    area: str
    case: str
    metric: str
    value: float
    passed: bool
    details: str


def _row(area, case, metric, value, passed, **details) -> HardeningRow:
    value = float(value) if value is not None else float("nan")
    return HardeningRow(
        area, case, metric, value, bool(passed and isfinite(value)),
        "; ".join(f"{key}={value}" for key, value in details.items()),
    )


def _linear(t, state, params, *, rng_key=None, inputs=None):
    del t, rng_key, inputs
    return params * state


def _cubic(t, state, params, *, rng_key=None, inputs=None):
    del t, rng_key, inputs
    return params * state**3


def _zero(t, state, params, *, rng_key=None, inputs=None):
    del t, params, rng_key, inputs
    return jnp.zeros_like(state)


def _fails_later(t, state, params, *, rng_key=None, inputs=None):
    del params, rng_key, inputs
    return jnp.where(t > 0.1, jnp.full_like(state, jnp.nan), jnp.zeros_like(state))


def _run(solver, *, model=_linear, ts=None, rate=-1.0):
    return simulate(
        model=model,
        ts=jnp.asarray([0.0, 1.0], dtype=jnp.float32) if ts is None else ts,
        solver=solver,
        initial_state=jnp.asarray(1.0, dtype=jnp.float32),
        params=jnp.asarray(rate, dtype=jnp.float32),
    )


def _reference_weights(nodes, order):
    """Independent high-precision closed-form interval integrals."""
    with mp.workdps(70):
        times = [mp.mpf(float(t)) for t in nodes]
        a = mp.mpf(float(order))
        normalization = mp.gamma(a)
        predictor, left, right = [], [], []
        for start, end in zip(times[:-1], times[1:], strict=True):
            upper, lower = times[-1] - start, times[-1] - end
            width = end - start
            zeroth = (upper**a - lower**a) / a
            first = (upper ** (a + 1) - lower ** (a + 1)) / (a + 1)
            predictor.append(zeroth / normalization)
            left.append((first - lower * zeroth) / (width * normalization))
            right.append((upper * zeroth - first) / (width * normalization))
        corrector = [left[0]]
        corrector.extend(left[j] + right[j - 1] for j in range(1, len(left)))
        return np.asarray([*predictor, *corrector, right[-1]], dtype=float)


def graded_precision_rows() -> list[HardeningRow]:
    """Check individual graded weights and an isolated early L1 increment."""
    nodes = (jnp.arange(160, dtype=jnp.float32) / 159) ** 4
    order = jnp.asarray(0.4, dtype=jnp.float32)
    p, c, diagonal = _nonuniform_pece_weights(
        159, nodes, jnp.arange(160), order, jnp.float32
    )
    # corrector[step] already contains the diagonal: include it only once.
    actual = np.concatenate([np.asarray(p[:-1]), np.asarray(c[:-1]), [float(diagonal)]])
    reference = _reference_weights(np.asarray(nodes), float(order))
    error = float(np.max(np.abs(actual / reference - 1)))
    rows = [_row(
        "solver", "graded_history_weights_float32", "max_relative_error", error,
        np.all(actual > 0) and error <= 5e-6,
        order=float(order), nodes=160, grade=4, precision="float32",
        reference="mpmath_70_digits_actual_nodes", tolerance=5e-6,
        min_corrector=float(jnp.min(c[:-1])),
    )]

    signal = jnp.concatenate([jnp.zeros(1, dtype=jnp.float32),
                              jnp.ones(159, dtype=jnp.float32)])
    value = float(caputo(signal, t=nodes, order=order)[-1])
    with mp.workdps(70):
        a = mp.mpf(float(order))
        upper = mp.mpf(float(nodes[-1])) - mp.mpf(float(nodes[0]))
        lower = mp.mpf(float(nodes[-1])) - mp.mpf(float(nodes[1]))
        width = mp.mpf(float(nodes[1])) - mp.mpf(float(nodes[0]))
        expected = float(
            (upper ** (1 - a) - lower ** (1 - a)) / (width * mp.gamma(2 - a))
        )
    error = abs(value / expected - 1)
    rows.append(_row(
        "operator", "graded_caputo_history_float32", "relative_error", error,
        error <= 5e-6, order=float(order), nodes=160, grade=4, precision="float32",
        reference="mpmath_70_digits_first_interval", tolerance=5e-6,
        actual=value, expected=expected,
    ))
    return rows


def nonuniform_refinement_row() -> HardeningRow:
    """Float32 graded-mesh decay versus a host-float64 Mittag-Leffler series."""
    errors = []
    for n in (40, 80, 160):
        nodes = (jnp.arange(n, dtype=jnp.float32) / (n - 1)) ** 4
        result = _run(NonUniformPredictorCorrector(order=0.4), ts=nodes)
        z = -np.asarray(nodes, dtype=float) ** 0.4
        reference = np.asarray([
            fsum(value**k / gamma(0.4 * k + 1) for k in range(100)) for value in z
        ])
        errors.append(float(np.max(
            np.abs(np.asarray(result.latent_state) - reference)
        )))
    return _row(
        "solver", "nonuniform_fde_refinement_float32", "last_max_abs_error", errors[-1],
        all(isfinite(error) for error in errors)
        and errors[0] > errors[1] > errors[2] and errors[-1] <= 5e-4,
        order=0.4, rate=-1.0, nodes="40|80|160", grade=4, precision="float32",
        reference="float64_mittag_leffler_100_terms_mild_argument",
        errors="|".join(map(str, errors)), finest_tolerance=5e-4,
    )


def _cubic_scaled_residual(states, nodes, solver) -> float:
    """Reconstruct the fixed first-step equation in independent host float64."""
    values = np.asarray(states, dtype=float)
    if (
        values.shape != (2,) or not np.all(np.isfinite(values)) or values[0] != 1.0
        or not np.array_equal(nodes, [0.0, solver.dt])
    ):
        return float("nan")
    order = float(solver.order)
    scale = solver.dt**order / gamma(order + 2)
    known = 1.0 + scale * order * (-2.0)
    y = float(values[-1])
    residual = y - known - scale * (-2.0 * y**3)
    tolerance = solver.newton_atol + solver.newton_rtol * max(abs(y), abs(known))
    return abs(residual) / tolerance


def _residual_agrees(reported, independent) -> bool:
    # Host gamma and arithmetic differ from float32 evaluation. Permit one
    # scaled tolerance unit of roundoff, plus relative float32 accumulation error;
    # the independent convergence/failure threshold is checked separately.
    return bool(
        reported is not None and isfinite(float(reported)) and reported >= 0
        and isfinite(independent)
        and isclose(float(reported), independent, rel_tol=1e-5, abs_tol=1.0)
    )


def implicit_rows() -> list[HardeningRow]:
    """Converged and deliberately under-iterated cubic correctors, plus JIT."""
    nodes = jnp.asarray([0.0, 0.5], dtype=jnp.float32)
    # A strong float32 order also fixes the gamma/scaling dtype when callers
    # have enabled x64; Python scalar orders can otherwise promote this path.
    order = jnp.asarray(0.7, dtype=jnp.float32)
    rows = []
    for iterations, expected_success in ((8, True), (1, False)):
        solver = ImplicitPredictorCorrector(
            dt=0.5, order=order, newton_iterations=iterations,
            newton_rtol=1e-6, newton_atol=1e-8,
        )
        result = _run(solver, model=_cubic, ts=nodes, rate=-2.0)
        diag = result.solver_info.diagnostics
        residual = diag["newton_max_scaled_residual"]
        independent = _cubic_scaled_residual(result.latent_state, result.ts, solver)
        expected_failures = 0 if expected_success else 1
        passed = (
            diag["newton_converged"] is expected_success
            and diag["newton_failure_count"] == expected_failures
            and diag["newton_nonfinite_count"] == 0
            and diag["newton_step_converged"] == [expected_success]
            and bool(result.solver_info.warnings) is (not expected_success)
            and _residual_agrees(residual, independent)
            and diag["newton_scaled_residuals"] == [residual]
            and ((residual <= 1 and independent <= 1) if expected_success
                 else (residual > 1 and independent > 1))
        )
        case = ("implicit_nonlinear_converged" if expected_success
                else "implicit_nonlinear_failure_detected")
        rows.append(_row(
            "solver", case, "independent_scaled_residual", independent, passed,
            order=0.7, rate=-2.0, dt=0.5, precision="float32", iterations=iterations,
            rtol=1e-6, atol=1e-8, threshold=1, expected_success=expected_success,
            reported_scaled_residual=residual, reference="float64_first_step_equation",
            agreement_rtol=1e-5, agreement_atol=1.0,
        ))

    jit_solver = ImplicitPredictorCorrector(
        dt=0.5, order=order, newton_iterations=1, newton_rtol=1e-6, newton_atol=1e-8
    )

    def compiled_status(rate):
        result = _run(jit_solver, model=_cubic, ts=nodes, rate=rate)
        diag = result.solver_info.diagnostics
        return (
            result.latent_state, result.ts, diag["newton_converged"],
            diag["newton_failure_count"], diag["newton_max_scaled_residual"],
        )

    states, times, converged, failures, residual = jax.jit(compiled_status)(
        jnp.asarray(-2.0, dtype=jnp.float32)
    )
    independent = _cubic_scaled_residual(states, times, jit_solver)
    rows.append(_row(
        "solver", "implicit_jit_failure_detected", "failure_count", failures,
        converged is not None and not bool(converged) and int(failures) == 1
        and _residual_agrees(residual, independent)
        and independent > 1 and residual > 1,
        order=0.7, rate=-2.0, dt=0.5, precision="float32", iterations=1,
        expected_failure_count=1,
        independent_scaled_residual=independent,
        reported_scaled_residual=float(residual),
        reference="float64_first_step_equation",
        agreement_rtol=1e-5, agreement_atol=1.0,
    ))
    return rows


def _adaptive_ok(result, *, terminal, budget, accepted, rejected) -> bool:
    """Validate independent expected status/counts and retained-prefix semantics."""
    diag = result.solver_info.diagnostics
    status_names = ("success", "max_steps_reached", "nonfinite_detected",
                    "step_size_underflow")
    n = diag["n_nodes"]
    nodes, states = np.asarray(result.ts), np.asarray(result.latent_state)
    if n is None or not 1 <= n <= len(nodes):
        return False
    mask = np.asarray(diag["valid_mask"])
    return bool(
        all(bool(diag[name]) == (name == terminal) for name in status_names)
        and bool(diag["reached_t_end"]) == (terminal == "success")
        and diag["n_accepted"] == accepted and diag["n_rejected"] == rejected
        and diag["n_attempted"] == accepted + rejected <= budget
        and n == accepted + 1 and result.solver_info.n_steps == accepted
        and mask.shape == (budget + 2,) and mask.dtype == np.bool_
        and np.array_equal(mask, np.arange(budget + 2) < n)
        and len(nodes) == budget + 2 and len(states) == budget + 2
        and np.all(np.diff(nodes[:n]) > 0) and np.all(np.isfinite(states))
        and np.all(states == 1) and np.all(nodes[n:] == nodes[n - 1])
        and float(diag["last_time"]) == float(nodes[n - 1])
        and bool(result.solver_info.warnings) == (terminal != "success")
    )


def adaptive_rows() -> list[HardeningRow]:
    """Completion and intentionally failing integrations with fixed configurations."""
    rows = []
    complete = None
    for name, first_step, terminal in (
        ("adaptive_completed", 0.5, "success"),
        ("adaptive_budget_failure_detected", 0.01, "max_steps_reached"),
    ):
        solver = AdaptivePredictorCorrector(
            order=0.7, first_step=first_step, max_steps=2
        )
        result = _run(solver, model=_zero)
        diag = result.solver_info.diagnostics
        if terminal == "success":
            complete = result
        gap = abs(float(result.ts[-1]) - 1.0)
        passed = _adaptive_ok(
            result, terminal=terminal, budget=2, accepted=2, rejected=0
        )
        passed = passed and ((gap == 0) if terminal == "success" else (gap > 0))
        rows.append(_row(
            "solver", name, "endpoint_time_gap", gap, passed,
            order=0.7, first_step=first_step, max_steps=2, precision="float32",
            expected_terminal=terminal, accepted=diag["n_accepted"],
            rejected=diag["n_rejected"], last_time=diag["last_time"],
        ))

    solver = AdaptivePredictorCorrector(order=0.7, first_step=0.05, max_steps=16)
    result = _run(solver, model=_fails_later)
    rows.append(_row(
        "solver", "adaptive_nonfinite_failure_detected", "rejected_steps",
        result.solver_info.diagnostics["n_rejected"],
        _adaptive_ok(
            result, terminal="nonfinite_detected", budget=16, accepted=1, rejected=1
        )
        and abs(float(result.ts[-1]) - 0.05) < 1e-7,
        order=0.7, first_step=0.05, max_steps=16, precision="float32",
        nonfinite_after=0.1, expected_accepted=1, expected_rejected=1,
    ))

    start = jnp.asarray(1.0, dtype=jnp.float32)
    end = jnp.nextafter(start, jnp.asarray(2.0, dtype=jnp.float32))
    solver = AdaptivePredictorCorrector(
        order=0.7, first_step=float(end - start), max_steps=4
    )
    result = _run(solver, model=_zero, ts=jnp.stack([start, end]))
    rows.append(_row(
        "solver", "adaptive_step_size_failure_detected", "accepted_steps",
        result.solver_info.diagnostics["n_accepted"],
        _adaptive_ok(
            result, terminal="step_size_underflow", budget=4, accepted=0, rejected=1
        ),
        order=0.7, precision="float32", bounds="adjacent_floats_above_1",
        expected_accepted=0, expected_rejected=1,
    ))

    solver = AdaptivePredictorCorrector(order=0.7, max_steps=8)
    result = _run(solver, model=_zero,
                  ts=jnp.asarray([2**24, 2**24 + 1], dtype=jnp.int32))
    diag = result.solver_info.diagnostics
    rows.append(_row(
        "solver", "adaptive_integer_precision_failure_detected", "attempted_steps",
        diag["n_attempted"],
        _adaptive_ok(
            result, terminal="step_size_underflow", budget=8, accepted=0, rejected=0
        )
        and diag["time_bounds_representable"] is False
        and diag["requested_t_end"] == 2**24 + 1,
        order=0.7, bounds="16777216|16777217", input_dtype="int32",
        time_dtype="float32", expected_attempts=0,
    ))

    solver = AdaptivePredictorCorrector(order=0.7, first_step=0.5, max_steps=2)

    def compiled_status(rate):
        result = _run(solver, model=_zero, rate=rate)
        diag = result.solver_info.diagnostics
        return (
            result.latent_state, diag["success"],
            diag["valid_mask"], diag["n_attempted"],
        )

    states, success, mask, attempts = jax.jit(compiled_status)(
        jnp.asarray(-1.0, dtype=jnp.float32)
    )
    difference = float(np.max(
        np.abs(np.asarray(states) - np.asarray(complete.latent_state))
    ))
    rows.append(_row(
        "solver", "adaptive_jit_status", "max_abs_state_difference", difference,
        success is not None and bool(success) and int(attempts) == 2
        and np.array_equal(mask, complete.solver_info.diagnostics["valid_mask"])
        and difference <= 1e-7,
        order=0.7, first_step=0.5, max_steps=2, precision="float32", tolerance=1e-7,
    ))
    return rows


def generate_rows() -> list[HardeningRow]:
    """Run all fixed hardening cases; their configurations are recorded per row."""
    return [*graded_precision_rows(), nonuniform_refinement_row(),
            *implicit_rows(), *adaptive_rows()]


def write_csv(rows: Iterable[HardeningRow], stream: Any = sys.stdout) -> None:
    writer = csv.DictWriter(
        stream, fieldnames=["area", "case", "metric", "value", "passed", "details"]
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Exit 1 if any row fails.")
    args = parser.parse_args(argv)
    rows = generate_rows()
    write_csv(rows, sys.stdout)
    return int(args.check and (not rows or not all(row.passed for row in rows)))


if __name__ == "__main__":
    raise SystemExit(main())
