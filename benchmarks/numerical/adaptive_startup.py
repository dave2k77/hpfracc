"""Float64 first-panel scaling audit; does not modify the adaptive controller."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.metadata import version
from math import gamma, isfinite, log
from pathlib import Path

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np

from benchmarks.numerical.adaptive_calibration import ROOT, _context
from hpfracc.config import current_provenance
from hpfracc.solvers import (
    AdaptivePredictorCorrector,
    NonUniformPredictorCorrector,
    simulate,
)

ORDERS = (0.25, 0.5, 0.75)
Z_VALUES = (0.01, 0.005, 0.0025, 0.00125)
CASES = ("power_weak", "power_quadratic", "linear_decay", "nonlinear_decay")
BETAS = {"power_weak": 0.25, "power_quadratic": 2.0}
COEFFICIENT_TOLERANCE = 0.03
POWER_TOLERANCE = 0.05


def leading_coefficients(case, order):
    """Return power and coarse/fine/exact coefficients at that power.

    For autonomous cases the common constant and h**alpha terms are omitted.
    These are asymptotic coefficients, not finite-step error bounds.
    """
    if case not in CASES or not 0 < order < 1:
        raise ValueError("Expected a known case and 0 < order < 1")
    c, d, q = 1 / gamma(order + 1), 1 / gamma(order + 2), 2 ** (-order)
    if case in BETAS:
        beta = BETAS[case]
        return (order + beta, d, d * (2 * (1 - q) * 2 ** (-beta) + q),
                gamma(beta + 1) / gamma(order + beta + 1))
    product = 1 if case == "linear_decay" else 2  # f'(1)*f(1)
    return (2 * order, product * c * d, product * c * d * q * (3 - 2 * q),
            product / gamma(2 * order + 1))


@lru_cache(maxsize=32)
def _series_coefficients(case, order, terms):
    """Independent Volterra power-series recurrence, evaluated at 70 digits."""
    with mp.workdps(70):
        a = mp.mpf(order)
        coefficients = [mp.mpf(1)]
        for n in range(terms - 1):
            if case == "linear_decay":
                forcing = -coefficients[n]
            else:
                forcing = -mp.fsum(coefficients[k] * coefficients[n - k]
                                   for k in range(n + 1))
            coefficients.append(
                forcing * mp.gamma(n * a + 1) / mp.gamma((n + 1) * a + 1)
            )
        return tuple(coefficients)


def reference(case, order, step):
    """Return host value and 80/120-term agreement (not a rigorous tail bound)."""
    if case not in CASES or not 0 < order <= 1 or step <= 0:
        raise ValueError("Invalid reference case/order/step")
    with mp.workdps(70):
        a, h = mp.mpf(order), mp.mpf(step)
        if case in BETAS:
            beta = mp.mpf(BETAS[case])
            value = mp.gamma(beta + 1) / mp.gamma(a + beta + 1) * h ** (a + beta)
            return float(value), 0.0
        z = h**a
        if z > mp.mpf("0.04"):
            raise ValueError("Series reference is scoped to h**order <= 0.04")
        values = []
        for terms in (80, 120):
            coefficients = _series_coefficients(case, order, terms)
            value = mp.mpf(0)
            for coefficient in reversed(coefficients):
                value = value * z + coefficient
            values.append(value)
        return float(values[-1]), float(abs(values[-1] - values[0]))


@dataclass(frozen=True, slots=True)
class Measurement:
    step: float
    z: float
    coarse: float | None
    fine: float | None
    reference: float
    reference_agreement: float
    coarse_error: float | None
    fine_error: float | None
    valid: bool


@dataclass(frozen=True, slots=True)
class StartupRow:
    case: str
    order: float
    beta: float | None
    expected_power: float
    observed_power: float | None
    controller_power: float
    expected_coarse_coefficient: float
    measured_coarse_coefficient: float | None
    expected_fine_coefficient: float
    measured_fine_coefficient: float | None
    leading_coarse_gain: float
    measured_coarse_gain: float | None
    indicator_underestimation: float | None
    passed: bool


@dataclass(frozen=True, slots=True)
class StartupResult:
    summary: StartupRow
    measurements: tuple[Measurement, ...]


def _finite(value):
    return float(value) if isfinite(value) else None


def run_case(case, order):
    if not jax.config.x64_enabled:
        raise ValueError("Startup coefficient audit requires JAX_ENABLE_X64=1")
    power, coarse_coefficient, fine_coefficient, exact_coefficient = (
        leading_coefficients(case, order)
    )
    initial = 0.0 if case in BETAS else 1.0

    def dynamics(t, state, params, *, rng_key=None, inputs=None):
        del params, rng_key, inputs
        if case in BETAS:
            return t ** BETAS[case]
        return -state if case == "linear_decay" else -state**2

    measurements = []
    for z in Z_VALUES:
        step = z ** (1 / order)
        truth, agreement = reference(case, order, step)
        endpoints, valid = [], True
        for nodes in ([0.0, step], [0.0, step / 2, step]):
            times = jnp.asarray(nodes, dtype=jnp.float64)
            result = simulate(
                model=dynamics, ts=times,
                solver=NonUniformPredictorCorrector(
                    order=jnp.asarray(order, dtype=jnp.float64),
                ), initial_state=jnp.asarray(initial, dtype=jnp.float64), params=None,
            )
            values = np.asarray(result.latent_state)
            valid = bool(valid and values.shape == times.shape
                         and values.dtype == np.dtype("float64")
                         and np.array_equal(np.asarray(result.ts), np.asarray(times))
                         and np.all(np.isfinite(values)) and values[0] == initial)
            endpoints.append(float(values[-1]))
        coarse, fine = endpoints
        measurements.append(Measurement(
            step, step**order, _finite(coarse), _finite(fine), truth, agreement,
            _finite(coarse - truth), _finite(fine - truth), valid,
        ))
    previous, last = measurements[-2:]
    expected_coarse = coarse_coefficient - exact_coefficient
    expected_fine = fine_coefficient - exact_coefficient
    gain = abs(expected_coarse / (coarse_coefficient - fine_coefficient))
    observed = measured_coarse = measured_fine = measured_gain = underestimate = None
    valid = all(item.valid and item.reference_agreement <= 1e-50
                for item in measurements)
    if valid and previous.coarse_error and last.coarse_error:
        observed = log(abs(previous.coarse_error / last.coarse_error)) / log(
            previous.step / last.step
        )
        measured_coarse = last.coarse_error / last.step**power
        measured_fine = last.fine_error / last.step**power
        difference = abs(last.coarse - last.fine)
        if difference > 0:
            measured_gain = abs(last.coarse_error) / difference
            underestimate = measured_gain * (2 ** (order + 1) - 1)
    passed = bool(
        valid and observed is not None and measured_gain is not None
        and all(isfinite(value) for value in (observed, measured_coarse, measured_fine,
                                              measured_gain, underestimate))
        and abs(observed - power) <= POWER_TOLERANCE
        and abs(measured_coarse / expected_coarse - 1) <= COEFFICIENT_TOLERANCE
        and abs(measured_fine / expected_fine - 1) <= COEFFICIENT_TOLERANCE
    )
    return StartupResult(StartupRow(
        case, order, BETAS.get(case), power, observed, order + 2,
        expected_coarse, measured_coarse, expected_fine, measured_fine,
        gain, measured_gain, underestimate, passed,
    ), tuple(measurements))


def collect_results():
    return [run_case(case, order) for case in CASES for order in ORDERS]


def blind_spot_rows():
    """A smooth cubic whose coarse/fine samples lie on the same straight line."""
    if not jax.config.x64_enabled:
        raise ValueError("Startup coefficient audit requires JAX_ENABLE_X64=1")

    def forcing(t, state, params, *, rng_key=None, inputs=None):
        del state, params, rng_key, inputs
        return (t - t * (2 * t - 1) * (t - 1)) / 3

    rows = []
    for a in ORDERS:
        values = []
        valid = True
        for nodes in ([0.0, 1.0], [0.0, 0.5, 1.0]):
            result = simulate(
                model=forcing, ts=jnp.asarray(nodes, dtype=jnp.float64),
                solver=NonUniformPredictorCorrector(order=jnp.asarray(a)),
                initial_state=jnp.asarray(0.0, dtype=jnp.float64), params=None,
            )
            valid = bool(valid and result.latent_state.dtype == jnp.float64)
            values.append(float(result.latent_state[-1]))
        coarse, fine = values
        oracle = 1 / (3 * gamma(a + 2))
        truth = 2 / gamma(a + 3) - 4 / gamma(a + 4)
        error = coarse - truth
        error_oracle = a * (a - 1) / (3 * (a + 2) * (a + 3) * gamma(a + 2))
        matched = bool(valid and np.allclose(values, [oracle, oracle],
                                             rtol=2e-13, atol=0)
                       and np.isclose(error, error_oracle, rtol=2e-12, atol=0))
        adaptive = simulate(
            model=forcing, ts=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
            solver=AdaptivePredictorCorrector(
                order=jnp.asarray(a), first_step=1.0, rtol=1e-8, atol=1e-10,
                max_steps=2,
            ), initial_state=jnp.asarray(0.0, dtype=jnp.float64), params=None,
        )
        diag = adaptive.solver_info.diagnostics
        completed = bool(diag["success"] and float(adaptive.ts[-1]) == 1
                         and np.isfinite(float(adaptive.latent_state[-1])))
        rows.append({
            "order": a, "coarse": _finite(coarse), "fine": _finite(fine),
            "reference": truth, "signed_error": _finite(error),
            "difference": _finite(coarse - fine),
            "pair_matches_oracle": matched, "adaptive_completed": completed,
            "n_accepted": int(diag["n_accepted"]),
            "n_rejected": int(diag["n_rejected"]),
            "adaptive_abs_error": (
                abs(float(adaptive.latent_state[-1]) - truth) if completed else None
            ),
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="check selected asymptotic coefficients and powers")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if not jax.config.x64_enabled:
        parser.error("Startup coefficient audit requires JAX_ENABLE_X64=1")
    results = collect_results()
    blind_spots = blind_spot_rows()
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(StartupRow.__dataclass_fields__),
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(result.summary) for result in results)
    text = stream.getvalue()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / "adaptive_startup_float64"
        stem.with_suffix(".csv").write_text(text, encoding="utf-8", newline="")
        context = _context("float64", 2, True)
        context["scope"] = "selected first-panel asymptotic audit; not a controller fix"
        context["provenance"] = current_provenance(config={
            "audit": "adaptive_startup", "orders": ORDERS, "cases": CASES,
            "z_values": Z_VALUES, "step": "z**(1/order)", "dtype": "float64",
            "reference_dps": 70, "reference_terms": [80, 120],
            "coefficient_relative_tolerance": COEFFICIENT_TOLERANCE,
            "power_absolute_tolerance": POWER_TOLERANCE,
            "blind_spot": {"forcing": "t^2 - (2/3)*t^3", "bounds": [0, 1],
                           "first_step": 1.0, "max_steps": 2,
                           "rtol": 1e-8, "atol": 1e-10},
        }).to_dict()
        context["mpmath"] = version("mpmath")
        name = "benchmarks/numerical/adaptive_startup.py"
        context["source_sha256"][name] = hashlib.sha256(
            (ROOT / name).read_bytes()
        ).hexdigest()
        context["csv_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        context["measurements"] = [
            {"case": result.summary.case, "order": result.summary.order,
             "samples": [asdict(item) for item in result.measurements]}
            for result in results
        ]
        context["blind_spots"] = blind_spots
        stem.with_suffix(".json").write_text(
            json.dumps(context, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    sys.stdout.write(text)
    passed = (bool(results) and all(result.summary.passed for result in results)
              and bool(blind_spots)
              and all(row["pair_matches_oracle"] for row in blind_spots))
    return int(args.check and not passed)


if __name__ == "__main__":
    raise SystemExit(main())
