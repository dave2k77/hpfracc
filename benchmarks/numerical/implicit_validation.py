"""Independent implicit-solver accuracy and algebraic-residual measurements.

--check validates complete measurements and diagnostic contracts. It deliberately
retains unconverged/nonfinite configurations; it is not an accuracy certificate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import gamma
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gamma as jax_gamma
from scipy.special import erfcx

from benchmarks.numerical.adaptive_calibration import ROOT, _context
from hpfracc.config import current_provenance
from hpfracc.solvers import ImplicitPredictorCorrector, simulate

SCALES = {"half_decay": (1.,), "cubic": (1.,),
          "coupled": (1., 1.), "scaled": (1e-3, 1e3)}
ORDERS = (.3, .5, .8)
STRENGTHS = (1., 30.)
LENGTHS = (17, 65, 257)
ITERATIONS = (1, 3, 8)
RTOL = {"float32": 1e-5, "float64": 1e-10}
FIELDS = ("newton_converged", "newton_failure_count", "newton_nonfinite_count",
          "newton_step_converged", "newton_scaled_residuals",
          "newton_max_scaled_residual")


def orders(case):
    return (.5,) if case == "half_decay" else ORDERS


def reference(case, times, strength, scales):
    """Continuous oracles; no Mittag-Leffler truncation or refined solver path."""
    t = np.asarray(times, dtype=np.float64)
    if case == "half_decay":
        return erfcx(strength * np.sqrt(t))[..., None] * scales
    profile = np.stack((1 + t**2, .5 + t**3), axis=-1)
    return profile[..., :len(scales)] * scales


def dynamics(case, t, state, params, *, rng_key=None, inputs=None):
    del rng_key, inputs
    strength, forcing_order, scales = params
    if case == "half_decay":
        return -strength * state
    dimension = len(scales)
    profile = jnp.stack((1 + t**2, .5 + t**3))[:dimension]
    forcing = jnp.stack((2 / jax_gamma(3 - forcing_order) * t**(2 - forcing_order),
                         6 / jax_gamma(4 - forcing_order) * t**(3 - forcing_order)))
    if dimension == 1:
        restoring = strength * ((state / scales)**3 - profile**3)
    else:
        matrix = jnp.stack((jnp.stack((strength + 1, -strength)),
                            jnp.stack((-strength, strength + 1))))
        restoring = matrix @ ((state / scales)**3 - profile**3)
    return scales * (forcing[:dimension] - restoring)


def host_rhs(case, times, states, strength, order, scales):
    """Host float64 RHS for the independent discrete residual calculation."""
    if case == "half_decay":
        return -strength * states
    t = np.asarray(times, dtype=np.float64)
    profile = reference(case, t, strength, np.ones_like(scales))
    forcing = np.stack((2 / gamma(3 - order) * t**(2 - order),
                        6 / gamma(4 - order) * t**(3 - order)), axis=-1)
    matrix = (np.asarray([[strength]]) if len(scales) == 1 else
              np.asarray([[strength + 1, -strength], [-strength, strength + 1]]))
    return scales * (forcing[:, :len(scales)]
                     - ((states / scales)**3 - profile**3) @ matrix.T)


@lru_cache(maxsize=32)
def quadrature_weights(n, order):
    """Independent linear-interpolant Volterra integration on nominal dyadic nodes.

    Integrate each past panel by Gauss-Legendre quadrature. The newest panel's
    singular endpoint uses exact beta moments. This avoids the solver's uniform
    second-power-difference formulas and imports none of its private helpers.
    """
    dt = 1 / (n - 1)
    scale = dt**order / gamma(order)
    left, right = np.empty(n - 1), np.empty(n - 1)
    left[0], right[0] = scale / (order + 1), scale / (order * (order + 1))
    nodes, weights = np.polynomial.legendre.leggauss(32)
    fraction = (nodes + 1) / 2
    lags = np.arange(1, n - 1)[:, None] + fraction
    integrand = lags**(order - 1) * (weights / 2)
    left[1:] = scale * (integrand @ fraction)
    right[1:] = scale * (integrand @ (1 - fraction))
    result = np.zeros((n, n))
    for step in range(1, n):
        result[step, :step] += left[step - 1::-1]
        result[step, 1:step + 1] += right[step - 1::-1]
    return result


def independent_residuals(case, times, states, strength, order, scales, rtol, atol):
    weights = quadrature_weights(len(times), order)
    history_weights = weights.copy()
    np.fill_diagonal(history_weights, 0.)
    rhs = host_rhs(case, times, states, strength, order, scales)
    known = states[0] + history_weights @ rhs
    residual = states - known - np.diag(weights)[:, None] * rhs
    threshold = atol + rtol * np.maximum(np.abs(states), np.abs(known))
    return np.max(np.abs(residual[1:]) / threshold[1:], axis=1)


@lru_cache(maxsize=48)
def runner(case, n, iterations, dtype):
    scales = jnp.asarray(SCALES[case], dtype=getattr(jnp, dtype))
    ts = jnp.linspace(0., 1., n, dtype=scales.dtype)

    def model(t, state, params, *, rng_key=None, inputs=None):
        return dynamics(case, t, state, params, rng_key=rng_key, inputs=inputs)

    @jax.jit
    def run(initial, strength, order):
        result = simulate(
            model=model, ts=ts, initial_state=initial, params=(strength, order, scales),
            solver=ImplicitPredictorCorrector(
                dt=1 / (n - 1), order=order, newton_iterations=iterations,
                newton_rtol=RTOL[dtype], newton_atol=.01 * RTOL[dtype],
            ),
        )
        d = result.solver_info.diagnostics
        return result.ts, result.latent_state, {key: d[key] for key in FIELDS}
    return run


@dataclass(frozen=True, slots=True)
class ImplicitRow:
    case: str
    dtype: str
    order: float
    strength: float
    n_samples: int
    newton_iterations: int
    newton_rtol: float
    newton_atol: float
    measurement_ok: bool
    finite_trajectory: bool
    newton_converged: bool
    newton_failure_count: int
    newton_nonfinite_count: int
    reported_max_scaled_residual: float | None
    independent_max_scaled_residual: float | None
    residual_flag_disagreements: int | None
    max_dimensionless_error: float | None
    endpoint_dimensionless_error: float | None
    minimum_dimensionless_state: float | None
    component_max_abs_error: tuple[float, ...] | None


def summarize(output, case, dtype, order, strength, n, iterations):
    times, values, d = jax.device_get(output)
    scales = np.asarray(SCALES[case], dtype=dtype).astype(np.float64)
    initial = reference(case, np.asarray(0.), strength, scales).astype(dtype)
    residuals, flags = np.asarray(d["newton_scaled_residuals"]), np.asarray(
        d["newton_step_converged"],
    )
    failures = int(d["newton_failure_count"])
    nonfinite = int(d["newton_nonfinite_count"])
    valid = bool(
        times.shape == (n,) and values.shape == (n, len(scales))
        and str(times.dtype) == dtype and str(values.dtype) == dtype
        and np.array_equal(times, np.linspace(0, 1, n).astype(dtype))
        and np.array_equal(values[0], initial)
        and residuals.shape == flags.shape == (n - 1,) and flags.dtype == np.bool_
        and np.all(residuals >= 0) and np.array_equal(flags, residuals <= 1)
        and failures == np.count_nonzero(~flags)
        and nonfinite == np.count_nonzero(~np.isfinite(residuals))
        and bool(d["newton_converged"]) == bool(np.all(flags))
        and float(d["newton_max_scaled_residual"]) == float(np.max(residuals))
    )
    finite = bool(np.all(np.isfinite(values)))
    if not finite and nonfinite == 0:
        valid = False
    maximum = endpoint = minimum = component = independent = disagreements = None
    if valid and finite:
        truth = reference(case, times, strength, scales)
        with np.errstate(over="ignore", invalid="ignore"):
            error = np.abs(values.astype(np.float64) - truth)
            normalized = values.astype(np.float64) / scales
            reconstructed = independent_residuals(
                case, times, values.astype(np.float64), strength, order,
                scales, RTOL[dtype], .01 * RTOL[dtype],
            )
        valid = bool(np.all(np.isfinite(error)) and np.all(np.isfinite(reconstructed)))
        if valid:
            component = tuple(float(e) for e in np.max(error, axis=0))
            maximum, endpoint = float(np.max(error / scales)), float(np.max(
                error[-1] / scales,
            ))
            minimum = float(np.min(normalized))
            independent = float(np.max(reconstructed))
            disagreements = int(np.count_nonzero((reconstructed <= 1) != flags))
    reported = float(d["newton_max_scaled_residual"])
    return ImplicitRow(
        case, dtype, order, strength, n, iterations, RTOL[dtype], .01 * RTOL[dtype],
        valid, finite, bool(d["newton_converged"]), failures, nonfinite,
        reported if np.isfinite(reported) else None, independent, disagreements,
        maximum, endpoint, minimum, component,
    )


def run_case(case, n, iterations, strength, order, dtype):
    if dtype not in RTOL or case not in SCALES:
        raise ValueError("Unknown precision or case")
    if dtype == "float64" and not jax.config.x64_enabled:
        raise ValueError("float64 audit requires JAX_ENABLE_X64=1")
    if case == "half_decay" and order != .5:
        raise ValueError("The erfcx reference requires order=0.5")
    scales = np.asarray(SCALES[case], dtype=dtype)
    initial = reference(case, np.asarray(0.), strength, scales).astype(dtype)
    alpha = jnp.asarray(order, dtype=getattr(jnp, dtype))
    rate = jnp.asarray(strength, dtype=alpha.dtype)
    output = runner(case, n, iterations, dtype)(jnp.asarray(initial), rate, alpha)
    return summarize(output, case, dtype, float(alpha), float(rate), n, iterations)


def collect_rows(dtype):
    return [run_case(case, n, iterations, strength, order, dtype)
            for case in SCALES for order in orders(case) for strength in STRENGTHS
            for n in LENGTHS for iterations in ITERATIONS]


def rows_passed(rows, dtype):
    expected = {(case, float(np.asarray(order, dtype=dtype)), strength, n, iterations)
                for case in SCALES for order in orders(case) for strength in STRENGTHS
                for n in LENGTHS for iterations in ITERATIONS}
    actual = {(r.case, r.order, r.strength, r.n_samples, r.newton_iterations)
              for r in rows}
    return (len(rows) == len(expected) and actual == expected
            and all(r.dtype == dtype and r.measurement_ok for r in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=tuple(RTOL), default="float32")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.dtype == "float64" and not jax.config.x64_enabled:
        parser.error("float64 audit requires JAX_ENABLE_X64=1")
    rows = collect_rows(args.dtype)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(ImplicitRow.__dataclass_fields__),
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    text = stream.getvalue()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / f"implicit_validation_{args.dtype}"
        stem.with_suffix(".csv").write_text(text, encoding="utf-8", newline="")
        context = _context(args.dtype, 0, False)
        context["scope"] = (
            "implicit accuracy and algebraic residuals; not general robustness"
        )
        context["provenance"] = current_provenance(config={
            "report": "implicit_validation", "dtype": args.dtype,
            "cases": SCALES, "orders": ORDERS, "half_decay_order": .5,
            "strengths": STRENGTHS, "lengths": LENGTHS, "iterations": ITERATIONS,
            "newton_rtol": RTOL[args.dtype], "newton_atol": .01 * RTOL[args.dtype],
            "bounds": [0., 1.], "reference_quadrature_points": 32,
        }).to_dict()
        for name in ("benchmarks/numerical/implicit_validation.py",
                     "docs/validation/audits/implicit-2026-09-15/protocol.json"):
            context["source_sha256"][name] = hashlib.sha256(
                (ROOT / name).read_bytes(),
            ).hexdigest()
        context["csv_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        context["summary"] = {
            "configurations": len(rows),
            "valid_measurements": sum(r.measurement_ok for r in rows),
            "finite_trajectories": sum(r.finite_trajectory for r in rows),
            "converged_trajectories": sum(r.newton_converged for r in rows),
            "by_iterations": {
                str(i): {
                    "configurations": sum(r.newton_iterations == i for r in rows),
                    "finite": sum(r.finite_trajectory for r in rows
                                  if r.newton_iterations == i),
                    "converged": sum(r.newton_converged for r in rows
                                     if r.newton_iterations == i),
                } for i in ITERATIONS
            },
        }
        context["rows"] = [asdict(row) for row in rows]
        stem.with_suffix(".json").write_text(
            json.dumps(context, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    sys.stdout.write(text)
    return int(args.check and not rows_passed(rows, args.dtype))


if __name__ == "__main__":
    raise SystemExit(main())
