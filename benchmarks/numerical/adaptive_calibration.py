"""Adaptive tolerance audit against independent continuous solutions.

This is a measurement report, not a global-error acceptance certificate.
--check requires complete finite measurements; scaled errors may exceed one.
No solver updates, controller choices, or global JAX precision flags are changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from math import gamma
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import erfcx

from hpfracc.config import current_provenance
from hpfracc.solvers import (
    AdaptivePredictorCorrector,
    NonUniformPredictorCorrector,
    simulate,
)

TOLERANCES = (1e-3, 1e-4, 1e-5)
CASES = {
    "constant_forcing": 0.5,
    "quadratic_forcing": 0.5,
    "half_order_decay": 0.5,
    "manufactured_feedback": 0.8,
}
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    """Errors at accepted nodes on [0, 1]; failed runs have no error metrics."""

    case: str
    dtype: str
    order: float
    rtol: float
    atol: float
    max_steps: int
    first_step: float
    n_accepted: int
    n_rejected: int
    operational_success: bool
    completed: bool
    max_abs_error: float | None
    endpoint_abs_error: float | None
    max_scaled_error: float | None


def reference(case: str, times: np.ndarray, order: float) -> np.ndarray:
    """Host float64 continuous oracles, independent of discrete solver weights."""
    times = np.asarray(times, dtype=np.float64)
    if case == "constant_forcing":
        return times**order / gamma(order + 1)
    if case == "quadratic_forcing":
        return 2 * times ** (order + 2) / gamma(order + 3)
    if case == "half_order_decay":
        if order != 0.5:
            raise ValueError("The erfcx identity requires order=0.5")
        return erfcx(np.sqrt(times))
    if case == "manufactured_feedback":
        return times**2
    raise ValueError(f"Unknown calibration case: {case}")


def run_case(case: str, *, rtol: float, dtype: str = "float32",
             max_steps: int = 512) -> CalibrationRow:
    """Measure one scalar case without equating local and global tolerances."""
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if dtype == "float64" and not jax.config.x64_enabled:
        raise ValueError("float64 audit requires JAX_ENABLE_X64=1")
    dt = getattr(jnp, dtype)
    order = jnp.asarray(CASES[case], dtype=dt)
    actual_order = float(order)
    coefficient = jnp.asarray(2 / gamma(3 - actual_order), dtype=dt)

    def dynamics(t, state, params, *, rng_key=None, inputs=None):
        del params, rng_key, inputs
        if case == "constant_forcing":
            return jnp.ones_like(state)
        if case == "quadratic_forcing":
            return t**2
        if case == "half_order_decay":
            return -state
        # D_C^alpha(t^2) plus a restoring term that vanishes on y=t^2.
        return coefficient * t ** (2 - order) - (state - t**2)

    atol = rtol * 0.01
    solver = AdaptivePredictorCorrector(
        order=order, rtol=rtol, atol=atol, max_steps=max_steps, first_step=0.01,
    )
    initial = jnp.asarray(1.0 if case == "half_order_decay" else 0.0, dtype=dt)
    result = simulate(model=dynamics, ts=jnp.asarray([0.0, 1.0], dtype=dt),
                      solver=solver, initial_state=initial, params=None)
    diag = result.solver_info.diagnostics
    n = int(diag["n_nodes"])
    times = np.asarray(result.ts)
    values = np.asarray(result.latent_state)
    mask = np.asarray(diag["valid_mask"], dtype=bool)
    success = bool(diag["success"])
    completed = bool(
        success and 2 <= n <= len(times)
        and values.shape == times.shape and str(values.dtype) == dtype
        and np.array_equal(mask, np.arange(len(times)) < n)
        and int(diag["n_accepted"]) == n - 1
        and int(diag["n_attempted"]) == n - 1 + int(diag["n_rejected"])
        and np.all(np.isfinite(times[:n])) and np.all(np.isfinite(values[:n]))
        and times[0] == 0 and times[n - 1] == 1
        and values[0] == float(initial) and np.all(np.diff(times[:n]) > 0)
    )
    max_error = endpoint_error = scaled_error = None
    if completed:
        truth = reference(case, times[:n], actual_order)
        error = np.abs(values[:n].astype(np.float64) - truth)
        scaled = error / (atol + rtol * np.abs(truth))
        completed = bool(np.all(np.isfinite(error)) and np.all(np.isfinite(scaled)))
        if completed:
            max_error = float(np.max(error))
            endpoint_error = float(error[-1])
            scaled_error = float(np.max(scaled))
    return CalibrationRow(
        case, str(values.dtype), actual_order, float(rtol), float(atol),
        int(max_steps), 0.01, int(diag["n_accepted"]), int(diag["n_rejected"]),
        success, completed, max_error, endpoint_error, scaled_error,
    )


def collect_rows(*, dtype: str = "float32", max_steps: int = 512):
    return [run_case(case, rtol=tol, dtype=dtype, max_steps=max_steps)
            for case in CASES for tol in TOLERANCES]


@dataclass(frozen=True, slots=True)
class FirstStepRow:
    """Quadratic-forcing panel identity and observed adaptive acceptance."""

    dtype: str
    order: float
    step: float
    coarse_value: float
    fine_value: float
    exact_value: float
    coarse_abs_error: float
    indicator: float
    underestimation_factor: float | None
    pair_matches_oracle: bool
    atol: float
    scaled_indicator: float
    scaled_coarse_error: float
    adaptive_success: bool
    n_accepted: int
    n_rejected: int
    retained_coarse: bool


def first_step_rows(*, dtype: str = "float32") -> list[FirstStepRow]:
    """Audit the current indicator formula, without assuming Richardson scaling.

    Closed forms integrate the linear interpolants of f(t)=t^2. The adaptive
    call records whether the coarse trial is retained; that behavior is observed,
    not required by the oracle check, so a future controller may reject it.
    """
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if dtype == "float64" and not jax.config.x64_enabled:
        raise ValueError("float64 audit requires JAX_ENABLE_X64=1")
    dt = getattr(jnp, dtype)
    h = float(jnp.asarray(0.1, dtype=dt))
    tolerance = 5e-6 if dtype == "float32" else 2e-13

    def forcing(t, state, params, *, rng_key=None, inputs=None):
        del state, params, rng_key, inputs
        return t**2

    rows = []
    for nominal_order in (0.1, 0.5, 0.9):
        order = jnp.asarray(nominal_order, dtype=dt)
        a = float(order)
        amplitude = h ** (a + 2) / gamma(a + 2)
        truth = 2 * h ** (a + 2) / gamma(a + 3)
        fine_oracle = amplitude * (0.5 + 2 ** (-a - 1))
        divisor = 2 ** (a + 1) - 1
        oracle_indicator = (amplitude - fine_oracle) / divisor
        values, precisions = [], []
        for nodes in ([0.0, h], [0.0, h / 2, h]):
            result = simulate(
                model=forcing, ts=jnp.asarray(nodes, dtype=dt),
                solver=NonUniformPredictorCorrector(order=order),
                initial_state=jnp.asarray(0.0, dtype=dt), params=None,
            )
            values.append(float(result.latent_state[-1]))
            precisions.append(str(result.latent_state.dtype))
        coarse, fine = values
        indicator = abs(coarse - fine) / divisor
        error = abs(coarse - truth)
        matched = bool(all(precision == dtype for precision in precisions)
                       and np.allclose(values, [amplitude, fine_oracle],
                                       rtol=tolerance, atol=0))
        # Select tolerance from the analytic oracle, not the measured pair.
        atol = 1.1 * oracle_indicator
        scale = atol + 1e-12 * max(abs(coarse), abs(fine))
        adaptive = simulate(
            model=forcing, ts=jnp.asarray([0.0, h], dtype=dt),
            solver=AdaptivePredictorCorrector(
                order=order, rtol=1e-12, atol=atol, first_step=h, max_steps=2,
            ), initial_state=jnp.asarray(0.0, dtype=dt), params=None,
        )
        diag = adaptive.solver_info.diagnostics
        retained = bool(diag["success"] and diag["n_accepted"] == 1
                        and diag["n_rejected"] == 0
                        and np.isclose(float(adaptive.latent_state[-1]), coarse,
                                       rtol=tolerance, atol=0))
        rows.append(FirstStepRow(
            dtype, a, h, coarse, fine, truth, error, indicator,
            error / indicator if indicator > 0 and np.isfinite(indicator) else None,
            matched, atol, indicator / scale, error / scale,
            bool(diag["success"]), int(diag["n_accepted"]), int(diag["n_rejected"]),
            retained,
        ))
    return rows


def _context(dtype: str, max_steps: int, first_step: bool) -> dict:
    sources = (
        "benchmarks/numerical/adaptive_calibration.py",
        "src/hpfracc/solvers/predictor_corrector.py",
        "src/hpfracc/solvers/base.py",
        "src/hpfracc/ops/_quadrature.py",
        "src/hpfracc/ops/orders.py",
    )
    try:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True, check=True)
        dirty = bool(status.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    config = {
        "audit": "adaptive_calibration", "dtype": dtype,
        "max_steps": max_steps, "first_step": 0.01,
        "tolerances": list(TOLERANCES), "atol_over_rtol": 0.01,
        "bounds": [0.0, 1.0],
    }
    if first_step:
        config = {
            "audit": "adaptive_first_step", "dtype": dtype, "max_steps": 2,
            "bounds": [0.0, 0.1], "first_step": 0.1, "rtol": 1e-12,
            "atol": "1.1 * analytic first-panel indicator",
            "nominal_orders": [0.1, 0.5, 0.9],
            "indicator": "abs(coarse - fine) / (2**(1+order) - 1)",
        }
    return {
        "scope": "development audit, not release sign-off or global-error guarantee",
        "provenance": current_provenance(config=config).to_dict(),
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": version("jaxlib"),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "devices": [str(device) for device in jax.devices()],
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                          for path in sources},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-steps", type=int, default=512,
                        help="attempt budget for the tolerance sweep only")
    parser.add_argument("--first-step", action="store_true",
                        help="audit quadratic first-panel identities and acceptance")
    parser.add_argument("--check", action="store_true",
                        help="require complete finite measurements, not error <= tol")
    parser.add_argument("--output-dir", type=Path,
                        help="also save CSV and companion source/environment JSON")
    args = parser.parse_args(argv)
    if args.dtype == "float64" and not jax.config.x64_enabled:
        parser.error("float64 audit requires JAX_ENABLE_X64=1")
    if args.first_step:
        rows = first_step_rows(dtype=args.dtype)
        fields = FirstStepRow.__dataclass_fields__
        complete = bool(rows) and all(row.pair_matches_oracle for row in rows)
        name = "adaptive_first_step"
    else:
        rows = collect_rows(dtype=args.dtype, max_steps=args.max_steps)
        fields = CalibrationRow.__dataclass_fields__
        complete = bool(rows) and all(
            row.completed and all(value is not None and np.isfinite(value)
                                  for value in (row.max_abs_error,
                                                row.endpoint_abs_error,
                                                row.max_scaled_error))
            for row in rows
        )
        name = "adaptive_calibration"
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(fields), lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    text = stream.getvalue()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / f"{name}_{args.dtype}"
        stem.with_suffix(".csv").write_text(text, encoding="utf-8", newline="")
        context = _context(args.dtype, args.max_steps, args.first_step)
        context["csv_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        stem.with_suffix(".json").write_text(
            json.dumps(context, indent=2, allow_nan=False)
            + "\n", encoding="utf-8",
        )
    sys.stdout.write(text)
    return int(args.check and not complete)


if __name__ == "__main__":
    raise SystemExit(main())
