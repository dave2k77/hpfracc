"""Coupled/scaled adaptive accuracy audit, not a calibrated error-bound check."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gamma

from benchmarks.numerical.adaptive_calibration import ROOT, _context
from hpfracc.config import current_provenance
from hpfracc.solvers import AdaptivePredictorCorrector, simulate

SCALES = {"balanced": (1., 1.), "mixed_scale": (1e-3, 1e3)}
ORDERS = (0.25, 0.5, 0.8)
MODES = ("step_doubling", "subdivision")
TOLERANCES = (1e-2, 1e-3)
STATUS_FIELDS = (
    "success", "reached_t_end", "n_nodes", "n_accepted", "n_rejected",
    "n_attempted", "valid_mask", "max_steps_reached", "nonfinite_detected",
    "step_size_underflow",
)


def reference(times):
    """Dimensionless continuous solution; no discrete solver weights involved."""
    t = np.asarray(times, dtype=np.float64)
    return np.stack((1 + t**2, 0.5 + t**3), axis=-1)


def dynamics(t, state, params, *, rng_key=None, inputs=None):
    del rng_key, inputs
    scale, alpha, feedback = params
    exact = jnp.stack((1 + t**2, 0.5 + t**3))
    error = state / scale - exact
    forcing = jnp.stack((
        2 / gamma(3 - alpha) * t ** (2 - alpha),
        6 / gamma(4 - alpha) * t ** (3 - alpha),
    ))
    coupling = jnp.asarray([[-1., .5], [.25, -2.]], dtype=state.dtype)
    return scale * (forcing + feedback * (coupling @ (error + .1 * error**3)))


@lru_cache(maxsize=8)
def runner(mode, rtol, max_steps):
    """Dynamic state/order/scale inputs reuse compilation across audit cases."""
    @jax.jit
    def run(initial, scale, order, feedback):
        result = simulate(
            model=dynamics, ts=jnp.asarray([0., 1.], dtype=initial.dtype),
            solver=AdaptivePredictorCorrector(
                order=order, rtol=rtol, atol=.01 * rtol, first_step=.01,
                max_steps=max_steps, error_estimator=mode,
            ), initial_state=initial, params=(scale, order, feedback),
        )
        d = result.solver_info.diagnostics
        return result.ts, result.latent_state, {key: d[key] for key in STATUS_FIELDS}
    return run


@dataclass(frozen=True, slots=True)
class CoupledRow:
    case: str
    dtype: str
    order: float
    estimator: str
    rtol: float
    atol: float
    max_steps: int
    n_accepted: int
    n_rejected: int
    valid_output: bool
    completed: bool
    max_steps_reached: bool
    nonfinite_detected: bool
    step_size_underflow: bool
    component_max_abs_error: tuple[float, ...] | None
    component_endpoint_abs_error: tuple[float, ...] | None
    max_dimensionless_error: float | None
    max_scaled_error: float | None
    endpoint_scaled_error: float | None


def summarize(output, case, mode, order, rtol, dtype, max_steps):
    """Validate fixed-shape outputs and retain failures without full-span errors."""
    times, values, d = jax.device_get(output)
    n, accepted, rejected = (int(d[key]) for key in (
        "n_nodes", "n_accepted", "n_rejected",
    ))
    scale = np.asarray(SCALES[case], dtype=dtype).astype(np.float64)
    initial = (scale * np.asarray([1., .5])).astype(dtype)
    terminal = [bool(d[key]) for key in (
        "success", "max_steps_reached", "nonfinite_detected", "step_size_underflow",
    )]
    valid = bool(
        times.shape == (max_steps + 2,) and values.shape == (max_steps + 2, 2)
        and str(times.dtype) == dtype and str(values.dtype) == dtype
        and 1 <= n <= max_steps + 1 and accepted == n - 1 and rejected >= 0
        and accepted + rejected == int(d["n_attempted"]) <= max_steps
        and np.all(np.isfinite(times)) and np.all(np.isfinite(values))
        and times[0] == 0 and np.all(np.diff(times[:n]) > 0)
        and 0 <= times[n - 1] <= 1 and np.array_equal(values[0], initial)
        and np.all(times[n:] == times[n - 1])
        and np.all(values[n:] == values[n - 1])
        and np.array_equal(d["valid_mask"], np.arange(len(times)) < n)
        and sum(terminal) == 1 and bool(d["success"]) == bool(d["reached_t_end"])
        and (not d["success"] or times[n - 1] == 1)
    )
    completed = valid and bool(d["success"])
    max_abs = endpoint_abs = dimensionless = scaled = endpoint_scaled = None
    if completed:
        truth = reference(times[:n]) * scale
        errors = np.abs(values[:n].astype(np.float64) - truth)
        ratios = errors / (.01 * rtol + rtol * np.abs(truth))
        completed = bool(np.all(np.isfinite(errors)) and np.all(np.isfinite(ratios)))
        if completed:
            max_abs = tuple(float(e) for e in np.max(errors, axis=0))
            endpoint_abs = tuple(float(e) for e in errors[-1])
            dimensionless = float(np.max(errors / scale))
            scaled, endpoint_scaled = float(np.max(ratios)), float(np.max(ratios[-1]))
    return CoupledRow(
        case, dtype, order, mode, rtol, .01 * rtol, max_steps, accepted, rejected,
        valid, completed, *terminal[1:], max_abs, endpoint_abs, dimensionless,
        scaled, endpoint_scaled,
    )


def run_case(case, mode, order, rtol, dtype, *, max_steps=1024):
    if dtype not in ("float32", "float64") or case not in SCALES or mode not in MODES:
        raise ValueError("Unknown dtype, case, or estimator")
    if dtype == "float64" and not jax.config.x64_enabled:
        raise ValueError("float64 audit requires JAX_ENABLE_X64=1")
    dt = getattr(jnp, dtype)
    scale = jnp.asarray(SCALES[case], dtype=dt)
    alpha = jnp.asarray(order, dtype=dt)
    output = runner(mode, rtol, max_steps)(
        scale * jnp.asarray([1., .5], dtype=dt), scale, alpha,
        jnp.asarray(1., dtype=dt),
    )
    return summarize(output, case, mode, float(alpha), rtol, dtype, max_steps)


def collect_rows(dtype, max_steps):
    return [run_case(case, mode, order, tol, dtype, max_steps=max_steps)
            for case in SCALES for order in ORDERS
            for mode in MODES for tol in TOLERANCES]


def rows_passed(rows, dtype, max_steps):
    expected = {(case, mode, float(np.asarray(order, dtype=dtype)), tol)
                for case in SCALES for order in ORDERS
                for mode in MODES for tol in TOLERANCES}
    actual = {(r.case, r.estimator, r.order, r.rtol) for r in rows}
    return (len(rows) == len(expected) and actual == expected
            and all(r.dtype == dtype and r.max_steps == max_steps
                    and r.valid_output and r.completed for r in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--check", action="store_true",
                        help="require exact matrix and finite completed measurements")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.max_steps < 2:
        parser.error("max-steps must be at least 2")
    if args.dtype == "float64" and not jax.config.x64_enabled:
        parser.error("float64 audit requires JAX_ENABLE_X64=1")
    rows = collect_rows(args.dtype, args.max_steps)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(CoupledRow.__dataclass_fields__),
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    text = stream.getvalue()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / f"adaptive_coupled_{args.dtype}"
        stem.with_suffix(".csv").write_text(text, encoding="utf-8", newline="")
        context = _context(args.dtype, args.max_steps, False)
        context["scope"] = "coupled/scaled accepted-node accuracy; not error bounds"
        context["provenance"] = current_provenance(config={
            "report": "adaptive_coupled", "dtype": args.dtype, "orders": ORDERS,
            "scales": SCALES, "estimators": MODES, "tolerances": TOLERANCES,
            "atol_over_rtol": .01, "first_step": .01, "bounds": [0., 1.],
            "feedback": 1., "max_steps": args.max_steps,
        }).to_dict()
        for name in ("benchmarks/numerical/adaptive_coupled.py",
                     "docs/validation/development-2026-09-14.md"):
            context["source_sha256"][name] = hashlib.sha256(
                (ROOT / name).read_bytes(),
            ).hexdigest()
        context["csv_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        context["rows"] = [asdict(row) for row in rows]
        stem.with_suffix(".json").write_text(
            json.dumps(context, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    sys.stdout.write(text)
    return int(args.check and not rows_passed(rows, args.dtype, args.max_steps))


if __name__ == "__main__":
    raise SystemExit(main())
