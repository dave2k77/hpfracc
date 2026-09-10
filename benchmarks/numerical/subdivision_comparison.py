"""Paired scalar evidence for an experimental indicator, not an error-bound gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from dataclasses import asdict, dataclass
from math import gamma
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.numerical.adaptive_calibration import ROOT, _context, reference
from hpfracc.config import current_provenance
from hpfracc.solvers import AdaptivePredictorCorrector, simulate

CASES = ("cubic", "weak_power", "half_order_decay", "manufactured_feedback")
MODES = ("step_doubling", "subdivision")
TOLERANCES = (1e-3, 1e-4)


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    case: str
    dtype: str
    order: float
    estimator: str
    rtol: float
    atol: float
    max_steps: int
    first_step: float
    n_accepted: int
    n_rejected: int
    completed: bool
    max_abs_error: float | None
    max_scaled_error: float | None


def run_case(case, mode, rtol, dtype, *, max_steps=512):
    if case not in CASES or mode not in MODES or dtype not in ("float32", "float64"):
        raise ValueError("Unknown comparison case, mode, or dtype")
    if dtype == "float64" and not jax.config.x64_enabled:
        raise ValueError("float64 comparison requires JAX_ENABLE_X64=1")
    dt = getattr(jnp, dtype)
    order = jnp.asarray(0.8 if case == "manufactured_feedback" else 0.5, dtype=dt)
    a = float(order)
    coefficient = jnp.asarray(2 / gamma(3 - a), dtype=dt)
    initial = 1.0 if case == "half_order_decay" else 0.0
    first_step = 1.0 if case == "cubic" else 0.01

    def dynamics(t, state, params, *, rng_key=None, inputs=None):
        del params, rng_key, inputs
        if case == "cubic":
            return (t - t * (2 * t - 1) * (t - 1)) / 3
        if case == "weak_power":
            return t**0.25
        if case == "half_order_decay":
            return -state
        return coefficient * t ** (2 - order) - (state - t**2)

    atol = rtol * 0.01
    result = simulate(
        model=dynamics, ts=jnp.asarray([0., 1.], dtype=dt),
        solver=AdaptivePredictorCorrector(
            order=order, first_step=first_step, rtol=rtol, atol=atol,
            max_steps=max_steps, error_estimator=mode,
        ), initial_state=jnp.asarray(initial, dtype=dt), params=None,
    )
    d = result.solver_info.diagnostics
    n = int(d["n_nodes"])
    all_times, values = np.asarray(result.ts), np.asarray(result.latent_state)
    completed = bool(
        d["success"] and 2 <= n <= len(all_times)
        and values.shape == all_times.shape and str(values.dtype) == dtype
        and np.all(np.isfinite(values[:n])) and np.all(np.isfinite(all_times[:n]))
        and np.all(np.diff(all_times[:n]) > 0)
        and all_times[0] == 0 and all_times[n - 1] == 1 and values[0] == initial
        and int(d["n_accepted"]) == n - 1
        and int(d["n_attempted"]) == n - 1 + int(d["n_rejected"])
        and np.array_equal(np.asarray(d["valid_mask"]), np.arange(len(all_times)) < n)
    )
    error = scaled = None
    if completed:
        t = all_times[:n].astype(np.float64)
        if case == "cubic":
            truth = 2 * t ** (a + 2) / gamma(a + 3) - 4 * t ** (a + 3) / gamma(a + 4)
        elif case == "weak_power":
            truth = gamma(1.25) / gamma(a + 1.25) * t ** (a + .25)
        else:
            truth = reference(case, t, a)
        errors = np.abs(values[:n].astype(np.float64) - truth)
        ratios = errors / (atol + rtol * np.abs(truth))
        completed = bool(np.all(np.isfinite(errors)) and np.all(np.isfinite(ratios)))
        if completed:
            error, scaled = float(np.max(errors)), float(np.max(ratios))
    return ComparisonRow(case, str(values.dtype), a, mode, rtol, atol, max_steps,
                         first_step, int(d["n_accepted"]), int(d["n_rejected"]),
                         completed, error, scaled)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--check", action="store_true",
                        help="require complete finite measurements, not error <= tol")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.dtype == "float64" and not jax.config.x64_enabled:
        parser.error("float64 comparison requires JAX_ENABLE_X64=1")
    rows = [run_case(case, mode, tol, args.dtype, max_steps=args.max_steps)
            for case in CASES for tol in TOLERANCES for mode in MODES]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(ComparisonRow.__dataclass_fields__),
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    text = stream.getvalue()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / f"subdivision_comparison_{args.dtype}"
        stem.with_suffix(".csv").write_text(text, encoding="utf-8", newline="")
        context = _context(args.dtype, args.max_steps, False)
        context["scope"] = "experimental comparison, not an accuracy guarantee"
        context["provenance"] = current_provenance(config={
            "report": "subdivision_comparison", "dtype": args.dtype,
            "cases": CASES, "estimators": MODES, "tolerances": TOLERANCES,
            "atol_over_rtol": 0.01, "bounds": [0, 1], "max_steps": args.max_steps,
            "first_step": {case: 1.0 if case == "cubic" else .01 for case in CASES},
        }).to_dict()
        name = "benchmarks/numerical/subdivision_comparison.py"
        context["source_sha256"][name] = hashlib.sha256(
            (ROOT / name).read_bytes()
        ).hexdigest()
        context["csv_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        stem.with_suffix(".json").write_text(
            json.dumps(context, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    sys.stdout.write(text)
    return int(args.check and (not rows or not all(row.completed for row in rows)))


if __name__ == "__main__":
    raise SystemExit(main())
