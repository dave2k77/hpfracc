"""Nonlinear/long-span evidence with approximate work matching, not error bounds."""

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
from scipy.special import erfcx

from benchmarks.numerical.adaptive_calibration import ROOT, _context
from hpfracc.config import current_provenance
from hpfracc.solvers import AdaptivePredictorCorrector, simulate


@dataclass(frozen=True, slots=True)
class Problem:
    end: float
    feedback: float


CASES = {
    "nonlinear": Problem(1.0, 1.0),
    "strong_feedback": Problem(1.0, 10.0),
    "long_decay": Problem(16.0, 0.0),
}
MODES = ("step_doubling", "subdivision")
TOLERANCES = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
MATCH_ALLOWANCE = 0.2


@dataclass(frozen=True, slots=True)
class StressRow:
    case: str
    dtype: str
    order: float
    end: float
    feedback: float
    estimator: str
    rtol: float
    atol: float
    first_step: float
    max_steps: int
    n_accepted: int
    n_rejected: int
    completed: bool
    max_steps_reached: bool
    nonfinite_detected: bool
    step_size_underflow: bool
    pece_evaluations: int
    rhs_evaluations: int
    weight_slots: int
    max_abs_error: float | None
    endpoint_abs_error: float | None
    max_scaled_error: float | None


def reference(case, times):
    """Continuous solution evaluated in host float64, not a refined PECE solve."""
    if case not in CASES:
        raise ValueError("Unknown stress case")
    times = np.asarray(times, dtype=np.float64)
    return erfcx(np.sqrt(times)) if case == "long_decay" else 1 + times**2


def rhs(case, t, state):
    if case == "long_decay":
        return -state
    # D^0.5 (1+t^2) = 2*t^1.5/Gamma(2.5). Feedback vanishes on the oracle.
    return 2 / gamma(2.5) * t**1.5 - CASES[case].feedback * (
        state**2 - (1 + t**2)**2
    )


def work_counts(mode, accepted, rejected, max_steps):
    """Source-level work proxy; not FLOPs, active panels, timing or memory bytes.

    Each trial calls _adaptive_pc_at 3/7 times, over buffers max_steps+2/+4.
    Each such call builds both predictor and corrector weights. Count one
    buffer slot per weight-pair builder, not each scalar arithmetic operation.
    Model calls are 5/13 per attempted trial plus the initial evaluation.
    Include terminating/rejected attempts; exclude inactive scan iterations.
    """
    if mode not in MODES:
        raise ValueError("Unknown estimator")
    attempts = accepted + rejected
    pece, model_calls, padding = (3, 5, 2) if mode == MODES[0] else (7, 13, 4)
    return (pece * attempts, 1 + model_calls * attempts,
            pece * attempts * (max_steps + padding))


def run_case(case, mode, rtol, dtype, *, max_steps=1024):
    if case not in CASES or mode not in MODES or dtype not in ("float32", "float64"):
        raise ValueError("Unknown stress case, mode, or dtype")
    if dtype == "float64" and not jax.config.x64_enabled:
        raise ValueError("float64 stress audit requires JAX_ENABLE_X64=1")
    problem = CASES[case]
    dt = getattr(jnp, dtype)
    order = jnp.asarray(0.5, dtype=dt)
    atol, first_step = rtol * 0.01, 0.01

    def dynamics(t, state, params, *, rng_key=None, inputs=None):
        del params, rng_key, inputs
        return rhs(case, t, state)

    result = simulate(
        model=dynamics, ts=jnp.asarray([0., problem.end], dtype=dt),
        solver=AdaptivePredictorCorrector(
            order=order, first_step=first_step, rtol=rtol, atol=atol,
            max_steps=max_steps, error_estimator=mode,
        ), initial_state=jnp.asarray(1., dtype=dt), params=None,
    )
    return summarize_result(result, case, mode, rtol, dtype, max_steps=max_steps)


def summarize_result(result, case, mode, rtol, dtype, *, max_steps):
    """Apply the same independent measurement checks to eager or compiled output."""
    problem = CASES[case]
    order, atol, first_step = 0.5, rtol * 0.01, 0.01
    d = result.solver_info.diagnostics
    n = int(d["n_nodes"])
    accepted, rejected = int(d["n_accepted"]), int(d["n_rejected"])
    times, values = np.asarray(result.ts), np.asarray(result.latent_state)
    completed = bool(
        d["success"] and d["reached_t_end"] and 2 <= n <= len(times)
        and values.shape == times.shape == (max_steps + 2,)
        and str(values.dtype) == dtype and str(times.dtype) == dtype
        and np.all(np.isfinite(times[:n])) and np.all(np.isfinite(values[:n]))
        and np.all(np.diff(times[:n]) > 0)
        and times[0] == 0 and times[n - 1] == problem.end and values[0] == 1
        and accepted == n - 1 and rejected >= 0
        and int(d["n_attempted"]) == accepted + rejected <= max_steps
        and not any(d[key] for key in (
            "max_steps_reached", "nonfinite_detected", "step_size_underflow",
        ))
        and np.array_equal(np.asarray(d["valid_mask"]), np.arange(len(times)) < n)
    )
    error = endpoint = scaled = None
    if completed:
        truth = reference(case, times[:n])
        errors = np.abs(values[:n].astype(np.float64) - truth)
        ratios = errors / (atol + rtol * np.abs(truth))
        completed = bool(np.all(np.isfinite(errors)) and np.all(np.isfinite(ratios)))
        if completed:
            error, endpoint, scaled = (
                float(np.max(errors)), float(errors[-1]), float(np.max(ratios)),
            )
    pece, calls, slots = work_counts(mode, accepted, rejected, max_steps)
    return StressRow(
        case, str(values.dtype), float(order), problem.end, problem.feedback,
        mode, rtol, atol, first_step, max_steps, accepted, rejected, completed,
        bool(d["max_steps_reached"]), bool(d["nonfinite_detected"]),
        bool(d["step_size_underflow"]), pece, calls, slots, error, endpoint, scaled,
    )


def match_work(rows, *, allowance=MATCH_ALLOWANCE):
    """Emit every eligible pair, selected only by work, never by error improvement.

    A match means max(work)/min(work) <= 1+allowance. No interpolation or
    extrapolation. Comparing each mesh's maximum is not a common-grid bound;
    endpoint errors do share a physical time. Rows can appear in several pairs.
    """
    if not 0 <= allowance < 1:
        raise ValueError("Match allowance must be in [0, 1)")
    pairs = []
    for default in rows:
        if not default.completed or default.estimator != MODES[0]:
            continue
        for guarded in rows:
            if not guarded.completed or guarded.estimator != MODES[1]:
                continue
            keys = ("case", "dtype", "order", "end", "feedback", "max_steps",
                    "first_step")
            if any(getattr(default, key) != getattr(guarded, key) for key in keys):
                continue
            if min(default.weight_slots, guarded.weight_slots) <= 0:
                continue
            mismatch = max(default.weight_slots, guarded.weight_slots) / min(
                default.weight_slots, guarded.weight_slots,
            ) - 1
            if mismatch <= allowance:
                pairs.append({
                    "case": default.case, "dtype": default.dtype,
                    "default_rtol": default.rtol, "subdivision_rtol": guarded.rtol,
                    "relative_work_mismatch": mismatch,
                    "default": asdict(default), "subdivision": asdict(guarded),
                })
    return pairs


def collect_rows(*, dtype, max_steps):
    return [run_case(case, mode, tol, dtype, max_steps=max_steps)
            for case in CASES for tol in TOLERANCES for mode in MODES]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--check", action="store_true",
                        help="require all runs complete and a work pair per case")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("max-steps must be positive")
    if args.dtype == "float64" and not jax.config.x64_enabled:
        parser.error("float64 stress audit requires JAX_ENABLE_X64=1")
    rows = collect_rows(dtype=args.dtype, max_steps=args.max_steps)
    pairs = match_work(rows)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(StressRow.__dataclass_fields__),
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    text = stream.getvalue()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / f"subdivision_stress_{args.dtype}"
        stem.with_suffix(".csv").write_text(text, encoding="utf-8", newline="")
        context = _context(args.dtype, args.max_steps, False)
        context["scope"] = "nonlinear/long-span work-proxy audit, not error bounds"
        context["provenance"] = current_provenance(config={
            "report": "subdivision_stress", "dtype": args.dtype, "order": 0.5,
            "cases": {case: asdict(problem) for case, problem in CASES.items()},
            "estimators": MODES, "tolerances": TOLERANCES, "atol_over_rtol": 0.01,
            "first_step": 0.01, "max_steps": args.max_steps,
            "match_allowance": MATCH_ALLOWANCE,
            "work_metric": "PECE evaluations * padded history buffer length",
        }).to_dict()
        name = "benchmarks/numerical/subdivision_stress.py"
        context["source_sha256"][name] = hashlib.sha256(
            (ROOT / name).read_bytes(),
        ).hexdigest()
        context["csv_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        context["matched_work_pairs"] = pairs
        context["unmatched_cases"] = sorted(set(CASES) - {p["case"] for p in pairs})
        stem.with_suffix(".json").write_text(
            json.dumps(context, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    sys.stdout.write(text)
    expected = {(case, mode, tol, args.dtype, args.max_steps)
                for case in CASES for mode in MODES for tol in TOLERANCES}
    actual = {(row.case, row.estimator, row.rtol, row.dtype, row.max_steps)
              for row in rows}
    complete = (
        len(rows) == len(expected) and actual == expected
        and all(row.completed for row in rows)
    )
    covered = {pair["case"] for pair in pairs} == set(CASES)
    return int(args.check and not (complete and covered))


if __name__ == "__main__":
    raise SystemExit(main())
