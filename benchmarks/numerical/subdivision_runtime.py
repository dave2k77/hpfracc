"""Synchronized compiled-solver timings; local evidence, not speed guarantees."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.numerical import subdivision_stress as stress
from benchmarks.numerical.adaptive_calibration import ROOT, _context
from hpfracc.config import current_provenance
from hpfracc.solvers import AdaptivePredictorCorrector, simulate
from hpfracc.solvers.base import SimulationResult, SolverInfo

DIAGNOSTICS = (
    "n_nodes", "n_accepted", "n_rejected", "n_attempted", "success", "reached_t_end",
    "max_steps_reached", "nonfinite_detected", "step_size_underflow", "valid_mask",
)

# Fixed before the dense experiment; do not tune this grid to observed errors.
DENSE_TOLERANCES = (
    1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1.5e-3, 1e-3,
    7e-4, 5e-4, 3e-4, 2e-4, 1.5e-4, 1e-4,
)


def sweep_tolerances(sweep):
    if sweep == "coarse":
        return stress.TOLERANCES
    if sweep == "dense":
        return DENSE_TOLERANCES
    raise ValueError("Unknown runtime sweep")


@dataclass(frozen=True, slots=True)
class RuntimeRow:
    measurement: stress.StressRow
    compile_seconds: float
    warmup_seconds: float
    samples_seconds: tuple[float, ...]
    stable_repeats: bool

    @property
    def median_seconds(self):
        return median(self.samples_seconds)


def timed_call(fn, args):
    """Time host dispatch through completion of every numerical output leaf."""
    start = time.perf_counter()
    result = fn(*args)
    jax.block_until_ready(result)
    return result, time.perf_counter() - start


def prepare(case, mode, rtol, dtype, max_steps):
    dt = getattr(jnp, dtype)
    order = jnp.asarray(0.5, dtype=dt)
    solver = AdaptivePredictorCorrector(
        order=order, first_step=0.01, rtol=rtol, atol=rtol * 0.01,
        max_steps=max_steps, error_estimator=mode,
    )

    def model(t, state, params, *, rng_key=None, inputs=None):
        del params, rng_key, inputs
        return stress.rhs(case, t, state)

    def solve(initial, bounds):
        result = simulate(model=model, ts=bounds, solver=solver,
                          initial_state=initial, params=None)
        return (result.ts, result.latent_state,
                {key: result.solver_info.diagnostics[key] for key in DIAGNOSTICS})

    # Dynamic inputs prevent constant-folding a closed-over solved trajectory.
    args = (jnp.asarray(1., dtype=dt),
            jnp.asarray([0., stress.CASES[case].end], dtype=dt))
    jax.block_until_ready(args)
    start = time.perf_counter()
    compiled = jax.jit(solve).lower(*args).compile()
    compile_seconds = time.perf_counter() - start
    output, warmup_seconds = timed_call(compiled, args)
    return compiled, args, output, compile_seconds, warmup_seconds


def _host_tree(output):
    return tuple(np.asarray(leaf).copy() for leaf in jax.tree.leaves(output))


def _same_output(output, expected):
    actual = _host_tree(output)
    return len(actual) == len(expected) and all(
        np.array_equal(a, b, equal_nan=True)
        for a, b in zip(actual, expected, strict=True)
    )


def collect_rows(*, dtype, max_steps, repeats, seed=1729,
                 cases=tuple(stress.CASES), tolerances=stress.TOLERANCES):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if max_steps < 1 or dtype not in ("float32", "float64"):
        raise ValueError("Invalid max_steps or dtype")
    if dtype == "float64" and not jax.config.x64_enabled:
        raise ValueError("float64 timing requires JAX_ENABLE_X64=1")
    if not cases or any(case not in stress.CASES for case in cases):
        raise ValueError("Unknown or empty cases")
    keys = [(case, mode, tol) for case in cases
            for tol in tolerances for mode in stress.MODES]
    prepared, reference_outputs = [], []
    # Compile/warm all cases before the timed rounds. No concurrent benchmarking.
    for case, mode, tol in keys:
        item = prepare(case, mode, tol, dtype, max_steps)
        prepared.append(item)
        reference_outputs.append(_host_tree(item[2]))
    samples = [[] for _ in keys]
    stable = [True for _ in keys]
    rng = random.Random(seed)
    for _ in range(repeats):
        order = list(range(len(keys)))
        rng.shuffle(order)
        for i in order:
            compiled, args, _, _, _ = prepared[i]
            output, elapsed = timed_call(compiled, args)
            samples[i].append(elapsed)
            # Host transfer/validation intentionally lies outside timed regions.
            stable[i] &= _same_output(output, reference_outputs[i])
    rows = []
    for i, (case, mode, tol) in enumerate(keys):
        _, _, (ts, values, diagnostics), compilation, warmup = prepared[i]
        result = SimulationResult(
            ts=ts, latent_state=values,
            solver_info=SolverInfo(
                name="adaptive_predictor_corrector", method="caputo_adaptive_pece",
                diagnostics=diagnostics,
            ),
        )
        measurement = stress.summarize_result(
            result, case, mode, tol, dtype, max_steps=max_steps,
        )
        rows.append(RuntimeRow(measurement, compilation, warmup,
                               tuple(samples[i]), bool(stable[i])))
    return rows


def valid_timing(row, repeats=None):
    return bool(
        row.stable_repeats and row.samples_seconds
        and (repeats is None or len(row.samples_seconds) == repeats)
        and all(np.isfinite(x) and x > 0 for x in (
            row.compile_seconds, row.warmup_seconds, *row.samples_seconds,
        ))
    )


def match_runtime(rows, *, allowance=stress.MATCH_ALLOWANCE):
    """All complete compatible pairs within the median-runtime allowance."""
    if not 0 <= allowance < 1:
        raise ValueError("Match allowance must be in [0, 1)")
    pairs = []
    for a in rows:
        if not valid_timing(a) or not a.measurement.completed:
            continue
        if a.measurement.estimator != stress.MODES[0]:
            continue
        for b in rows:
            if not valid_timing(b) or not b.measurement.completed:
                continue
            if b.measurement.estimator != stress.MODES[1]:
                continue
            keys = ("case", "dtype", "order", "end", "feedback", "max_steps",
                    "first_step")
            if any(getattr(a.measurement, key) != getattr(b.measurement, key)
                   for key in keys):
                continue
            mismatch = max(a.median_seconds, b.median_seconds) / min(
                a.median_seconds, b.median_seconds,
            ) - 1
            if mismatch <= allowance:
                pairs.append({
                    "case": a.measurement.case,
                    "default_rtol": a.measurement.rtol,
                    "subdivision_rtol": b.measurement.rtol,
                    "relative_runtime_mismatch": mismatch,
                    "default_median_seconds": a.median_seconds,
                    "subdivision_median_seconds": b.median_seconds,
                    "default_max_abs_error": a.measurement.max_abs_error,
                    "subdivision_max_abs_error": b.measurement.max_abs_error,
                    "default_endpoint_abs_error": a.measurement.endpoint_abs_error,
                    "subdivision_endpoint_abs_error": b.measurement.endpoint_abs_error,
                })
    return pairs


def checked(rows, *, dtype, max_steps, repeats, tolerances=stress.TOLERANCES):
    expected = {(case, mode, tol, dtype, max_steps) for case in stress.CASES
                for mode in stress.MODES for tol in tolerances}
    actual = {(r.measurement.case, r.measurement.estimator, r.measurement.rtol,
               r.measurement.dtype, r.measurement.max_steps) for r in rows}
    return (len(rows) == len(expected) and actual == expected
            and all(valid_timing(r, repeats) and r.measurement.completed for r in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--sweep", choices=("coarse", "dense"), default="coarse")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true",
                        help="require a complete sweep of successful stable timed runs")
    args = parser.parse_args(argv)
    if args.max_steps < 1 or args.repeats < 1:
        parser.error("max-steps and repeats must be positive")
    if args.dtype == "float64" and not jax.config.x64_enabled:
        parser.error("float64 timing requires JAX_ENABLE_X64=1")
    tolerances = sweep_tolerances(args.sweep)
    rows = collect_rows(dtype=args.dtype, max_steps=args.max_steps,
                        repeats=args.repeats, seed=args.seed, tolerances=tolerances)
    fields = [*stress.StressRow.__dataclass_fields__, "compile_seconds",
              "warmup_seconds", "repeats", "median_seconds", "min_seconds",
              "max_seconds", "stable_repeats"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({**asdict(row.measurement),
                         "compile_seconds": row.compile_seconds,
                         "warmup_seconds": row.warmup_seconds,
                         "repeats": len(row.samples_seconds),
                         "median_seconds": row.median_seconds,
                         "min_seconds": min(row.samples_seconds),
                         "max_seconds": max(row.samples_seconds),
                         "stable_repeats": row.stable_repeats})
    text = stream.getvalue()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / f"subdivision_runtime_{args.dtype}"
        stem.with_suffix(".csv").write_text(text, encoding="utf-8", newline="")
        context = _context(args.dtype, args.max_steps, False)
        context["scope"] = "local synchronized compiled timings, not speed guarantees"
        context["sweep"] = args.sweep
        context["tolerances"] = list(tolerances)
        context["provenance"] = current_provenance(config={
            "report": "subdivision_runtime", "dtype": args.dtype,
            "max_steps": args.max_steps, "repeats": args.repeats, "seed": args.seed,
            "cases": {k: asdict(v) for k, v in stress.CASES.items()},
            "estimators": stress.MODES, "tolerances": tolerances, "sweep": args.sweep,
            "order": 0.5, "first_step": 0.01, "atol_over_rtol": 0.01,
            "match_allowance": stress.MATCH_ALLOWANCE,
            "timing": "host dispatch plus block_until_ready of all output leaves",
            "compile": "lower+compile only; not a cold-cache guarantee",
            "schedule": "one warmup per case, seeded shuffled serial rounds",
        }).to_dict()
        context["runtime_environment"] = {key: os.environ.get(key) for key in (
            "JAX_PLATFORMS", "XLA_FLAGS", "OMP_NUM_THREADS", "JAX_NUM_THREADS",
        )}
        for name in ("benchmarks/numerical/subdivision_runtime.py",
                     "benchmarks/numerical/subdivision_stress.py"):
            context["source_sha256"][name] = hashlib.sha256(
                (ROOT / name).read_bytes(),
            ).hexdigest()
        context["csv_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        context["rows"] = [asdict(row) for row in rows]
        context["matched_runtime_pairs"] = match_runtime(rows)
        context["matched_work_pairs"] = stress.match_work([r.measurement for r in rows])
        stem.with_suffix(".json").write_text(
            json.dumps(context, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    sys.stdout.write(text)
    return int(args.check and not checked(
        rows, dtype=args.dtype, max_steps=args.max_steps, repeats=args.repeats,
        tolerances=tolerances,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
