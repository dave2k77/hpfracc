"""Paired direct/stable coefficient and implicit-solver precision measurements."""

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

from benchmarks.numerical import implicit_validation as implicit
from benchmarks.numerical.adaptive_calibration import ROOT, _context
from hpfracc.config import current_provenance
from hpfracc.solvers import ImplicitPredictorCorrector, simulate
from hpfracc.solvers._weights import _stable_pece_weights
from hpfracc.solvers.predictor_corrector import _pece_weights

EVALUATIONS = ("direct", "stable")
ORDERS = (.01, .1, .3, .5, .8, .9, .99)
STEPS = (1, 3, 16, 256, 4096)
TOLERANCES = {"float32": 5e-6, "float64": 5e-12}


@lru_cache(maxsize=64)
def reference_row(step, alpha):
    """Independent unit-panel quadrature, without allocating a square matrix."""
    nodes, weights = np.polynomial.legendre.leggauss(32)
    fraction = (nodes + 1) / 2
    lag = np.arange(1, step)[:, None] + fraction
    integrand = lag**(alpha - 1) * (weights / 2)
    integral = np.r_[1 / alpha, np.sum(integrand, axis=1)]
    left = np.r_[1 / (alpha + 1), integrand @ fraction]
    right = np.r_[1 / (alpha * (alpha + 1)), integrand @ (1 - fraction)]
    predictor = alpha * integral[::-1]
    corrector = alpha * (alpha + 1) * (left[::-1] + np.r_[0., right[::-1][:-1]])
    return predictor, corrector


@dataclass(frozen=True, slots=True)
class CoefficientRow:
    dtype: str
    weight_evaluation: str
    order: float
    step: int
    measurement_ok: bool
    positive_history_weights: bool
    max_relative_predictor_error: float
    max_relative_corrector_error: float
    max_absolute_corrector_error: float
    future_nonzero_count: int


def coefficient_case(dtype, evaluation, alpha, step):
    dt = getattr(jnp, dtype)
    order = jnp.asarray(alpha, dtype=dt)
    indices = jnp.arange(step + 3)
    fn = _pece_weights if evaluation == "direct" else _stable_pece_weights
    pred, corr = jax.jit(lambda a: fn(jnp.asarray(step), indices, a, dt))(order)
    pred, corr = np.asarray(pred), np.asarray(corr)
    expected_p, expected_c = reference_row(step, float(order))
    future = int(np.count_nonzero(pred[step:]) + np.count_nonzero(corr[step:]))
    valid = bool(pred.shape == corr.shape == (step + 3,)
                 and str(pred.dtype) == str(corr.dtype) == dtype
                 and np.all(np.isfinite(pred)) and np.all(np.isfinite(corr))
                 and future == 0)
    return CoefficientRow(
        dtype, evaluation, float(order), step, valid,
        bool(np.all(pred[:step] > 0) and np.all(corr[:step] > 0)),
        float(np.max(np.abs(pred[:step] / expected_p - 1))),
        float(np.max(np.abs(corr[:step] / expected_c - 1))),
        float(np.max(np.abs(corr[:step] - expected_c))), future,
    )


def collect_coefficients(dtype):
    return [coefficient_case(dtype, evaluation, a, step)
            for a in ORDERS for step in STEPS for evaluation in EVALUATIONS]


def coefficients_passed(rows, dtype):
    expected = {(float(np.asarray(a, dtype=dtype)), step, evaluation)
                for a in ORDERS for step in STEPS for evaluation in EVALUATIONS}
    actual = {(r.order, r.step, r.weight_evaluation) for r in rows}
    return (len(rows) == len(expected) and actual == expected and all(
        r.dtype == dtype and r.measurement_ok
        and (r.weight_evaluation == "direct" or (
            r.positive_history_weights
            and r.max_relative_predictor_error <= TOLERANCES[dtype]
            and r.max_relative_corrector_error <= TOLERANCES[dtype]
        )) for r in rows))


@lru_cache(maxsize=64)
def solver_runner(case, n, dtype, evaluation):
    scales = jnp.asarray(implicit.SCALES[case], dtype=getattr(jnp, dtype))
    times = jnp.linspace(0., 1., n, dtype=scales.dtype)

    def model(t, state, params, *, rng_key=None, inputs=None):
        return implicit.dynamics(case, t, state, params, rng_key=rng_key, inputs=inputs)

    @jax.jit
    def run(initial, strength, alpha):
        result = simulate(
            model=model, ts=times, initial_state=initial,
            params=(strength, alpha, scales),
            solver=ImplicitPredictorCorrector(
                dt=1/(n-1), order=alpha, newton_iterations=8,
                newton_rtol=implicit.RTOL[dtype],
                newton_atol=.01 * implicit.RTOL[dtype], weight_evaluation=evaluation,
            ),
        )
        d = result.solver_info.diagnostics
        return result.ts, result.latent_state, {key: d[key] for key in implicit.FIELDS}
    return run


@dataclass(frozen=True, slots=True)
class SolverRow:
    weight_evaluation: str
    result: implicit.ImplicitRow

    def to_dict(self):
        return {"weight_evaluation": self.weight_evaluation, **asdict(self.result)}


def solver_case(case, n, strength, alpha, dtype, evaluation):
    scales = np.asarray(implicit.SCALES[case], dtype=dtype)
    initial = implicit.reference(case, np.asarray(0.), strength, scales).astype(dtype)
    a = jnp.asarray(alpha, dtype=getattr(jnp, dtype))
    rate = jnp.asarray(strength, dtype=a.dtype)
    output = solver_runner(case, n, dtype, evaluation)(jnp.asarray(initial), rate, a)
    return SolverRow(evaluation, implicit.summarize(
        output, case, dtype, float(a), float(rate), n, 8,
    ))


def collect_solvers(dtype):
    return [solver_case(case, n, k, a, dtype, evaluation)
            for case in implicit.SCALES for a in implicit.orders(case)
            for k in implicit.STRENGTHS for n in implicit.LENGTHS
            for evaluation in EVALUATIONS]


def solvers_passed(rows, dtype):
    expected = {(case, float(np.asarray(a, dtype=dtype)), k, n, evaluation)
                for case in implicit.SCALES for a in implicit.orders(case)
                for k in implicit.STRENGTHS for n in implicit.LENGTHS
                for evaluation in EVALUATIONS}
    actual = {(r.result.case, r.result.order, r.result.strength,
               r.result.n_samples, r.weight_evaluation) for r in rows}
    return (len(rows) == len(expected) and actual == expected
            and all(r.result.dtype == dtype and r.result.newton_iterations == 8
                    and r.result.measurement_ok for r in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("coefficients", "solvers"),
                        default="coefficients")
    parser.add_argument("--dtype", choices=tuple(TOLERANCES), default="float32")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.dtype == "float64" and not jax.config.x64_enabled:
        parser.error("float64 validation requires JAX_ENABLE_X64=1")
    if args.mode == "coefficients":
        rows = collect_coefficients(args.dtype)
        passed = coefficients_passed(rows, args.dtype)
        payloads = [asdict(row) for row in rows]
    else:
        rows = collect_solvers(args.dtype)
        passed = solvers_passed(rows, args.dtype)
        payloads = [row.to_dict() for row in rows]
    stream = io.StringIO(newline="")
    fields = (list(CoefficientRow.__dataclass_fields__) if args.mode == "coefficients"
              else ["weight_evaluation", *implicit.ImplicitRow.__dataclass_fields__])
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(payloads)
    text = stream.getvalue()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / f"uniform_{args.mode}_{args.dtype}"
        stem.with_suffix(".csv").write_text(text, encoding="utf-8", newline="")
        context = _context(args.dtype, 0, False)
        context["scope"] = (
            "uniform weight precision; same method, not global error control"
        )
        context["provenance"] = current_provenance(config={
            "report": "uniform_weight_validation", "mode": args.mode,
            "dtype": args.dtype, "evaluations": EVALUATIONS,
            "coefficient_orders": ORDERS, "coefficient_steps": STEPS,
            "coefficient_tolerance": TOLERANCES[args.dtype],
            "solver_cases": implicit.SCALES, "solver_orders": implicit.ORDERS,
            "half_decay_order": .5, "solver_strengths": implicit.STRENGTHS,
            "solver_lengths": implicit.LENGTHS, "newton_iterations": 8,
            "newton_rtol": implicit.RTOL[args.dtype],
            "newton_atol": .01 * implicit.RTOL[args.dtype],
        }).to_dict()
        for name in ("benchmarks/numerical/uniform_weight_validation.py",
                     "benchmarks/numerical/implicit_validation.py",
                     "src/hpfracc/solvers/_weights.py",
                     "docs/validation/audits/uniform-weights-2026-09-15/protocol.json"):
            context["source_sha256"][name] = hashlib.sha256(
                (ROOT / name).read_bytes(),
            ).hexdigest()
        context["csv_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        context["rows"] = payloads
        stem.with_suffix(".json").write_text(
            json.dumps(context, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    sys.stdout.write(text)
    return int(args.check and not passed)


if __name__ == "__main__":
    raise SystemExit(main())
