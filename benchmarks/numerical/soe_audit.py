"""SOE approximation measurements and separate fitted-kernel execution scaling.

Accuracy --check requires complete finite measurements, not small approximation
errors. Scaling --check requires agreement and valid measurements, not speedup.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import gamma
from pathlib import Path
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gamma as jax_gamma

from benchmarks.numerical.adaptive_calibration import ROOT, _context
from hpfracc.config import current_provenance
from hpfracc.ops import caputo, grunwald_letnikov
from hpfracc.ops._soe import _fit_soe
from hpfracc.ops.kernels import _full_history_convolution

METHODS = ("legacy", "scaled_tail")
OPERATORS = {"caputo": caputo, "grunwald_letnikov": grunwald_letnikov}
ORDERS = (.3, .5, .9)
LENGTHS = (64, 256, 1024, 4096)
POLES = (8, 12)
SIGNALS = ("constant", "quadratic", "oscillatory", "transient")
SCALE_LENGTHS, DIMENSIONS, SCALE_POLES = (256, 1024, 4096), (1, 4), (8, 16)


def full_weights(alpha, n, operator):
    """Independent host float64 discrete reference coefficients."""
    if operator == "caputo":
        k = np.arange(1, n, dtype=np.float64)
        return np.r_[1., k**(1 - alpha) * np.expm1((1 - alpha) * np.log1p(1 / k))]
    weights = np.ones(n)
    for k in range(1, n):
        weights[k] = weights[k - 1] * (k - 1 - alpha) / k
    return weights


@lru_cache(maxsize=64)
def reference_case(n, dtype, operator, alpha):
    t = np.linspace(0., 1., n).astype(dtype)
    values = np.stack((np.ones(n), t**2, np.sin(8 * np.pi * t),
                       np.where(t >= .25, np.exp(-4 * (t - .25)), 0.)), axis=1)
    values = values.astype(dtype)
    signal = np.asarray(values, dtype=np.float64)
    scale = (n - 1)**alpha
    if operator == "caputo":
        signal = np.diff(signal, axis=0)
        scale /= gamma(2 - alpha)
    weights = full_weights(alpha, len(signal), operator)
    expected = np.stack([np.convolve(signal[:, i], weights)[:len(signal)]
                         for i in range(len(SIGNALS))], axis=1) * scale
    if operator == "caputo":
        expected = np.concatenate((np.zeros((1, len(SIGNALS))), expected), axis=0)
    analytic = 2 * t.astype(np.float64)**(2 - alpha) / gamma(3 - alpha)
    return values, expected, analytic


@lru_cache(maxsize=64)
def accuracy_runner(n, operator, method, poles):
    @jax.jit
    def run(values, alpha):
        result = OPERATORS[operator](
            values, dt=1 / (n - 1), order=alpha, history="soe", soe_fit=method,
            soe_poles=poles, return_info=True,
        )
        d = result.operator_info.diagnostics
        return (result.values, d["soe_kernel_max_abs_error"],
                d["soe_kernel_relative_l2_error"], d["soe_fit_finite"])
    return run


@dataclass(frozen=True, slots=True)
class AccuracyRow:
    dtype: str
    operator: str
    fit_method: str
    n_samples: int
    poles: int
    order: float
    target_kernel: str
    fit_finite: bool
    measurement_ok: bool
    kernel_max_abs_error: float | None
    kernel_relative_l2_error: float | None
    signal_max_abs_errors: tuple[float, ...] | None
    signal_relative_l2_errors: tuple[float | None, ...] | None
    quadratic_analytic_max_abs_error: float | None


def accuracy_case(n, operator, method, poles, order, dtype):
    alpha = float(np.asarray(order, dtype=dtype))
    values, expected, analytic = reference_case(n, dtype, operator, alpha)
    output, kernel_max, kernel_relative, finite = jax.device_get(
        accuracy_runner(n, operator, method, poles)(
            jnp.asarray(values), jnp.asarray(alpha, dtype=getattr(jnp, dtype)),
        ),
    )
    ok = bool(finite and output.shape == values.shape and str(output.dtype) == dtype
              and np.all(np.isfinite(output)) and np.isfinite(kernel_max)
              and np.isfinite(kernel_relative))
    maximum = relative = analytic_error = None
    if ok:
        errors = np.asarray(output, dtype=np.float64) - expected
        maximum = tuple(float(e) for e in np.max(np.abs(errors), axis=0))
        norms = np.linalg.norm(expected, axis=0)
        relative = tuple(float(e / norm) if norm > 0 else None
                         for e, norm in zip(np.linalg.norm(errors, axis=0), norms,
                                            strict=True))
        analytic_error = float(np.max(np.abs(output[:, 1] - analytic)))
    target_kernel = ("l1" if operator == "caputo" else
                     "gl_prefix" if method == "scaled_tail" else "gl")
    return AccuracyRow(dtype, operator, method, n, poles, alpha, target_kernel,
                       bool(finite), ok,
                       float(kernel_max) if np.isfinite(kernel_max) else None,
                       float(kernel_relative) if np.isfinite(kernel_relative) else None,
                       maximum, relative, analytic_error)


def collect_accuracy(dtype):
    return [accuracy_case(n, operator, method, poles, order, dtype)
            for operator in OPERATORS for n in LENGTHS for poles in POLES
            for method in METHODS for order in ORDERS]


def accuracy_passed(rows, dtype):
    expected = {(operator, n, poles, method, float(np.asarray(a, dtype=dtype)))
                for operator in OPERATORS for n in LENGTHS for poles in POLES
                for method in METHODS for a in ORDERS}
    actual = {(r.operator, r.n_samples, r.poles, r.fit_method, r.order) for r in rows}
    return (len(rows) == len(expected) and actual == expected
            and all(r.dtype == dtype and r.measurement_ok for r in rows))


def dense_fitted(values, alpha, poles):
    n = values.shape[0]
    fit = _fit_soe(alpha, n - 1, n_poles=poles, t_max=n / (n - 1),
                   kernel="l1", method="scaled_tail")
    tail = (fit.rhos[None, :] ** jnp.arange(n - 2)[:, None]) @ fit.coefficients
    weights = jnp.concatenate((jnp.ones(1, dtype=tail.dtype), tail))
    out = _full_history_convolution(values[1:] - values[:-1], weights)
    out /= jax_gamma(2 - alpha) * (1 / (n - 1))**alpha
    return jnp.concatenate((jnp.zeros_like(values[:1]), out), axis=0)


@dataclass(frozen=True, slots=True)
class ScalingRow:
    dtype: str
    application: str
    n_samples: int
    state_dim: int
    poles: int
    compile_seconds: float
    warmup_seconds: float
    sample_seconds: tuple[float, ...]
    median_seconds: float
    argument_bytes: int | None
    output_bytes: int | None
    temporary_bytes: int | None
    alias_bytes: int | None
    max_abs_difference: float
    agrees_with_dense: bool
    repeatable: bool


def collect_scaling(dtype, *, repeats=7, seed=1729, lengths=SCALE_LENGTHS,
                    dimensions=DIMENSIONS, poles_values=SCALE_POLES):
    prepared = []
    for n in lengths:
        for dimension in dimensions:
            t = jnp.linspace(0., 1., n, dtype=getattr(jnp, dtype))
            values = jnp.sin(3 * t[:, None]) * jnp.arange(1, dimension + 1)[None, :]
            alpha = jnp.asarray(.5, dtype=values.dtype)
            jax.block_until_ready((values, alpha))
            for poles in poles_values:
                reference = jax.block_until_ready(dense_fitted(values, alpha, poles))
                for application in ("dense_fitted", "recursive"):
                    def run(x, a, application=application, poles=poles, n=n):
                        if application == "dense_fitted":
                            return dense_fitted(x, a, poles)
                        return caputo(x, dt=1 / (n - 1), order=a, history="soe",
                                      soe_fit="scaled_tail", soe_poles=poles)

                    start = perf_counter()
                    compiled = jax.jit(run).lower(values, alpha).compile()
                    compile_time = perf_counter() - start
                    start = perf_counter()
                    warm = jax.block_until_ready(compiled(values, alpha))
                    warm_time = perf_counter() - start
                    tolerance = 2e-5 if dtype == "float32" else 2e-11
                    agrees = bool(np.allclose(warm, reference,
                                             rtol=tolerance, atol=tolerance))
                    difference = float(jnp.max(jnp.abs(warm - reference)))
                    memory = compiled.memory_analysis()
                    buffers = tuple(int(getattr(memory, key)) if memory else None
                                    for key in ("argument_size_in_bytes",
                                                "output_size_in_bytes",
                                                "temp_size_in_bytes",
                                                "alias_size_in_bytes"))
                    prepared.append({
                        "compiled": compiled, "inputs": (values, alpha), "warm": warm,
                        "samples": [], "repeatable": True,
                        "config": (dtype, application, n, dimension, poles,
                                   compile_time, warm_time, buffers, difference,
                                   agrees),
                    })
    rng = random.Random(seed)
    for _ in range(repeats):
        order = list(range(len(prepared)))
        rng.shuffle(order)
        for index in order:
            item = prepared[index]
            start = perf_counter()
            out = jax.block_until_ready(item["compiled"](*item["inputs"]))
            item["samples"].append(perf_counter() - start)
            item["repeatable"] &= bool(np.array_equal(out, item["warm"]))
    rows = []
    for item in prepared:
        (dt, app, n, dim, poles, compilation, warmup,
         buffers, error, agrees) = item["config"]
        samples = tuple(item["samples"])
        rows.append(ScalingRow(dt, app, n, dim, poles, compilation, warmup, samples,
                               median(samples), *buffers, error, agrees,
                               item["repeatable"]))
    return rows


def scaling_passed(rows, *, dtype, repeats, lengths=SCALE_LENGTHS,
                   dimensions=DIMENSIONS, poles_values=SCALE_POLES):
    expected = {(n, dimension, poles, application) for n in lengths
                for dimension in dimensions for poles in poles_values
                for application in ("dense_fitted", "recursive")}
    actual = {(r.n_samples, r.state_dim, r.poles, r.application) for r in rows}
    return (len(rows) == len(expected) and actual == expected and all(
        r.dtype == dtype and r.agrees_with_dense and r.repeatable
        and len(r.sample_seconds) == repeats
        and all(np.isfinite(t) and t > 0 for t in (
            r.compile_seconds, r.warmup_seconds, *r.sample_seconds,
        )) and all(b is not None and b >= 0 for b in (
            r.argument_bytes, r.output_bytes, r.temporary_bytes, r.alias_bytes,
        )) for r in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("accuracy", "scaling"), default="accuracy")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.dtype == "float64" and not jax.config.x64_enabled:
        parser.error("float64 audit requires JAX_ENABLE_X64=1")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.mode == "accuracy":
        rows = collect_accuracy(args.dtype)
        passed = accuracy_passed(rows, args.dtype)
        row_type = AccuracyRow
    else:
        rows = collect_scaling(args.dtype, repeats=args.repeats, seed=args.seed)
        passed = scaling_passed(rows, dtype=args.dtype, repeats=args.repeats)
        row_type = ScalingRow
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(row_type.__dataclass_fields__),
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    text = stream.getvalue()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / f"soe_{args.mode}_{args.dtype}"
        stem.with_suffix(".csv").write_text(text, encoding="utf-8", newline="")
        context = _context(args.dtype, 0, False)
        context["scope"] = f"SOE {args.mode}; not universal error or performance bounds"
        context["provenance"] = current_provenance(config={
            "report": "soe_audit", "mode": args.mode, "dtype": args.dtype,
            "accuracy_lengths": LENGTHS, "orders": ORDERS, "poles": POLES,
            "fit_methods": METHODS, "signals": SIGNALS, "operators": list(OPERATORS),
            "scaling_lengths": SCALE_LENGTHS, "dimensions": DIMENSIONS,
            "scaling_poles": SCALE_POLES, "scaling_order": .5,
            "scaling_fit": "scaled_tail", "repeats": args.repeats, "seed": args.seed,
            "memory": "XLA compiler buffer accounting; not peak RSS/device memory",
        }).to_dict()
        for name in ("benchmarks/numerical/soe_audit.py", "src/hpfracc/ops/_soe.py",
                     "src/hpfracc/ops/fractional.py", "src/hpfracc/ops/kernels.py",
                     "docs/validation/audits/soe-2026-09-14/protocol.json"):
            context["source_sha256"][name] = hashlib.sha256(
                (ROOT / name).read_bytes(),
            ).hexdigest()
        context["csv_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        context["rows"] = [asdict(row) for row in rows]
        stem.with_suffix(".json").write_text(
            json.dumps(context, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    sys.stdout.write(text)
    return int(args.check and not passed)


if __name__ == "__main__":
    raise SystemExit(main())
