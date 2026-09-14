"""Windowed versus dense-truncated Caputo: timings and compiler buffer accounting.

Memory figures come from XLA executable.memory_analysis(), not a sampled process
or device high-water mark. Run serially without other numerical workloads.
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
from pathlib import Path
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gamma

from benchmarks.numerical.adaptive_calibration import ROOT, _context
from hpfracc.config import current_provenance
from hpfracc.ops import caputo
from hpfracc.ops.fractional import _l1_weights
from hpfracc.ops.kernels import _full_history_convolution

METHODS = ("dense_truncated", "windowed")


def dense_truncated(values, alpha, *, dt, window):
    """Pre-change short-memory algorithm, with a dynamic order input."""
    n = values.shape[0] - 1
    weights = _l1_weights(alpha, n)
    weights = jnp.where(jnp.arange(n) < max(1, min(window, n)), weights, 0.)
    tail = _full_history_convolution(values[1:] - values[:-1], weights)
    tail = tail / (gamma(2 - alpha) * dt**alpha)
    return jnp.concatenate((jnp.zeros_like(values[:1]), tail), axis=0)


@dataclass(frozen=True, slots=True)
class WindowRow:
    method: str
    dtype: str
    n_samples: int
    state_dim: int
    window_steps: int
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


def generate_rows(*, dtype="float32", lengths=(256, 1024, 4096),
                  dimensions=(1, 4), windows=(8, 32), repeats=7, seed=1729):
    if dtype == "float64" and not jax.config.x64_enabled:
        raise ValueError("float64 benchmark requires JAX_ENABLE_X64=1")
    if repeats < 1 or min(lengths) < 2 or min(dimensions) < 1 or min(windows) < 1:
        raise ValueError("Expected positive sizes/repeats and lengths >= 2")
    prepared = []
    for n in lengths:
        for dimension in dimensions:
            dt = 1. / (n - 1)
            t = jnp.linspace(0., 1., n, dtype=getattr(jnp, dtype))
            values = jnp.sin(3 * t[:, None]) * jnp.arange(1, dimension + 1)[None, :]
            alpha = jnp.asarray(.5, dtype=values.dtype)
            jax.block_until_ready((values, alpha))
            for window in windows:
                reference = dense_truncated(values, alpha, dt=dt, window=window)
                jax.block_until_ready(reference)
                for method in METHODS:
                    def run(x, a, method=method, dt=dt, window=window):
                        if method == "dense_truncated":
                            return dense_truncated(x, a, dt=dt, window=window)
                        return caputo(x, dt=dt, order=a, history="short_memory",
                                      window_steps=window)

                    start = perf_counter()
                    compiled = jax.jit(run).lower(values, alpha).compile()
                    compile_seconds = perf_counter() - start
                    start = perf_counter()
                    warm = jax.block_until_ready(compiled(values, alpha))
                    warm_seconds = perf_counter() - start
                    difference = float(jnp.max(jnp.abs(warm - reference)))
                    tolerance = 2e-5 if dtype == "float32" else 2e-12
                    agrees = bool(np.allclose(warm, reference,
                                             rtol=tolerance, atol=tolerance))
                    memory = compiled.memory_analysis()
                    fields = ("argument_size_in_bytes", "output_size_in_bytes",
                              "temp_size_in_bytes", "alias_size_in_bytes")
                    buffers = tuple(int(getattr(memory, f)) if memory else None
                                    for f in fields)
                    prepared.append({
                        "compiled": compiled, "inputs": (values, alpha),
                        "warm": warm, "samples": [], "repeatable": True,
                        "row": (method, dtype, n, dimension, window, compile_seconds,
                                warm_seconds, buffers, difference, agrees),
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
            # Host transfer and repeatability checks are outside timed intervals.
            item["repeatable"] &= bool(np.array_equal(out, item["warm"]))
    rows = []
    for item in prepared:
        (method, dt, n, dim, window, compile_s, warm_s,
         buffers, error, agrees) = item["row"]
        samples = tuple(item["samples"])
        rows.append(WindowRow(method, dt, n, dim, window, compile_s, warm_s,
                              samples, median(samples), *buffers, error, agrees,
                              item["repeatable"]))
    return rows


def rows_passed(rows, *, repeats):
    return bool(rows) and all(
        r.agrees_with_dense and r.repeatable and len(r.sample_seconds) == repeats
        and all(np.isfinite(s) and s > 0 for s in (
            r.compile_seconds, r.warmup_seconds, *r.sample_seconds,
        )) and all(b is not None and b >= 0 for b in (
            r.argument_bytes, r.output_bytes, r.temporary_bytes, r.alias_bytes,
        )) for r in rows
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.dtype == "float64" and not jax.config.x64_enabled:
        parser.error("float64 benchmark requires JAX_ENABLE_X64=1")
    rows = generate_rows(dtype=args.dtype, repeats=args.repeats, seed=args.seed)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(WindowRow.__dataclass_fields__),
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    text = stream.getvalue()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / f"windowed_history_{args.dtype}"
        stem.with_suffix(".csv").write_text(text, encoding="utf-8", newline="")
        context = _context(args.dtype, 0, False)
        context["scope"] = (
            "Caputo forward timings and XLA buffer accounting, not peak RSS"
        )
        context["provenance"] = current_provenance(config={
            "report": "windowed_history", "dtype": args.dtype, "order": .5,
            "lengths": [256, 1024, 4096], "dimensions": [1, 4], "windows": [8, 32],
            "repeats": args.repeats, "seed": args.seed,
            "methods": METHODS, "memory_source": "XLA executable.memory_analysis",
        }).to_dict()
        for name in ("benchmarks/numerical/windowed_history.py",
                     "src/hpfracc/ops/fractional.py", "src/hpfracc/ops/kernels.py"):
            context["source_sha256"][name] = hashlib.sha256(
                (ROOT / name).read_bytes(),
            ).hexdigest()
        context["csv_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        context["rows"] = [asdict(row) for row in rows]
        stem.with_suffix(".json").write_text(
            json.dumps(context, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    sys.stdout.write(text)
    return int(args.check and not (len(rows) == 24
                                  and rows_passed(rows, repeats=args.repeats)))


if __name__ == "__main__":
    raise SystemExit(main())
