"""Isolated pure-JAX CPU probe for mixed-allocation float64 convolution crashes.

The parent survives a native child failure and records its output/exit status.
This module imports no HPFRACC numerical code and is not a timing benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path


def child(kernel, repeats):
    import faulthandler

    import jax
    import jax.numpy as jnp
    import numpy as np

    faulthandler.enable()
    print(json.dumps({
        "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
        "x64_enabled": jax.config.x64_enabled,
    }), flush=True)
    if not jax.config.x64_enabled:
        raise ValueError("Probe requires JAX_ENABLE_X64=1")

    def dense(x, amplitude, window):
        n = len(x)
        weights = jnp.ones(n, dtype=x.dtype) * amplitude
        weights = jnp.where(jnp.arange(n) < window, weights, 0.)
        lag = jnp.arange(n)[:, None] - jnp.arange(n)[None, :]
        matrix = jnp.where(lag >= 0, weights[jnp.clip(lag, 0, n - 1)], 0.)
        return matrix @ x

    def convolve(x, amplitude, window):
        weights = jnp.ones(window, dtype=x.dtype) * amplitude
        if kernel == "convolution":
            return jax.vmap(lambda v: jnp.convolve(v, weights)[:len(x)],
                            in_axes=1, out_axes=1)(x)
        indices = jnp.arange(len(x))

        def add_lag(total, lag):
            source = indices - lag
            shifted = jnp.where((source >= 0)[:, None], x[jnp.maximum(source, 0)], 0.)
            return total + weights[lag] * shifted, None

        return jax.lax.scan(add_lag, jnp.zeros_like(x), jnp.arange(window))[0]

    prepared = []
    for name, n, window in (("dense", 256, 8), (kernel, 4096, 32)):
        print(f"prepare {name} shape=({n},4) window={window}", flush=True)
        x = jnp.sin(3 * jnp.linspace(0., 1., n, dtype=jnp.float64)[:, None])
        x = x * jnp.arange(1, 5)[None, :]
        amplitude = jnp.asarray(.5, dtype=x.dtype)
        # Eager dense reference deliberately exercises the allocator before the
        # compiled convolution. This is part of the reproducer, not timed work.
        expected = jax.block_until_ready(dense(x, amplitude, window))
        fn = dense if name == "dense" else convolve
        compiled = jax.jit(lambda x, a, fn=fn, w=window: fn(x, a, w)).lower(
            x, amplitude,
        ).compile()
        print(f"warmup {name}", flush=True)
        warm = jax.block_until_ready(compiled(x, amplitude))
        np.testing.assert_allclose(warm, expected, rtol=2e-13, atol=2e-13)
        prepared.append((name, compiled, x, amplitude, warm))
    rng = random.Random(1729)
    for repeat in range(repeats):
        order = list(range(len(prepared)))
        rng.shuffle(order)
        for index in order:
            name, compiled, x, amplitude, warm = prepared[index]
            print(f"repeat={repeat} execute={name}", flush=True)
            result = jax.block_until_ready(compiled(x, amplitude))
            np.testing.assert_array_equal(result, warm)
    print("completed; all repeated outputs equal", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=("convolution", "scan"),
                        default="convolution")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.processes < 1:
        parser.error("repeats and processes must be positive")
    if args.child:
        child(args.kernel, args.repeats)
        return 0
    command = [sys.executable, "-u", "-X", "faulthandler",
               str(Path(__file__).resolve()),
               "--child", "--kernel", args.kernel, "--repeats", str(args.repeats)]
    env = {**os.environ, "JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu"}
    runs = []
    # Native failures are allocation-sensitive. Retain every preselected process
    # outcome, including successes, rather than stopping at a favorable result.
    for index in range(args.processes):
        result = subprocess.run(command, capture_output=True, text=True, env=env,
                                timeout=120, check=False)
        runs.append({
            "index": index, "returncode": result.returncode,
            "returncode_hex": hex(result.returncode & 0xffffffff),
            "stdout": result.stdout, "stderr": result.stderr,
        })
    payload = {
        "scope": "pure-JAX mixed-allocation diagnostic, not a performance report",
        "kernel": args.kernel, "repeats": args.repeats, "processes": args.processes,
        "platform": platform.platform(), "python": platform.python_version(),
        "command": command, "runs": runs,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return int(any(run["returncode"] != 0 for run in runs))


if __name__ == "__main__":
    raise SystemExit(main())
