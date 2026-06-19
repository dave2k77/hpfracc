"""Performance-tier scaling guard for the full-history operator.

Deselected by default (``-m "not performance"`` in pyproject); run explicitly
with ``uv run python -m pytest -m performance``. Timing tests are
machine-dependent and must not gate the unit suite.

The full-history Caputo convolution is ``O(n**2)``. This fits an empirical
exponent across grid sizes and asserts it stays clearly sub-cubic, catching
accidental algorithmic regressions (e.g. a dense path going ``O(n**3)``) without
pinning a brittle absolute wall-clock budget. Once the opt-in ``fft`` history
path lands (WS-1), add a direct ``fft``-vs-``full`` comparison here.
"""

from __future__ import annotations

import math
import time

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from hpfracc.ops import caputo

pytestmark = pytest.mark.performance

_SIZES = (256, 512, 1024, 2048)
_REPS = 5


def _median_time(fn, x) -> float:
    fn(x).block_until_ready()  # warm up / compile
    samples = []
    for _ in range(_REPS):
        start = time.perf_counter()
        fn(x).block_until_ready()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


def test_full_history_caputo_is_subcubic() -> None:
    run = jax.jit(lambda x: caputo(x, dt=0.01, order=0.5))
    logs_n = []
    logs_t = []
    for n in _SIZES:
        x = jnp.arange(n, dtype=jnp.float32) * 0.01
        logs_n.append(math.log(n))
        logs_t.append(math.log(_median_time(run, x)))

    # Least-squares slope of log(time) vs log(n) -> empirical scaling exponent.
    mean_n = sum(logs_n) / len(logs_n)
    mean_t = sum(logs_t) / len(logs_t)
    cov = sum(
        (a - mean_n) * (b - mean_t) for a, b in zip(logs_n, logs_t, strict=True)
    )
    var = sum((a - mean_n) ** 2 for a in logs_n)
    exponent = cov / var

    assert exponent < 2.7, f"full-history Caputo scaled as ~O(n^{exponent:.2f})"
