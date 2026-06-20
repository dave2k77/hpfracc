"""Regression test pinning operator outputs to a committed golden artifact.

Guards against silent numerical drift: the operators must reproduce the stored
``operator_outputs.csv`` for the fixed case. If an operator's numerics change
intentionally, regenerate with ``tests/regression/golden/_generate.py`` and
review the diff.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from hpfracc.ops import caputo, grunwald_letnikov

pytestmark = pytest.mark.regression

# Must match tests/regression/golden/_generate.py.
ALPHA = 0.5
DT = 0.05
N = 32
GOLDEN = Path(__file__).parent / "golden" / "operator_outputs.csv"


def _load_golden() -> dict[str, list[float]]:
    columns: dict[str, list[float]] = {"caputo": [], "grunwald_letnikov": []}
    with GOLDEN.open(newline="") as handle:
        for row in csv.DictReader(handle):
            columns["caputo"].append(float(row["caputo"]))
            columns["grunwald_letnikov"].append(float(row["grunwald_letnikov"]))
    return columns


def test_operator_outputs_match_golden(enable_x64) -> None:
    golden = _load_golden()
    ts = jnp.arange(N, dtype=jnp.float64) * DT
    x = ts**2

    actual = {
        "caputo": caputo(x, dt=DT, order=ALPHA),
        "grunwald_letnikov": grunwald_letnikov(x, dt=DT, order=ALPHA),
    }
    for name, values in actual.items():
        expected = jnp.asarray(golden[name])
        assert jnp.allclose(values, expected, rtol=1e-12, atol=1e-12), name
