from __future__ import annotations

import io

import pytest

pytest.importorskip("jax")

from benchmarks.numerical.operator_scaling import generate_rows, write_csv


def test_operator_scaling_generates_expected_rows() -> None:
    rows = generate_rows(n_steps_values=(8,), state_dims=(1, 2), repeats=1)
    assert len(rows) == 4
    assert {row.operator for row in rows} == {"caputo", "grunwald_letnikov"}
    assert {row.output_shape for row in rows} == {(8, 1), (8, 2)}
    assert all(row.seconds >= 0.0 for row in rows)


def test_operator_scaling_writes_csv() -> None:
    rows = generate_rows(n_steps_values=(8,), state_dims=(1,), repeats=1)
    stream = io.StringIO()
    write_csv(rows, stream)
    output = stream.getvalue()
    assert output.startswith("operator,n_steps,state_dim,order,seconds,output_shape")
    assert "caputo,8,1" in output

