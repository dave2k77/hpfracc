"""Independent report references, incomplete measurements, and matrix gates."""

from dataclasses import replace
from math import gamma

import jax.numpy as jnp
import numpy as np

from benchmarks.numerical import soe_audit as audit


def test_reference_constant_and_polynomial():
    _, caputo, _ = audit.reference_case(129, "float64", "caputo", .5)
    np.testing.assert_array_equal(caputo[:, 0], np.zeros(129))
    _, rl, _ = audit.reference_case(129, "float64", "grunwald_letnikov", .5)
    assert rl[-1, 0] > 0
    np.testing.assert_allclose(rl[-1, 0], 1 / gamma(.5), rtol=.002)
    np.testing.assert_allclose(caputo[-1, 1], 2 / gamma(2.5), rtol=.001)


def test_accuracy_measurements_and_failure_suppression(monkeypatch):
    row = audit.accuracy_case(64, "caputo", "scaled_tail", 8, .5, "float32")
    assert row.measurement_ok and row.signal_relative_l2_errors[0] is None
    assert row.signal_max_abs_errors[0] == 0

    def bad_runner(*args):
        return lambda x, a: (jnp.full_like(x, jnp.nan), jnp.nan, jnp.nan, False)

    monkeypatch.setattr(audit, "accuracy_runner", bad_runner)
    failed = audit.accuracy_case(64, "caputo", "legacy", 8, .5, "float32")
    assert not failed.measurement_ok and not failed.fit_finite
    assert failed.signal_max_abs_errors is None and failed.kernel_max_abs_error is None


def test_exact_accuracy_matrix_gate():
    row = audit.accuracy_case(64, "caputo", "scaled_tail", 8, .5, "float32")
    rows = [replace(row, operator=op, n_samples=n, poles=p, fit_method=m,
                    order=float(np.float32(a)))
            for op in audit.OPERATORS for n in audit.LENGTHS for p in audit.POLES
            for m in audit.METHODS for a in audit.ORDERS]
    assert audit.accuracy_passed(rows, "float32")
    assert not audit.accuracy_passed(rows[:-1], "float32")
    assert not audit.accuracy_passed(rows[:-1] + [rows[0]], "float32")
    assert not audit.accuracy_passed(
        rows[:-1] + [replace(rows[-1], measurement_ok=False)], "float32",
    )


def test_small_real_scaling_measurement_and_gate():
    config = dict(lengths=(32,), dimensions=(2,), poles_values=(4,), repeats=2)
    rows = audit.collect_scaling("float32", **config)
    assert audit.scaling_passed(rows, dtype="float32", **config)
    assert not audit.scaling_passed(rows[:-1] + [rows[0]], dtype="float32", **config)
    corrupt = [replace(rows[0], agrees_with_dense=False), rows[1]]
    assert not audit.scaling_passed(corrupt, dtype="float32", **config)
