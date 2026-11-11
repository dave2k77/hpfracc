"""Independent exact-input regressions for private host arithmetic.

Test/reference logic is preserved from the archived independent test design.
The subject loader, local names and dispatch typing are adapted to this repo.
High-precision references are validation-only; sampled cell checks are not proofs.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
import pytest

from benchmarks.numerical import _volterra_defect as subject

SUBJECT = Path(subject.__file__)
EXPECTED_SHA = hashlib.sha256(SUBJECT.read_bytes()).hexdigest()


@dataclass
class Case:
    name: str
    ts: np.ndarray
    ys: np.ndarray
    order: object
    matrix: np.ndarray
    powers: np.ndarray
    coefficients: np.ndarray

    def kwargs(self):
        out = dict(order=self.order, matrix=self.matrix, initial_state=self.ys[0])
        if self.powers.size:
            out.update(
                power_exponents=self.powers, power_coefficients=self.coefficients
            )
        return out


def case(
    name, ts, ys, order=0.5, matrix=None, powers=(), coefficients=None, dtype=np.float64
):
    ys = np.asarray(ys, dtype=dtype)
    d = ys.shape[1]
    return Case(
        name,
        np.asarray(ts, dtype=dtype),
        ys,
        dtype(order),
        np.zeros((d, d), dtype=dtype)
        if matrix is None
        else np.asarray(matrix, dtype=dtype),
        np.asarray(powers, dtype=dtype),
        np.empty((0, d), dtype=dtype)
        if coefficients is None
        else np.asarray(coefficients, dtype=dtype),
    )


def exact(value):
    n, d = float(value).as_integer_ratio()
    return mp.mpf(n) / mp.mpf(d)


def binary(value):
    arr = np.asarray(value)
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "hex": [float(x).hex() for x in arr.ravel()],
    }


def inputs(c, targets):
    return {
        "case": c.name,
        "ts": binary(c.ts),
        "states": binary(c.ys),
        "order": binary(c.order),
        "matrix": binary(c.matrix),
        "powers": binary(c.powers),
        "coefficients": binary(c.coefficients),
        "targets": binary(targets),
    }


@lru_cache(maxsize=100)
def _reference(encoded):
    raw = json.loads(encoded)

    def unpack(key):
        a = raw[key]
        return np.asarray([float.fromhex(x) for x in a["hex"]]).reshape(a["shape"])

    ts, ys, targets = unpack("ts"), unpack("states"), unpack("targets")
    with mp.workdps(420):
        nodes = [exact(x) for x in ts]
        values = [[exact(x) for x in row] for row in ys]
        a = exact(unpack("order"))
        mat = [[exact(x) for x in row] for row in unpack("matrix")]
        powers = [exact(x) for x in unpack("powers")]
        coeff = [[exact(x) for x in row] for row in unpack("coefficients")]
        rows = {"reconstruction": [], "fractional_rhs": [], "defect": []}
        for target in targets:
            t = exact(target)
            knot = next((i for i, x in enumerate(nodes) if t == x), None)
            if knot is not None:
                y = values[knot]
            else:
                j = next(
                    i for i in range(len(nodes) - 1) if nodes[i] < t < nodes[i + 1]
                )
                left, r = nodes[j : j + 2]
                y = [
                    (r - t) / (r - left) * v + (t - left) / (r - left) * w
                    for v, w in zip(values[j], values[j + 1], strict=True)
                ]
            integrals = [mp.mpf(0) for _ in y]
            for j, left in enumerate(nodes[:-1]):
                r = nodes[j + 1]
                if t <= left:
                    continue
                age, h = t - left, r - left
                q = min(h, age) / age
                # Integrate endpoint basis in x=(s-l)/(t-l).
                # This is independent of production's hypergeometric/expm1 branches.
                total = age**a / mp.gamma(a) * mp.betainc(1, a, 0, q)
                wr = age ** (a + 1) / (h * mp.gamma(a)) * mp.betainc(2, a, 0, q)
                wl = total - wr
                for k in range(len(y)):
                    integrals[k] += wl * values[j][k] + wr * values[j + 1][k]
            rhs = [mp.fsum(row[k] * integrals[k] for k in range(len(y))) for row in mat]
            tau = t - nodes[0]
            for b, co in zip(powers, coeff, strict=True):
                f = mp.gamma(b + 1) / mp.gamma(b + a + 1) * tau ** (b + a)
                rhs = [v + f * c for v, c in zip(rhs, co, strict=True)]
            defect = [v - z - f for v, z, f in zip(y, values[0], rhs, strict=True)]
            for key, vals in zip(rows, [y, rhs, defect], strict=True):
                rows[key].append([mp.nstr(v, 110) for v in vals])
    return rows


def reference(c, targets):
    return _reference(json.dumps(inputs(c, targets), sort_keys=True))


def as_floats(ref, key):
    return np.asarray([[float(v) for v in row] for row in ref[key]])


def record(event):
    path = os.environ.get("A1_OBSERVATIONS")
    if path:
        with Path(path).open("a") as f:
            f.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")


def run(api, c, targets, *, policy="must_recover") -> Any:
    targets = np.asarray(targets, dtype=c.ts.dtype)
    event = {
        "api": api,
        "policy": policy,
        "inputs": inputs(c, targets),
        "reference_420dps": reference(c, targets),
        "subject_sha256": EXPECTED_SHA,
    }
    try:
        if api == "point":
            result = subject.affine_power_defect(c.ts, c.ys, targets, **c.kwargs())
            arrays = {
                key: binary(getattr(result, key))
                for key in ("reconstruction", "fractional_rhs", "defect")
            }
        else:
            result = subject.affine_power_cells(c.ts, c.ys, targets, **c.kwargs())
            arrays = {
                key: binary(getattr(result, key))
                for key in ("verification_times", "model_lower", "model_upper")
            }
    except FloatingPointError as exc:
        event.update(outcome="explicit_FloatingPointError", message=str(exc))
        record(event)
        if policy == "must_recover":
            pytest.fail(
                f"{c.name}/{api}: successful recovery required; rejected: {exc}"
            )
        assert str(exc), "Arithmetic failure needs operation context"
        return None
    event.update(outcome="returned", arrays=arrays)
    record(event)
    assert result.arithmetic_verified is False
    assert result.total_numerical_certified is False
    json.dumps(result.to_dict(), allow_nan=False)
    for key in arrays:
        arr = getattr(result, key)
        assert arr.dtype == np.dtype("float64")
        assert np.isfinite(arr).all()
    if policy == "must_raise":
        pytest.fail(f"{c.name}/{api}: unsafe nonzero intermediate silently returned")
    return result


def close(actual, expected, *, scales=None, rtol=2e-12, atol=0):
    if scales is not None:
        actual, expected = actual / scales, expected / scales
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def covers(result, values, targets, scales):
    # Sampled corroboration at independent values, not a cell-enclosure proof.
    edges = result.verification_times
    for i, (left, r) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = (targets >= left) & (targets <= r)
        slack = 2e-12 * np.asarray(scales)
        assert np.all(values[selected] >= result.model_lower[i] - slack)
        assert np.all(values[selected] <= result.model_upper[i] + slack)


def canonical(which, sign=1.0):
    if which == "B1":
        return case(
            "B1" if sign == 1 else "B1_signed",
            [-2.0, -1.0, 1e-17],
            [[0.0], [sign * 1e20], [0.0]],
        )
    if which == "B2":
        return case(
            "B2" if sign == 1 else "B2_signed", [0.0, 1e300], [[0.0], [sign * 1e-100]]
        )
    return case(
        "B3",
        [-2.0, -1.0, 1e-17],
        [[0.0, 0.0], [0.0, 1e20], [0.0, 0.0]],
        order=1e-20,
        matrix=[[0.0, 1.0], [0.0, 0.0]],
    )


@pytest.mark.parametrize("which", ["B1", "B2", "B3"])
@pytest.mark.parametrize("api", ["point", "cells"])
def test_six_canonical_cases_require_successful_recovery(which, api):
    c = canonical(which)
    target = 5e299 if which == "B2" else 0.0
    ref = reference(c, np.asarray([target]))
    if api == "point":
        result = run(api, c, [target])
        # B3's coupled output must agree independently, not only its source state.
        if which == "B3":
            close(result.defect[:, :1], as_floats(ref, "defect")[:, :1])
        close(result.reconstruction, as_floats(ref, "reconstruction"))
        close(result.fractional_rhs, as_floats(ref, "fractional_rhs"))
        close(result.defect, as_floats(ref, "defect"))
    else:
        mesh = np.union1d(c.ts, [target])
        result = run(api, c, mesh)
        if which == "B1":
            close(result.model_upper[-1, :1], as_floats(ref, "defect")[0, :1])
        elif which == "B2":
            close(result.model_upper[0, :1], as_floats(ref, "defect")[0, :1])
        else:
            expected = as_floats(ref, "defect")[0, 0]
            assert result.model_lower[-1, 0] <= expected + abs(expected) * 2e-12


@pytest.mark.parametrize("which", ["B1", "B2"])
@pytest.mark.parametrize("api", ["point", "cells"])
def test_negative_state_siblings_reverse_cell_extrema(which, api):
    c = canonical(which, -1.0)
    target = 5e299 if which == "B2" else 0.0
    expected = as_floats(reference(c, np.asarray([target])), "defect")
    if api == "point":
        point = run("point", c, [target])
        close(point.defect, expected)
    else:
        cells = run("cells", c, np.union1d(c.ts, [target]))
        index = -1 if which == "B1" else 0
        close(cells.model_lower[index], expected[0])


@pytest.mark.parametrize("api", ["point", "cells"])
def test_opposite_huge_states_are_range_safe_or_explicitly_rejected(api):
    c = case(
        "opposite_huge_states", [0.0, 1.0, 2.0, 3.0], [[0.0], [1e308], [-1e308], [0.0]]
    )
    targets = np.asarray([1.25, 1.5, 1.75])
    ref = reference(c, targets)
    mesh = np.union1d(c.ts, targets) if api == "cells" else targets
    result = run(api, c, mesh, policy="recover_or_raise")
    if result is not None:
        if api == "point":
            close(result.reconstruction, as_floats(ref, "reconstruction"))
            assert result.reconstruction[1, 0] == 0.0
        else:
            covers(result, as_floats(ref, "defect"), targets, [1e308])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_endpoint_and_every_stored_knot_are_exact_identities(dtype):
    # No slope, width-ratio or state difference is needed at stored knots.
    s = 1e30 if dtype == np.float32 else 1e300
    c = case(
        "knot_identity_" + np.dtype(dtype).name,
        [-2.0, -1.0, 1e-17],
        [[0.0, 0.0], [s, -s], [-s, s]],
        dtype=dtype,
    )
    point = run("point", c, c.ts)
    np.testing.assert_array_equal(point.reconstruction, c.ys.astype(np.float64))
    cells = run("cells", c, c.ts)
    np.testing.assert_array_equal(cells.verification_times, c.ts.astype(np.float64))
    np.testing.assert_array_equal(
        cells.model_lower, np.minimum(c.ys[:-1], c.ys[1:]).astype(np.float64)
    )
    np.testing.assert_array_equal(
        cells.model_upper, np.maximum(c.ys[:-1], c.ys[1:]).astype(np.float64)
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("regime", ["small", "near_one"])
def test_signed_coupled_shifted_power_controls_at_represented_dtypes(dtype, regime):
    order = dtype(1e-20) if regime == "small" else np.nextafter(dtype(1), dtype(0))
    scales = np.asarray([2.0**-20, 2.0**20])
    ys = np.asarray([[1.0, -1.0], [-0.5, 0.25], [0.75, 1.0], [-1.0, -0.5]]) * scales
    matrix = [[-0.25, 0.125 / 2.0**40], [-0.5 * 2.0**40, 0.125]]
    coeff = np.asarray([[0.25, -0.125], [-0.5, 0.5], [1 / 3, -2 / 3]]) * scales
    c = case(
        "coupled_" + np.dtype(dtype).name + "_" + regime,
        [1.25, 1.375, 1.75, 2.0],
        ys,
        order=order,
        matrix=matrix,
        powers=[0.0, 0.25, 3.0],
        coefficients=coeff,
        dtype=dtype,
    )
    targets = np.asarray([1.25, 1.3125, 1.5, 1.875, 2.0], dtype=dtype)
    ref = reference(c, targets)
    point = run("point", c, targets)
    close(
        point.reconstruction,
        as_floats(ref, "reconstruction"),
        scales=scales,
        rtol=2e-13,
        atol=1e-14,
    )
    close(
        point.fractional_rhs,
        as_floats(ref, "fractional_rhs"),
        scales=scales,
        atol=2e-13,
    )
    close(point.defect, as_floats(ref, "defect"), scales=scales, atol=2e-13)
    assert point.provenance.config["input_state_dtype"] == np.dtype(dtype).name
    cells = run("cells", c, np.union1d(c.ts, targets))
    covers(cells, as_floats(ref, "defect"), targets, scales)
    covers(cells, point.defect, targets, scales)


@pytest.mark.parametrize(
    "order", [1e-20, 0.5, np.nextafter(1.0, 0.0)], ids=["small", "half", "near_one"]
)
def test_active_old_and_small_near_branch_boundaries_keep_positive_history(order):
    # Second zero panel makes the first panel old; targets straddle q=1/2.
    c = case(
        "branch_limits_" + float(order).hex(),
        [0.0, 1.0, 3.0],
        [[1.0], [0.0], [0.0]],
        order=order,
        matrix=[[100.0]],
    )
    targets = np.asarray(
        [
            order,
            np.nextafter(1.0, 0.0),
            1.0,
            np.nextafter(1.0, 2.0),
            np.nextafter(2.0, 1.0),
            2.0,
            np.nextafter(2.0, 3.0),
            3.0,
        ]
    )
    ref = reference(c, targets)
    point = run("point", c, targets)
    close(point.fractional_rhs, as_floats(ref, "fractional_rhs"))
    cells = run("cells", c, np.union1d(c.ts, targets))
    covers(cells, as_floats(ref, "defect"), targets, [100.0])
    # A coarse cell must include the independently located active-left peak.
    coarse = run("cells", c, c.ts)
    peak = as_floats(reference(c, np.asarray([order])), "defect")[0, 0]
    assert coarse.model_lower[0, 0] <= peak + 2e-12 * 100


def test_extreme_shifted_origin_and_amplitudes_do_not_reset_forcing_clock():
    c = case(
        "huge_shift",
        [-1e200, 0.0, 1e200],
        [[1e100, -1e100], [-0.5e100, 0.25e100], [0.75e100, -0.75e100]],
        matrix=[[1e-100, -2e-100], [0.5e-100, 1e-100]],
        powers=[0.0],
        coefficients=[[0.25, -0.5]],
    )
    targets = np.asarray([-1e200, -0.5e200, 0.0, 0.5e200, 1e200])
    ref = reference(c, targets)
    point = run("point", c, targets)
    close(
        point.reconstruction, as_floats(ref, "reconstruction"), scales=1e100, atol=1e-14
    )
    close(
        point.fractional_rhs, as_floats(ref, "fractional_rhs"), scales=1e100, atol=2e-13
    )
    cells = run("cells", c, np.union1d(c.ts, targets))
    covers(cells, as_floats(ref, "defect"), targets, [1e100, 1e100])


def guard(which):
    if which == "matrix_rescuable":
        return case(which, [0.0, 1e300], [[1e-200], [1e-200]], matrix=[[1e-200]]), [
            1e300
        ]
    if which == "matrix_reduction_masked":
        return case(
            which,
            [0.0, 1.0],
            [[1.0, 1e-200], [1.0, 1e-200]],
            matrix=[[1.0, 1e-200], [0.0, 0.0]],
        ), [1.0]
    if which == "power_contraction_zero":
        return case(
            which, [0.0, 1e-200], [[0.0], [0.0]], powers=[1.0], coefficients=[[1e-100]]
        ), [1e-200]
    if which == "power_reduction_masked":
        return case(
            which,
            [0.0, 1e-200],
            [[0.0], [0.0]],
            powers=[0.0, 1.0],
            coefficients=[[1.0], [1e-100]],
        ), [1e-200]
    return case(
        "prior_timepower",
        [0.0, 1.0],
        [[0.0], [0.0]],
        powers=[1100.0],
        coefficients=[[1e300]],
    ), [0.5]


@pytest.mark.parametrize(
    "which",
    [
        "matrix_rescuable",
        "matrix_reduction_masked",
        "power_contraction_zero",
        "power_reduction_masked",
        "prior_timepower",
    ],
)
@pytest.mark.parametrize("api", ["point", "cells"])
def test_unsafe_nonzero_products_fail_explicitly_instead_of_disappearing(which, api):
    c, targets = guard(which)
    if which in ("matrix_rescuable", "prior_timepower"):
        normal = float(reference(c, np.asarray(targets))["fractional_rhs"][0][0])
        assert normal > np.finfo(float).tiny and np.isfinite(normal)
    # These cases intentionally exercise unsafe primitive products. This cycle
    # promises conservative explicit rejection, not arbitrary scaled rescue.
    mesh = np.union1d(c.ts, targets) if api == "cells" else targets
    run(api, c, mesh, policy="must_raise")


@pytest.mark.parametrize("api", ["point", "cells"])
def test_range_safe_exact_zero_feedback_and_signed_forcing_cancellation(api):
    c = case(
        "legitimate_zeros",
        [0.0, 1.0],
        [[1.0, -1.0], [1.0, -1.0]],
        matrix=[[1.0, 1.0], [-1.0, -1.0]],
        powers=[0.0, 0.0],
        coefficients=[[0.25, -0.5], [-0.25, 0.5]],
    )
    targets = np.asarray([0.0, 0.5, 1.0])
    result = run(api, c, targets)
    if api == "point":
        np.testing.assert_array_equal(result.fractional_rhs, np.zeros((3, 2)))
        np.testing.assert_array_equal(result.defect, np.zeros((3, 2)))
    else:
        covers(result, np.zeros((3, 2)), targets, [1.0, 1.0])


@pytest.mark.parametrize("api", ["point", "cells"])
def test_ratio_underflow_is_explicit_or_scaled_value_is_recovered(api):
    c = case("rescuable_ratio", [0.0, 1e300], [[0.0], [1e300]])
    targets = np.asarray([1e-100])
    ref = as_floats(reference(c, targets), "defect")
    mesh = np.union1d(c.ts, targets) if api == "cells" else targets
    result = run(api, c, mesh, policy="recover_or_raise")
    if result is not None:
        got = result.defect[0] if api == "point" else result.model_upper[0]
        close(got, ref[0])
