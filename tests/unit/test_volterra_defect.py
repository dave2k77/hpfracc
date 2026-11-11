"""Independent checks for the private host-only A1 trajectory diagnostic."""

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest


def diagnostic_module():
    name = "benchmarks.numerical._volterra_defect"
    assert importlib.util.find_spec(name) is not None, "Missing PL defect diagnostic"
    return importlib.import_module(name)


def panel_reference(ts, states, targets, order, matrix, powers=(), coefficients=()):
    """80-digit panel quadrature, independent of closed-form basis weights."""
    with mp.workdps(80):
        nodes = [mp.mpf(float(t)) for t in ts]
        values = [[mp.mpf(float(x)) for x in row] for row in states]
        a = mp.mpf(float(order))
        mat = [[mp.mpf(float(x)) for x in row] for row in matrix]
        beta = [mp.mpf(float(x)) for x in powers]
        coeff = [[mp.mpf(float(x)) for x in row] for row in coefficients]
        result = []
        for target in targets:
            t = mp.mpf(float(target))
            totals = []
            for i in range(len(values[0])):
                panels = []
                for j, left in enumerate(nodes[:-1]):
                    right = min(nodes[j + 1], t)
                    if right <= left:
                        continue

                    def integrand(u, t=t, left=left, j=j, i=i):
                        s = t - u ** (1 / a)
                        fraction = (s - left) / (nodes[j + 1] - left)
                        y = [x + fraction * (z - x)
                             for x, z in zip(values[j], values[j + 1], strict=True)]
                        return (mp.fsum(mat[i][k] * y[k] for k in range(len(y)))
                                + mp.fsum(c[i] * max(s - nodes[0], 0) ** b
                                          for b, c in zip(beta, coeff, strict=True)))

                    # u=(t-s)**a removes the terminal kernel singularity.
                    panels.append(mp.quad(integrand,
                                          [(t - right) ** a, (t - left) ** a]))
                totals.append(float(mp.fsum(panels) / mp.gamma(a + 1)))
            result.append(totals)
        return np.asarray(result)


def test_reconstructs_stored_states_and_full_history_defect():
    ts = np.asarray([1., 1.25, 1.5, 2.])
    states = np.asarray([[1.], [1.5], [.75], [2.]])
    matrix = np.asarray([[.25]])
    result = diagnostic_module().affine_power_defect(
        ts, states, ts, order=.5, matrix=matrix, initial_state=states[0],
    )
    expected_rhs = panel_reference(ts, states, ts, .5, matrix)
    np.testing.assert_allclose(result.reconstruction, states, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.fractional_rhs, expected_rhs,
                               rtol=2e-13, atol=1e-14)
    np.testing.assert_allclose(result.defect, states - states[0] - expected_rhs,
                               rtol=2e-13, atol=1e-14)
    assert result.arithmetic_verified is False
    assert result.total_numerical_certified is False
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("alpha", [.3, .5, .75])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_coupled_weak_power_and_cubic_forcing_at_represented_inputs(alpha, dtype):
    ts = np.asarray([1.25, 1.375, 1.75, 2.], dtype=dtype)
    scales = np.asarray([2.**-10, 2.**10])
    states = np.asarray(np.asarray([[1., 1.], [1.125, .875], [.75, 1.25],
                                    [1.25, .5]]) * scales, dtype=dtype)
    matrix = np.asarray([[-.25, .125/2.**20], [.5*2.**20, -.125]], dtype=dtype)
    powers = np.asarray([0., .25, 2., 3.], dtype=dtype)
    coefficients = np.asarray(np.asarray([[.25, .125], [.5, -.5], [1., 1.],
                                          [-2/3, 1/3]]) * scales, dtype=dtype)
    targets = np.asarray([1.25, 1.3125, 1.5, 1.875, 2.], dtype=dtype)
    order = dtype(alpha)
    result = diagnostic_module().affine_power_defect(
        ts, states, targets, order=order, matrix=matrix, initial_state=states[0],
        power_exponents=powers, power_coefficients=coefficients,
    )
    expected = panel_reference(ts, states, targets, order, matrix, powers,
                               coefficients)
    np.testing.assert_allclose(result.fractional_rhs/scales, expected/scales,
                               rtol=2e-12, atol=2e-13)
    independent_y = np.stack([np.interp(targets, ts, states[:, i])
                              for i in range(2)], axis=-1)
    np.testing.assert_allclose(result.reconstruction/scales, independent_y/scales,
                               rtol=2e-13, atol=1e-14)
    assert result.fractional_rhs.dtype == np.dtype("float64")
    assert result.provenance.config["input_state_dtype"] == np.dtype(dtype).name
    assert result.arithmetic_verified is result.total_numerical_certified is False


@pytest.mark.parametrize(("field", "bad_value"), [
    ("order", 0.), ("order", 1.), ("order", np.nan), ("order", True),
    ("order", np.asarray([.5, .5])), ("order", .5 + 0j),
    ("ts", np.asarray([0., 0.])), ("ts", np.asarray([1., 0.])),
    ("ts", np.asarray([0.])), ("ts", np.asarray([0., np.inf])),
    ("states", np.asarray([0., 1.])), ("states", np.zeros((2, 1, 1))),
    ("states", np.asarray([[0.], [np.nan]])),
    ("states", np.asarray([[0j], [1j]])),
    ("states", np.asarray([[0.], [1.]], dtype=np.float16)),
    ("matrix", np.asarray([1.])), ("matrix", np.eye(2)),
    ("matrix", np.asarray([[np.nan]])),
    ("initial_state", np.asarray([1.])),
    ("evaluation_times", np.asarray([])),
    ("evaluation_times", np.asarray([-.1, 1.])),
    ("evaluation_times", np.asarray([0., 1.1])),
    ("evaluation_times", np.asarray([np.nan])),
    ("power_exponents", np.asarray([-.25])),
    ("power_exponents", np.asarray([np.inf])),
    ("power_exponents", np.asarray([.25])),
    ("power_coefficients", np.asarray([[1.]])),
])
def test_invalid_inputs_fail_before_numerical_evaluation(field, bad_value):
    problem = dict(ts=np.asarray([0., 1.]), states=np.asarray([[0.], [1.]]),
                   evaluation_times=np.asarray([0., .5, 1.]), order=.5,
                   matrix=np.asarray([[-.25]]), initial_state=np.asarray([0.]))
    problem[field] = bad_value
    with pytest.raises(ValueError, match="Expected"):
        diagnostic_module().affine_power_defect(**problem)


def test_cell_model_ranges_detect_interior_defect_hidden_at_nodes():
    module = diagnostic_module()
    assert hasattr(module, "affine_power_cells"), "Missing continuum-cell diagnostic"
    ts = np.asarray([0., .75, 1.])
    # Frozen independently before implementation, at represented binary64 values.
    states = np.asarray([[0.], [float.fromhex("0x1.2c3287b67ac5cp-2")],
                         [float.fromhex("0x1.341f6bc02c7ecp-1")]])
    kwargs = dict(order=.5, matrix=np.asarray([[0.]]), initial_state=states[0],
                  power_exponents=np.asarray([2.]),
                  power_coefficients=np.asarray([[1.]]))
    nodal = module.affine_power_defect(ts, states, ts, **kwargs)
    assert np.max(np.abs(nodal.defect)) < 1e-15
    target = np.asarray([.375])
    interior = module.affine_power_defect(ts, states, target, **kwargs)
    assert interior.defect[0, 0] > .09
    cells = module.affine_power_cells(ts, states, np.asarray([0., 1.]), **kwargs)
    np.testing.assert_array_equal(cells.verification_times, ts)
    assert cells.model_upper[0, 0] >= interior.defect[0, 0]
    fine = module.affine_power_cells(ts, states, np.linspace(0., 1., 65), **kwargs)
    coarse_max = np.max(np.maximum(np.abs(cells.model_lower),
                                   np.abs(cells.model_upper)))
    fine_max = np.max(np.maximum(np.abs(fine.model_lower), np.abs(fine.model_upper)))
    assert .09 < fine_max < coarse_max
    assert fine.arithmetic_verified is fine.total_numerical_certified is False
    json.dumps(fine.to_dict(), allow_nan=False)


@pytest.mark.parametrize("max_work", [0, True, 1.5])
def test_cell_work_budget_requires_a_positive_canonical_integer(max_work):
    with pytest.raises(ValueError, match="Expected positive integer max_work"):
        diagnostic_module().affine_power_cells(
            np.asarray([0., 1.]), np.asarray([[0.], [1.]]), np.asarray([0., 1.]),
            order=.5, matrix=np.asarray([[0.]]), initial_state=np.asarray([0.]),
            max_work=max_work,
        )


def test_cell_work_budget_exhaustion_is_explicit_not_history_truncation():
    kwargs = dict(ts=np.asarray([0., 1.]), states=np.asarray([[0.], [1.]]),
                  verification_mesh=np.asarray([0., 1.]), order=.5,
                  matrix=np.asarray([[0.]]), initial_state=np.asarray([0.]))
    with pytest.raises(ValueError, match="Cell work budget exhausted"):
        diagnostic_module().affine_power_cells(**kwargs, max_work=1)
    result = diagnostic_module().affine_power_cells(**kwargs, max_work=3)
    assert result.provenance.config["cell_term_components"] == 3


FIXTURE = (Path(__file__).resolve().parents[2] / "docs/validation/audits/"
           "a1-trajectory-defect-2026-11-11/frozen-fixtures.json")
assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == (
    "63311bcecc3c31458df93b0c1079d0cec1d8cbc64ee1155abc07b028e9f9aaef"
)
FROZEN_CASES = json.loads(FIXTURE.read_text())["cases"]


def decode(value) -> np.ndarray:
    if isinstance(value, dict):
        return np.asarray(float.fromhex(value["hex"]), dtype=np.float64)
    return np.asarray([decode(item) for item in value], dtype=np.float64)


@pytest.mark.parametrize("case", FROZEN_CASES, ids=lambda case: case["id"])
def test_frozen_math_audit_cases_against_independent_panels_and_cell_corroboration(
    case,
):
    ts, states = decode(case["ts"]), decode(case["states"])
    kwargs = dict(order=decode(case["alpha"]), matrix=decode(case["A"]),
                  initial_state=decode(case["initial_state"]),
                  power_exponents=decode([q["beta"] for q in case["forcing"]]),
                  power_coefficients=decode([q["coefficient"]
                                             for q in case["forcing"]]))
    scales = decode(case["scales"]) if "scales" in case else np.ones(states.shape[1])
    targets = (ts[:-1] + ts[1:]) / 2
    result = diagnostic_module().affine_power_defect(ts, states, targets, **kwargs)
    expected = panel_reference(ts, states, targets, kwargs["order"], kwargs["matrix"],
                               kwargs["power_exponents"], kwargs["power_coefficients"])
    np.testing.assert_allclose(result.fractional_rhs/scales, expected/scales,
                               rtol=2e-12, atol=2e-13)
    mesh = np.unique(np.concatenate((ts, np.linspace(ts[0], ts[-1], 17))))
    cells = diagnostic_module().affine_power_cells(ts, states, mesh, **kwargs)
    for i, (left, right) in enumerate(zip(mesh[:-1], mesh[1:], strict=True)):
        sampled = diagnostic_module().affine_power_defect(
            ts, states, np.linspace(left, right, 5), **kwargs,
        ).defect
        # Corroboration with explicit f64 slack, NOT an enclosure/certificate test.
        assert np.all(sampled >= cells.model_lower[i] - 2e-12*scales)
        assert np.all(sampled <= cells.model_upper[i] + 2e-12*scales)


@pytest.mark.parametrize("mesh", [np.asarray([0., .5]), np.asarray([.5, 1.]),
                                  np.asarray([0., 1., .5, 1.]),
                                  np.asarray([0., 0., 1.]), np.asarray([[0., 1.]]),
                                  np.asarray([0., np.nan, 1.])])
def test_cell_mesh_cannot_leave_gaps_or_collapse(mesh):
    with pytest.raises(ValueError, match="Expected"):
        diagnostic_module().affine_power_cells(
            np.asarray([0., 1.]), np.asarray([[0.], [1.]]), mesh,
            order=.5, matrix=np.asarray([[0.]]), initial_state=np.asarray([0.]),
        )


def test_float64_cell_under_enclosure_is_preserved_not_relabelled_as_certification():
    module = diagnostic_module()
    left = np.nextafter(1., 0.)
    cells = module.affine_power_cells(
        np.asarray([0., 1.]), np.asarray([[0.], [0.]]), np.asarray([0., left, 1.]),
        order=.5, matrix=np.asarray([[0.]]), initial_state=np.asarray([0.]),
        power_exponents=np.asarray([0.]), power_coefficients=np.asarray([[1.]]),
    )
    with mp.workdps(80):
        true_right_defect = -2/mp.sqrt(mp.pi)
        assert mp.mpf(float(cells.model_lower[-1, 0])) > true_right_defect
    assert cells.arithmetic_verified is cells.total_numerical_certified is False
    assert "no outward rounding" in cells.provenance.config["arithmetic"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pece_stored_state_reconstruction_keeps_nodal_volterra_mismatch(dtype):
    import jax
    import jax.numpy as jnp

    import hpfracc as hp

    def model(t, state, params, *, rng_key=None, inputs=None):
        del t, params, rng_key, inputs
        return .25*state

    with jax.enable_x64(dtype == np.float64):
        result = hp.solvers.simulate(
            model=model, ts=jnp.asarray([0., 1.], dtype=dtype), params=None,
            initial_state=jnp.asarray([1.], dtype=dtype),
            solver=hp.solvers.NonUniformPredictorCorrector(
                order=jnp.asarray(.5, dtype=dtype),  # type: ignore[arg-type]
            ),
        )
        assert result.latent_state.dtype == np.dtype(dtype)
        ts = np.asarray(result.ts, dtype=np.float64)
        states = np.asarray(result.latent_state, dtype=np.float64)
        matrix = np.asarray([[.25]])
        diagnostic = diagnostic_module().affine_power_defect(
            ts, states, ts, order=.5, matrix=matrix, initial_state=states[0],
        )
        np.testing.assert_allclose(diagnostic.reconstruction, states, rtol=0, atol=0)
        rhs = panel_reference(ts, states, ts, .5, matrix)
        np.testing.assert_allclose(diagnostic.defect, states-states[0]-rhs,
                                   rtol=2e-12, atol=1e-14)
        assert abs(diagnostic.defect[-1, 0]) > 1e-4


@pytest.mark.parametrize("beta", [1e16, 1e100, 1e300])
@pytest.mark.parametrize("alpha", [.3, .5, .75])
def test_large_finite_power_gamma_ratio_does_not_cancel_to_one(alpha, beta):
    module = diagnostic_module()
    kwargs = dict(ts=np.asarray([0., 1.]), states=np.zeros((2, 1)), order=alpha,
                  matrix=np.zeros((1, 1)), initial_state=np.asarray([0.]),
                  power_exponents=np.asarray([beta]),
                  power_coefficients=np.asarray([[1.]]))
    with mp.workdps(int(np.log10(beta)) + 80):
        b, a = mp.mpf(beta), mp.mpf(alpha)
        expected = float(mp.exp(mp.loggamma(b+1) - mp.loggamma(b+a+1)))
    point = module.affine_power_defect(**kwargs, evaluation_times=np.asarray([1.]))
    cells = module.affine_power_cells(**kwargs, verification_mesh=np.asarray([0., 1.]))
    np.testing.assert_allclose(point.fractional_rhs[0, 0], expected, rtol=2e-12, atol=0)
    np.testing.assert_allclose(cells.model_lower[0, 0], -expected, rtol=2e-12, atol=0)
    assert cells.arithmetic_verified is cells.total_numerical_certified is False


@pytest.mark.parametrize("alpha", [.3, .5, .75])
def test_extreme_grading_retains_constant_tail_state_and_panel_history(alpha):
    module = diagnostic_module()
    ts = np.asarray([0., 1e-17, 1.])
    states = np.asarray([[0.], [1.], [1.]])
    matrix = np.asarray([[1.]])
    point = module.affine_power_defect(
        ts, states, np.asarray([1.]), order=alpha, matrix=matrix,
        initial_state=states[0],
    )
    cells = module.affine_power_cells(
        ts, states, np.asarray([0., 1e-17, .5, np.nextafter(1., 0.), 1.]),
        order=alpha, matrix=matrix, initial_state=states[0],
    )
    expected_rhs = panel_reference(ts, states, np.asarray([1.]), alpha, matrix)
    assert point.reconstruction[0, 0] == 1.
    np.testing.assert_allclose(point.fractional_rhs, expected_rhs,
                               rtol=2e-12, atol=0)
    expected_defect = 1-expected_rhs[0, 0]
    assert cells.model_upper[-1, 0] < expected_defect/2
    assert cells.arithmetic_verified is cells.total_numerical_certified is False


@pytest.mark.parametrize("method", ["point", "cells"])
def test_unsafe_intermediate_underflow_fails_instead_of_returning_finite_zero(method):
    module = diagnostic_module()
    with mp.workdps(180):
        reference = float(mp.mpf(1e300) * mp.gamma(1101)/mp.gamma(mp.mpf(1101.5))
                          * mp.mpf(.5)**mp.mpf(1100.5))
    assert np.isfinite(reference) and reference > np.finfo(float).tiny
    kwargs = dict(order=.5, matrix=np.zeros((1, 1)), initial_state=np.asarray([0.]),
                  power_exponents=np.asarray([1100.]),
                  power_coefficients=np.asarray([[1e300]]))
    with pytest.raises(FloatingPointError, match="underflow"):
        if method == "point":
            module.affine_power_defect(np.asarray([0., 1.]), np.zeros((2, 1)),
                                       np.asarray([.5]), **kwargs)
        else:
            module.affine_power_cells(np.asarray([0., 1.]), np.zeros((2, 1)),
                                      np.asarray([0., .5, 1.]), **kwargs)


@pytest.mark.parametrize("alpha", [.3, .5, .75])
@pytest.mark.parametrize("knot", [1e-17, .25, np.nextafter(1., 0.)])
def test_positive_panel_weights_match_reference_near_both_kernel_limits(alpha, knot):
    ts = np.asarray([0., knot, 1.])
    states = np.asarray([[0.], [1.], [1.]])
    matrix = np.asarray([[1.]])
    result = diagnostic_module().affine_power_defect(
        ts, states, np.asarray([1.]), order=alpha, matrix=matrix,
        initial_state=states[0],
    )
    expected = panel_reference(ts, states, np.asarray([1.]), alpha, matrix)
    np.testing.assert_allclose(result.fractional_rhs, expected, rtol=2e-12, atol=0)


@pytest.mark.parametrize("alpha", [.3, .5, .75])
def test_cell_range_keeps_interior_left_basis_weight_peak(alpha):
    module = diagnostic_module()
    ts = np.asarray([0., 1.])
    states = np.asarray([[1.], [0.]])
    kwargs = dict(order=alpha, matrix=np.asarray([[100.]]), initial_state=states[0])
    point = module.affine_power_defect(ts, states, np.asarray([alpha]), **kwargs)
    cells = module.affine_power_cells(ts, states, ts, **kwargs)
    # A large coefficient discriminates an endpoint-only weight range.
    assert cells.model_lower[0, 0] <= point.defect[0, 0] + 2e-12
    assert cells.arithmetic_verified is cells.total_numerical_certified is False


def test_nonfinite_arithmetic_is_not_returned_as_a_diagnostic():
    with pytest.raises(FloatingPointError, match="[Nn]onfinite|overflow"):
        diagnostic_module().affine_power_defect(
            np.asarray([0., 1.]), np.asarray([[1e308], [1e308]]),
            np.asarray([1.]), order=.5, matrix=np.asarray([[1e308]]),
            initial_state=np.asarray([1e308]),
        )
