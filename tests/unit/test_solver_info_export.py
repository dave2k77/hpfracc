"""Host metadata export must not alter scalar-array numerical execution."""

import json
import math
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import digamma

from hpfracc.solvers import (
    AdaptivePredictorCorrector,
    ImplicitPredictorCorrector,
    NonUniformPredictorCorrector,
    PredictorCorrector,
    SolverInfo,
    simulate,
)


def _constant(t, state, params, *, rng_key=None, inputs=None):
    return params * jnp.ones_like(state)


@pytest.mark.parametrize(
    "solver_type", [PredictorCorrector, ImplicitPredictorCorrector]
)
def test_strong_scalar_order_metadata_exports_plain_json(solver_type):
    with jax.enable_x64(False):
        order = jnp.asarray(0.6, dtype=jnp.float32)
        result = simulate(
            model=_constant,
            ts=jnp.linspace(0.0, 0.5, 3, dtype=jnp.float32),
            initial_state=jnp.ones(1, dtype=jnp.float32),
            params=jnp.asarray(0.25, dtype=jnp.float32),
            solver=solver_type(dt=0.25, order=order),
        )
        info = result.solver_info
        assert info is not None
        original_order = info.fractional_order
        assert isinstance(original_order, jax.Array)
        assert original_order.dtype == jnp.float32
        payload = info.to_dict()
        decoded = json.loads(json.dumps(payload, allow_nan=False))
        assert type(payload["fractional_order"]) is float
        assert decoded["fractional_order"] == float(order)
        assert info.fractional_order is original_order
        assert result.latent_state.dtype == jnp.float32


@pytest.mark.parametrize("dtype,wide", [(jnp.float32, False), (jnp.float64, True)])
@pytest.mark.parametrize(
    "kind,evaluation",
    [
        ("explicit", "direct"), ("explicit", "stable"),
        ("implicit", "direct"), ("implicit", "stable"),
        ("nonuniform", None), ("adaptive", None),
    ],
)
def test_shared_solver_paths_export_concrete_order_without_mutation(
    kind, evaluation, dtype, wide
):
    with jax.enable_x64(wide):
        order = jnp.asarray(0.6, dtype=dtype)
        if kind == "nonuniform":
            solver = NonUniformPredictorCorrector(
                order=order,  # type: ignore[arg-type]
            )
            times = jnp.asarray([0.0, 0.125, 0.5], dtype=dtype)
        elif kind == "adaptive":
            solver = AdaptivePredictorCorrector(
                order=order,  # type: ignore[arg-type]
                first_step=0.125, max_steps=4,
            )
            times = jnp.asarray([0.0, 0.5], dtype=dtype)
        else:
            solver_type = (
                PredictorCorrector if kind == "explicit" else ImplicitPredictorCorrector
            )
            solver = solver_type(
                dt=0.25, order=order,  # type: ignore[arg-type]
                weight_evaluation=evaluation,
            )
            times = jnp.linspace(0.0, 0.5, 3, dtype=dtype)
        result = simulate(
            model=_constant, ts=times, solver=solver,
            initial_state=jnp.ones(1, dtype=dtype),
            params=jnp.asarray(0.25, dtype=dtype),
        )
        info = result.solver_info
        assert info is not None
        original_order = info.fractional_order
        original_diagnostics = info.diagnostics
        assert isinstance(original_order, jax.Array)
        assert original_order.shape == () and original_order.dtype == dtype
        payload = info.to_dict()
        decoded = json.loads(json.dumps(payload, allow_nan=False))
        assert type(payload["fractional_order"]) is float
        assert decoded["fractional_order"] == float(order)
        assert decoded["warnings"] == list(info.warnings)
        assert decoded["diagnostics"] == original_diagnostics
        assert info.fractional_order is original_order
        assert info.diagnostics is original_diagnostics
        assert result.latent_state.dtype == dtype


@pytest.mark.parametrize("dtype,wide", [(jnp.float32, False), (jnp.float64, True)])
@pytest.mark.parametrize(
    "solver_type", [PredictorCorrector, ImplicitPredictorCorrector]
)
def test_numerical_jit_gradients_remain_array_based(solver_type, dtype, wide):
    with jax.enable_x64(wide):
        times = jnp.linspace(0.0, 0.5, 3, dtype=dtype)
        order = jnp.asarray(0.6, dtype=dtype)
        initial = jnp.ones(1, dtype=dtype)
        rate = jnp.asarray(0.25, dtype=dtype)

        def objective(alpha, y0, strength):
            return simulate(
                model=_constant, ts=times, initial_state=y0, params=strength,
                solver=solver_type(dt=0.25, order=alpha),
            ).latent_state[-1].sum()

        value, gradients = jax.jit(jax.value_and_grad(
            objective, argnums=(0, 1, 2)
        ))(order, initial, rate)
        eager_value, eager_gradients = jax.value_and_grad(
            objective, argnums=(0, 1, 2)
        )(order, initial, rate)
        # Constant forcing has y(T) = y0 + strength*T**alpha/Gamma(1+alpha).
        # The reference and its derivatives are evaluated independently on host.
        scale = 0.5 ** float(order) / math.gamma(1 + float(order))
        alpha_derivative = float(rate) * scale * (
            math.log(0.5) - digamma(1 + float(order))
        )
        tolerance = 2e-5 if dtype == jnp.float32 else 2e-12
        assert value.dtype == eager_value.dtype == dtype
        np.testing.assert_allclose(value, 1 + float(rate) * scale,
                                   rtol=tolerance, atol=tolerance)
        for actual, eager, expected in zip(
            gradients, eager_gradients, (alpha_derivative, np.ones(1), scale),
            strict=True,
        ):
            assert actual.dtype == eager.dtype == dtype
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            np.testing.assert_allclose(actual, eager, rtol=tolerance, atol=tolerance)
        assert abs(float(gradients[0])) > 0


@pytest.mark.parametrize("order", [None, 0.6])
def test_python_order_and_missing_order_exports_remain_compatible(order):
    info = SolverInfo(name="metadata", method="contract", fractional_order=order,
                      warnings=("example",))
    payload = json.loads(json.dumps(info.to_dict(), allow_nan=False))
    assert payload["fractional_order"] == order
    assert payload["warnings"] == ["example"]


def test_plain_solver_metadata_export_does_not_import_jax():
    script = """
import importlib.abc
import json
import sys

class NoJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'jax' or fullname.startswith('jax.'):
            raise ModuleNotFoundError('JAX deliberately unavailable', name=fullname)

sys.meta_path.insert(0, NoJax())
from hpfracc.solvers import SolverInfo
for order in (None, 0.6):
    info = SolverInfo(name='plain', method='metadata', fractional_order=order)
    json.dumps(info.to_dict(), allow_nan=False)
assert 'jax' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True,
                   text=True, timeout=30)
