"""Validation for gradients with respect to the fractional order ``alpha``.

These promote alpha-gradients from provisional to validated: finite-difference
checks across a range of alpha for every operator and the solver endpoint, plus
a finiteness/JIT smoke test. The finiteness check is the load-bearing one --
the L1 and PECE weights take powers of a zero base at the most recent lag, whose
naive gradient is NaN; the operators use a NaN-safe power to keep it finite.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from benchmarks.numerical.gradient_checks import (
    operator_alpha_row,
    solver_alpha_row,
)
from hpfracc.ops import caputo, grunwald_letnikov, riemann_liouville

_OPERATORS = {
    "caputo": caputo,
    "grunwald_letnikov": grunwald_letnikov,
    "riemann_liouville": riemann_liouville,
}


@pytest.mark.parametrize("target", list(_OPERATORS))
@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.7, 0.9])
def test_operator_alpha_gradient_matches_finite_difference(target, alpha) -> None:
    row = operator_alpha_row(target, alpha=alpha)
    assert jnp.isfinite(row.autodiff)
    assert row.abs_error < 1e-5


@pytest.mark.parametrize("alpha", [0.5, 0.7, 0.9])
def test_solver_alpha_gradient_matches_finite_difference(alpha) -> None:
    row = solver_alpha_row(alpha=alpha)
    assert jnp.isfinite(row.autodiff)
    assert row.abs_error < 1e-4


@pytest.mark.parametrize("target", list(_OPERATORS))
def test_operator_alpha_gradient_is_finite_under_jit(target) -> None:
    operator = _OPERATORS[target]
    x = (jnp.arange(16, dtype=jnp.float32) * 0.05) ** 2

    def objective(order: object) -> object:
        return operator(x, dt=0.05, order=order).sum()

    # Small alpha stresses the zero-base power in the most recent weight, where a
    # naive gradient would be NaN.
    grad = jax.jit(jax.grad(objective))(0.15)
    assert jnp.isfinite(grad)
