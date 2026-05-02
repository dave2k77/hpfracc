"""Analytic reference formulas for fractional-operator validation."""

from __future__ import annotations

from math import gamma
from typing import Any

from hpfracc.ops.orders import validate_order


def caputo_power_law(t: Any, *, power: float, order: float) -> Any:
    """Analytic Caputo derivative of ``t**power`` for ``0 < order < 1``.

    For ``power = beta`` and ``beta > 0``, the Caputo derivative is

    ``Gamma(beta + 1) / Gamma(beta + 1 - alpha) * t**(beta - alpha)``.

    For a constant power ``beta = 0``, the Caputo derivative is zero.
    """

    alpha = validate_order(order)
    if float(power) < 0.0:
        msg = f"Expected nonnegative power for validation formula, got {power}."
        raise ValueError(msg)

    jnp = _jnp()
    values = jnp.asarray(t)
    if float(power) == 0.0:
        return jnp.zeros_like(values)

    coefficient = gamma(float(power) + 1.0) / gamma(float(power) + 1.0 - alpha)
    return coefficient * values ** (float(power) - alpha)


def _jnp() -> Any:
    try:
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        msg = (
            "JAX is required for hpfracc.ops analytic validation helpers. "
            "Install the package with its runtime dependencies before calling "
            "this function."
        )
        raise ModuleNotFoundError(msg) from exc
    return jnp

