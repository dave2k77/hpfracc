"""Roundoff-stable evaluation of the uniform full-history PECE weights."""

from __future__ import annotations

from typing import Any

from hpfracc.ops._quadrature import _power_interval_moments


def _validate_weight_evaluation(value: str) -> None:
    if value not in ("direct", "stable"):
        raise ValueError(
            f"Unknown weight_evaluation: {value!r}; use 'direct' or 'stable'."
        )


def _stable_pece_weights(
    step: Any, indices: Any, alpha: Any, dtype: Any,
) -> tuple[Any, Any]:
    """Return unscaled predictor/corrector weights without power cancellation.

    Integrate u**(alpha-1) on unit-width lag intervals. The predictor is alpha
    times the constant moment. A historical corrector node combines the left
    moment of its following interval and the right moment of its preceding
    interval, multiplied by alpha*(alpha+1). The current endpoint's coefficient
    is omitted here: callers retain their existing separate unit coefficient.

    The boundary, all interior nodes and future masks follow the same convention
    as the direct Diethelm formulas. This reuses the validated stable interval
    evaluation, not a different quadrature or a history approximation. The real
    floating weight dtype follows normal promotion of history and order inputs.
    """
    import jax.numpy as jnp

    weight_dtype = jnp.result_type(jnp.zeros((), dtype=dtype).real, alpha, float)
    a = jnp.asarray(alpha, dtype=weight_dtype)
    if a.ndim != 0:
        raise NotImplementedError("Uniform stable weights require a scalar order.")
    valid = indices < step
    lag = jnp.where(valid, (step - indices).astype(weight_dtype), 1.)
    integral, left, right = _power_interval_moments(
        lag, lag - 1, jnp.ones_like(lag), a, valid,
    )
    previous_right = jnp.concatenate((jnp.zeros_like(right[:1]), right[:-1]))
    predictor = a * integral
    corrector = jnp.where(valid, a * (a + 1) * (left + previous_right), 0.)
    return predictor, corrector
