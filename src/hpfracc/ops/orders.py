"""Fractional-order validation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FractionalOrder:
    """Scalar fractional order for v0.1 operators.

    v0.1 supports scalar orders in the open interval ``0 < alpha < 1``.
    Vector, field-valued, or trainable order structures are future extensions.
    """

    alpha: float

    def __post_init__(self) -> None:
        validate_order(self.alpha)


def validate_order(alpha: float) -> float:
    """Validate and return a scalar fractional order."""

    if not 0.0 < float(alpha) < 1.0:
        msg = f"Expected fractional order in the open interval (0, 1), got {alpha}."
        raise ValueError(msg)
    return float(alpha)


def as_order(alpha: Any) -> Any:
    """Return ``alpha`` for use in a differentiable numerical path.

    Unlike :func:`validate_order`, this does **not** coerce ``alpha`` to a Python
    ``float``: doing so would break ``jax.grad`` with respect to the fractional
    order, since a traced value cannot be converted to a concrete ``float``. The
    open-interval constraint is still enforced eagerly whenever ``alpha`` is
    concrete; under tracing (e.g. inside ``jax.grad``/``jax.jit``) the check is
    skipped and the constraint is assumed to hold, as it is enforced at eager
    construction sites (``FractionalOrder``, the solver config, and concrete
    operator calls).

    ``alpha`` may be a scalar **or** an array of per-state fractional orders. For
    an array, *every* entry must lie in the open interval ``(0, 1)``; the same
    elementwise check is skipped under tracing.
    """

    import numpy as np

    try:
        # Coerce both scalars and arrays to a concrete numpy view. A JAX tracer
        # raises ``TracerArrayConversionError`` (a ``TypeError`` subclass), so it
        # falls through to the deferral branch below.
        concrete = np.asarray(alpha, dtype=float)
    except (TypeError, ValueError):
        # Traced value under a JAX transform: defer to eager validation sites.
        return alpha
    if not np.all((concrete > 0.0) & (concrete < 1.0)):
        msg = f"Expected fractional order in the open interval (0, 1), got {alpha}."
        raise ValueError(msg)
    return alpha

