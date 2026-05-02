"""Fractional-order validation primitives."""

from __future__ import annotations

from dataclasses import dataclass


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

