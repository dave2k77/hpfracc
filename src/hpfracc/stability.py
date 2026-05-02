"""Public API stability tier labels."""

from __future__ import annotations

from enum import StrEnum


class StabilityTier(StrEnum):
    """Compatibility status for user-facing APIs."""

    STABLE = "stable"
    PROVISIONAL = "provisional"
    EXPERIMENTAL = "experimental"
    PRIVATE = "private"

