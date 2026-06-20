"""Shared contracts for fractional operators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from hpfracc.config import Provenance
from hpfracc.typing import PyTree


class OperatorFamily(StrEnum):
    """Fractional operator families exposed in v0.1."""

    RIEMANN_LIOUVILLE = "riemann_liouville"
    CAPUTO = "caputo"
    GRUNWALD_LETNIKOV = "grunwald_letnikov"


class HistoryMethod(StrEnum):
    """Named history strategies for fractional operators.

    New compression strategies must be added here and remain opt-in; the default
    stays ``FULL`` so that exact full-history behavior is never silently changed.
    """

    FULL = "full"
    FFT = "fft"
    SHORT_MEMORY = "short_memory"
    SOE = "soe"


@dataclass(frozen=True, slots=True)
class OperatorInfo:
    """Metadata describing a fractional operator evaluation."""

    family: OperatorFamily
    method: str
    fractional_order: float | tuple[float, ...]
    dt: float
    n_steps: int
    history: HistoryMethod = field(default=HistoryMethod.FULL)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation where possible."""

        payload = asdict(self)
        payload["family"] = self.family.value
        payload["history"] = self.history.value
        payload["warnings"] = list(self.warnings)
        if isinstance(self.fractional_order, tuple):
            payload["fractional_order"] = list(self.fractional_order)
        return payload


@dataclass(frozen=True, slots=True)
class OperatorResult:
    """Structured result for fractional operator evaluations."""

    values: PyTree
    operator_info: OperatorInfo
    metadata: Provenance = field(default_factory=Provenance)

