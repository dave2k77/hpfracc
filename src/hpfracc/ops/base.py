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


@dataclass(frozen=True, slots=True)
class OperatorInfo:
    """Metadata describing a fractional operator evaluation."""

    family: OperatorFamily
    method: str
    fractional_order: float
    dt: float
    n_steps: int
    history: str = "full"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation where possible."""

        payload = asdict(self)
        payload["family"] = self.family.value
        payload["warnings"] = list(self.warnings)
        return payload


@dataclass(frozen=True, slots=True)
class OperatorResult:
    """Structured result for fractional operator evaluations."""

    values: PyTree
    operator_info: OperatorInfo
    metadata: Provenance = field(default_factory=Provenance)

