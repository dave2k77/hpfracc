"""Shared solver result contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from hpfracc.config import Provenance
from hpfracc.typing import PyTree


@dataclass(frozen=True, slots=True)
class SolverInfo:
    """Metadata and diagnostics for a solver run."""

    name: str
    method: str
    fractional_order: float | None = None
    step_size: float | None = None
    n_steps: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Export concrete metadata on the host, outside JIT/gradient tracing.

        Supported scalar-array orders become Python floats in the exported
        payload only; the stored order and numerical execution are unchanged.
        """

        payload = asdict(self)
        if self.fractional_order is not None:
            payload["fractional_order"] = float(self.fractional_order)
        payload["warnings"] = list(self.warnings)
        return payload


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Structured result returned by public simulation routines."""

    ts: PyTree
    latent_state: PyTree
    observed: PyTree | None = None
    solver_info: SolverInfo | None = None
    metadata: Provenance = field(default_factory=Provenance)

