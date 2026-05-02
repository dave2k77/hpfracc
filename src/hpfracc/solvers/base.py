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
        """Return a JSON-compatible representation where possible."""

        payload = asdict(self)
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

