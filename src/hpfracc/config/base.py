"""Base configuration and provenance objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from hpfracc._version import __version__


class RuntimeTarget(StrEnum):
    """Execution targets supported by the v0.1 contract."""

    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Minimal reproducibility metadata attached to public results."""

    package_version: str = __version__
    commit_hash: str | None = None
    backend: str = "jax"
    runtime_target: RuntimeTarget | None = None
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation where possible."""

        payload = asdict(self)
        if self.runtime_target is not None:
            payload["runtime_target"] = self.runtime_target.value
        return payload


@dataclass(frozen=True, slots=True)
class ExperimentMetadata:
    """User-facing experiment metadata for reproducible workflows."""

    name: str
    description: str = ""
    tags: tuple[str, ...] = ()
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation where possible."""

        return {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "provenance": self.provenance.to_dict(),
        }

