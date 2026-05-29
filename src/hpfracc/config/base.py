"""Base configuration and provenance objects."""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import jax

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


def current_provenance(
    *,
    commit_hash: str | None = None,
    timestamp_utc: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> Provenance:
    """Create best-effort runtime provenance for research outputs.

    Git metadata is optional: if ``commit_hash`` is not supplied, this helper
    attempts to read ``git rev-parse HEAD`` and falls back to ``None`` when git
    is unavailable or the source tree is not a repository.
    """

    backend = jax.default_backend()
    runtime_target = _runtime_target_from_backend(backend)
    runtime_config: dict[str, Any] = {
        "jax_backend": backend,
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    if config is not None:
        runtime_config.update(dict(config))

    return Provenance(
        package_version=__version__,
        commit_hash=commit_hash if commit_hash is not None else _git_commit_hash(),
        backend="jax",
        runtime_target=runtime_target,
        timestamp_utc=timestamp_utc or datetime.now(UTC).isoformat(),
        config=runtime_config,
    )


def _runtime_target_from_backend(backend: str) -> RuntimeTarget | None:
    """Map a JAX backend name onto the v0.1 runtime target enum."""

    try:
        return RuntimeTarget(backend)
    except ValueError:
        return None


def _git_commit_hash() -> str | None:
    """Return the current git commit hash when available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


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

