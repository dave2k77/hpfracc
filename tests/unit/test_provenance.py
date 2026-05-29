from __future__ import annotations

import json
import subprocess

from hpfracc.config import RuntimeTarget, current_provenance


def test_current_provenance_captures_runtime_context() -> None:
    provenance = current_provenance(
        commit_hash="abc123",
        timestamp_utc="2026-05-29T00:00:00+00:00",
        config={"case": "unit"},
    )

    payload = provenance.to_dict()

    assert payload["package_version"] == "0.1.0a0"
    assert payload["commit_hash"] == "abc123"
    assert payload["backend"] == "jax"
    assert payload["runtime_target"] in {target.value for target in RuntimeTarget}
    assert payload["timestamp_utc"] == "2026-05-29T00:00:00+00:00"
    assert payload["config"]["case"] == "unit"
    assert payload["config"]["jax_backend"] == payload["runtime_target"]
    assert "jax_version" in payload["config"]
    assert "python_version" in payload["config"]
    assert "platform" in payload["config"]

    json.dumps(payload)


def test_current_provenance_does_not_require_git(monkeypatch) -> None:
    def raise_git_unavailable(
        *args: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        raise OSError("git unavailable")

    monkeypatch.setattr(subprocess, "run", raise_git_unavailable)

    provenance = current_provenance(timestamp_utc="2026-05-29T00:00:00+00:00")

    assert provenance.commit_hash is None
    assert provenance.runtime_target in set(RuntimeTarget)
    json.dumps(provenance.to_dict())
