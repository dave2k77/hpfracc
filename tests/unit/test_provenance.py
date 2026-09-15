from __future__ import annotations

import json
import subprocess
import sys

from hpfracc.config import RuntimeTarget, current_provenance


def test_current_provenance_captures_runtime_context() -> None:
    provenance = current_provenance(
        commit_hash="abc123",
        timestamp_utc="2026-05-29T00:00:00+00:00",
        config={"case": "unit"},
    )

    payload = provenance.to_dict()

    assert payload["package_version"] == "0.1.0a1"
    assert payload["commit_hash"] == "abc123"
    assert payload["backend"] == "jax"
    assert payload["runtime_target"] in {target.value for target in RuntimeTarget}
    assert payload["timestamp_utc"] == "2026-05-29T00:00:00+00:00"
    assert payload["config"]["case"] == "unit"
    assert payload["config"]["jax_backend"] == payload["runtime_target"]
    assert "jax_version" in payload["config"]
    assert isinstance(payload["config"]["jax_enable_x64"], bool)
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


def test_package_and_metadata_import_without_loading_jax():
    # A fresh interpreter avoids mistaking JAX cached by other tests for a lazy
    # import. Block attempted imports, not merely membership in sys.modules.
    script = """
import importlib.abc
import json
import sys

class NoJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'jax' or fullname.startswith('jax.'):
            raise ModuleNotFoundError('JAX deliberately unavailable', name=fullname)

sys.meta_path.insert(0, NoJax())
import hpfracc as hp
assert hp.ops.validate_order(.5) == .5
json.dumps(hp.config.Provenance().to_dict())
json.dumps(hp.config.ExperimentMetadata(name='metadata-only').to_dict())
assert 'jax' not in sys.modules
try:
    hp.config.current_provenance()
except ModuleNotFoundError as exc:
    assert exc.name == 'jax'
else:
    raise AssertionError('Runtime provenance should explicitly request JAX')
"""
    subprocess.run([sys.executable, "-c", script], check=True,
                   capture_output=True, text=True, timeout=30)
