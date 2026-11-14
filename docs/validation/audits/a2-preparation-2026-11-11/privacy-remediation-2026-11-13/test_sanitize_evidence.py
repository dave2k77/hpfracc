"""Packaging controls use synthetic data, never private receipt contents."""
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


def load_helper():
    path = Path(__file__).with_name("sanitize_evidence.py")
    assert path.is_file(), "new packaging helper is absent"
    spec = importlib.util.spec_from_file_location("sanitize_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive(path, bodies):
    with tarfile.open(path, "w:gz") as out:
        for name, body in bodies:
            entry = tarfile.TarInfo(name)
            entry.size = len(body)
            out.addfile(entry, io.BytesIO(body))


def setup_case(tmp_path):
    # These are explicitly synthetic controls, not copied worker observations.
    bodies = [
        ("assessment.json", b'{"passed":false}\n'),
        ("source/check.py", b"assert 2 == 2\n"),
        ("run/telegram-vertical-progress.json", b'{"chat_id":"SYNTHETIC"}\n'),
    ]
    original = tmp_path / "original.tar.gz"
    archive(original, bodies)
    removed = bodies[-1]
    hold = {
        "original_repository_path": "audit/original.tar.gz",
        "original_bytes": original.stat().st_size,
        "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "structured_json_identifier_scan": {
            "original.tar.gz": [{
                "path": removed[0],
                "bytes": len(removed[1]),
                "sha256": hashlib.sha256(removed[1]).hexdigest(),
            }]
        },
    }
    return original, hold, bodies


def test_derivative_preserves_nonprivate_bodies_and_labels_omissions(tmp_path):
    helper = load_helper()
    original, hold, bodies = setup_case(tmp_path)
    before = original.read_bytes()
    bundle = tmp_path / "bundle"
    helper.build(original, hold, bundle)
    manifest_bytes = (bundle / "inventory.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["historical_replay_closure_complete"] is False
    assert manifest["original_sha256"] == hold["original_sha256"]
    assert manifest["original_member_count"] == len(bodies)
    expected_omissions = hold["structured_json_identifier_scan"]["original.tar.gz"]
    assert manifest["omitted"] == expected_omissions
    with tarfile.open(bundle / manifest["derivative_name"], "r:gz") as out:
        actual = {}
        for member in out.getmembers():
            stream = out.extractfile(member)
            assert stream is not None
            actual[member.name] = stream.read()
    assert actual == dict(bodies[:-1])
    assert original.read_bytes() == before
    assert helper.verify(bundle, hashlib.sha256(manifest_bytes).hexdigest())["verified"]


def test_verify_refuses_extra_bundle_files(tmp_path):
    helper = load_helper()
    original, hold, _ = setup_case(tmp_path)
    bundle = tmp_path / "bundle"
    helper.build(original, hold, bundle)
    pin = hashlib.sha256((bundle / "inventory.json").read_bytes()).hexdigest()
    (bundle / "unrelated-receipt.json").write_bytes(b"SYNTHETIC EXTRA FILE")
    with pytest.raises(ValueError, match="bundle file allowlist"):
        helper.verify(bundle, pin)


@pytest.mark.parametrize("mutation", ["missing", "changed", "extra", "private"])
def test_resigned_archive_member_tamper_is_rejected(tmp_path, mutation):
    helper = load_helper()
    original, hold, bodies = setup_case(tmp_path)
    bundle = tmp_path / "bundle"
    manifest = helper.build(original, hold, bundle)
    changed = list(bodies[:-1])
    if mutation == "missing":
        changed.pop()
    elif mutation == "changed":
        changed[0] = (changed[0][0], b'{"passed":true}\n')
    elif mutation == "extra":
        changed.append(("unexpected.txt", b"SYNTHETIC EXTRA"))
    else:
        changed.append(bodies[-1])
    target = bundle / helper.DERIVATIVE
    archive(target, changed)
    manifest["derivative_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest["derivative_bytes"] = target.stat().st_size
    raw = json.dumps(manifest).encode()
    (bundle / "inventory.json").write_bytes(raw)
    with pytest.raises(ValueError, match="retained member mismatch"):
        helper.verify(bundle, hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a//b"])
def test_unsafe_member_names_are_rejected(tmp_path, name):
    helper = load_helper()
    target = tmp_path / "unsafe.tar.gz"
    archive(target, [(name, b"SYNTHETIC")])
    with pytest.raises(ValueError, match="unsafe archive member"):
        helper.unpack(target.read_bytes())


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE])
def test_nonregular_members_are_rejected(tmp_path, kind):
    helper = load_helper()
    target = tmp_path / "nonregular.tar.gz"
    with tarfile.open(target, "w:gz") as out:
        entry = tarfile.TarInfo("safe-looking-name")
        entry.type = kind
        entry.linkname = "outside"
        out.addfile(entry)
    with pytest.raises(ValueError, match="nonregular or duplicate"):
        helper.unpack(target.read_bytes())


def test_duplicate_member_is_rejected(tmp_path):
    helper = load_helper()
    target = tmp_path / "duplicate.tar.gz"
    archive(target, [("same", b"one"), ("same", b"two")])
    with pytest.raises(ValueError, match="nonregular or duplicate"):
        helper.unpack(target.read_bytes())


def test_existing_output_is_not_overwritten(tmp_path):
    helper = load_helper()
    original, hold, _ = setup_case(tmp_path)
    bundle = tmp_path / "bundle"
    helper.build(original, hold, bundle)
    before = {p.name: p.read_bytes() for p in bundle.iterdir()}
    with pytest.raises(FileExistsError):
        helper.build(original, hold, bundle)
    assert {p.name: p.read_bytes() for p in bundle.iterdir()} == before


def test_symlink_input_is_refused(tmp_path):
    helper = load_helper()
    original, hold, _ = setup_case(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(original)
    with pytest.raises(ValueError, match="symlink path refused"):
        helper.build(link, hold, tmp_path / "bundle")


def test_symlink_output_parent_is_refused(tmp_path):
    helper = load_helper()
    original, hold, _ = setup_case(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink output refused"):
        helper.build(original, hold, link / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_wrong_original_digest_is_refused_before_output(tmp_path):
    helper = load_helper()
    original, hold, _ = setup_case(tmp_path)
    hold["original_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="original archive identity"):
        helper.build(original, hold, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_wrong_omission_digest_is_refused_before_output(tmp_path):
    helper = load_helper()
    original, hold, _ = setup_case(tmp_path)
    hold["structured_json_identifier_scan"]["original.tar.gz"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="omission body identity"):
        helper.build(original, hold, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_external_inventory_pin_is_required(tmp_path):
    helper = load_helper()
    original, hold, _ = setup_case(tmp_path)
    bundle = tmp_path / "bundle"
    helper.build(original, hold, bundle)
    with pytest.raises(ValueError, match="inventory identity"):
        helper.verify(bundle, "0" * 64)


def test_resigned_partition_tamper_is_rejected(tmp_path):
    helper = load_helper()
    original, hold, _ = setup_case(tmp_path)
    bundle = tmp_path / "bundle"
    manifest = helper.build(original, hold, bundle)
    manifest["omitted"] = []
    raw = json.dumps(manifest).encode()
    (bundle / "inventory.json").write_bytes(raw)
    with pytest.raises(ValueError, match="partition mismatch"):
        helper.verify(bundle, hashlib.sha256(raw).hexdigest())


def test_resigned_replay_completeness_claim_is_rejected(tmp_path):
    helper = load_helper()
    original, hold, _ = setup_case(tmp_path)
    bundle = tmp_path / "bundle"
    manifest = helper.build(original, hold, bundle)
    manifest["historical_replay_closure_complete"] = True
    raw = json.dumps(manifest).encode()
    (bundle / "inventory.json").write_bytes(raw)
    with pytest.raises(ValueError, match="historical boundary mismatch"):
        helper.verify(bundle, hashlib.sha256(raw).hexdigest())


def test_portable_cli_without_original(tmp_path):
    helper = load_helper()
    original, hold, _ = setup_case(tmp_path)
    bundle = tmp_path / "bundle"
    helper.build(original, hold, bundle)
    portable = tmp_path / "portable"
    portable.mkdir()
    shutil.copytree(bundle, portable / "bundle")
    shutil.copyfile(Path(__file__).with_name("sanitize_evidence.py"),
                    portable / "sanitize_evidence.py")
    pin = hashlib.sha256((bundle / "inventory.json").read_bytes()).hexdigest()
    original.unlink()
    shutil.rmtree(bundle)
    result = subprocess.run(
        [sys.executable, "-B", "sanitize_evidence.py", "verify", "--bundle", "bundle",
         "--inventory-sha256", pin], cwd=portable, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "verified": True, "retained_members": 2, "omitted_members": 1,
    }
