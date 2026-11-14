"""Fixed-task evidence packaging, not a numerical or universal privacy validator.

Never extract archives. Use owner-controlled paths; pin inventory bytes externally.
Original manifests inside the derivative remain historical and may reference the
explicitly omitted receipts. They are not complete replay manifests for this copy.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import stat
import tarfile
from pathlib import Path, PurePosixPath

DERIVATIVE = "single-trace-independent-review-failed-evidence.sanitized.tar.gz"
MAX_ARCHIVE = 64 * 1024 * 1024
MAX_MEMBERS = 1000


def digest(body):
    return hashlib.sha256(body).hexdigest()


def read_owned_path(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink path refused")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_ARCHIVE:
            raise ValueError("not a bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            body = stream.read(MAX_ARCHIVE + 1)
        if len(body) > MAX_ARCHIVE:
            raise ValueError("file limit exceeded")
        return body
    finally:
        os.close(fd)


def safe_name(name):
    parts = PurePosixPath(name).parts
    if (not name or "\\" in name or name.startswith("/")
            or any(p in ("", ".", "..") for p in name.split("/"))
            or not parts or any(ord(c) < 32 or ord(c) > 126 for c in name)):
        raise ValueError("unsafe archive member name")


def unpack(body):
    result = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as archive:
        for member in archive:
            safe_name(member.name)
            if not member.isfile() or member.name in result:
                raise ValueError("nonregular or duplicate archive member")
            total += member.size
            if total > MAX_ARCHIVE or len(result) >= MAX_MEMBERS:
                raise ValueError("archive expansion limit exceeded")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("missing regular member stream")
            value = stream.read()
            if len(value) != member.size:
                raise ValueError("short archive member")
            result[member.name] = value
    return result


def descriptors(bodies):
    return [
        {"path": name, "bytes": len(body), "sha256": digest(body)}
        for name, body in sorted(bodies.items())
    ]


def pack(bodies):
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as out:
            for name, body in sorted(bodies.items()):
                entry = tarfile.TarInfo(name)
                entry.size = len(body)
                entry.mode = 0o644
                out.addfile(entry, io.BytesIO(body))
    return output.getvalue()


def strict_json(body):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError("duplicate JSON key")
            obj[key] = value
        return obj

    return json.loads(body, object_pairs_hook=pairs)


def build(original, hold, output):
    original_body = read_owned_path(original)
    if (digest(original_body) != hold["original_sha256"]
            or len(original_body) != hold["original_bytes"]):
        raise ValueError("original archive identity mismatch")
    source = unpack(original_body)
    original_name = Path(hold["original_repository_path"]).name
    omitted = hold["structured_json_identifier_scan"][original_name]
    omissions = {}
    for row in omitted:
        name = row["path"]
        if name in omissions or name not in source:
            raise ValueError("duplicate or missing omission")
        if len(source[name]) != row["bytes"] or digest(source[name]) != row["sha256"]:
            raise ValueError("omission body identity mismatch")
        omissions[name] = source[name]
    retained = {name: body for name, body in source.items() if name not in omissions}
    derivative = pack(retained)
    inventory = {
        "schema": "hpfracc-sanitized-archive-v1",
        "status": "SANITIZED DERIVATIVE; NOT THE ORIGINAL ARCHIVE",
        "original_name": original_name,
        "original_sha256": digest(original_body),
        "original_bytes": len(original_body),
        "original_member_count": len(source),
        "original_members": descriptors(source),
        "derivative_name": DERIVATIVE,
        "derivative_sha256": digest(derivative),
        "derivative_bytes": len(derivative),
        "retained": descriptors(retained),
        "omitted": descriptors(omissions),
        "historical_replay_closure_complete": False,
        "historical_manifests_are_authoritative_for_derivative": False,
        "privacy_scan_is_separate": True,
        "tar_metadata": "Canonical uid/gid/mtime zero, names empty, regular mode 644",
    }
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("symlink output refused")
    output.mkdir(mode=0o700, exist_ok=False)
    with (output / DERIVATIVE).open("xb") as stream:
        stream.write(derivative)
    raw_inventory = (json.dumps(inventory, indent=2) + "\n").encode()
    with (output / "inventory.json").open("xb") as stream:
        stream.write(raw_inventory)
    verify(output, digest(raw_inventory))
    return inventory


def verify(bundle, inventory_sha256):
    bundle = Path(bundle)
    if {path.name for path in bundle.iterdir()} != {"inventory.json", DERIVATIVE}:
        raise ValueError("bundle file allowlist mismatch")
    raw_inventory = read_owned_path(bundle / "inventory.json")
    if digest(raw_inventory) != inventory_sha256:
        raise ValueError("inventory identity mismatch")
    inventory = strict_json(raw_inventory)
    if (inventory["schema"] != "hpfracc-sanitized-archive-v1"
            or inventory["derivative_name"] != DERIVATIVE
            or inventory["historical_replay_closure_complete"] is not False
            or inventory["historical_manifests_are_authoritative_for_derivative"]
            is not False):
        raise ValueError("derivative identity or historical boundary mismatch")
    body = read_owned_path(bundle / DERIVATIVE)
    if (digest(body) != inventory["derivative_sha256"]
            or len(body) != inventory["derivative_bytes"]):
        raise ValueError("derivative archive identity mismatch")
    actual = unpack(body)
    if descriptors(actual) != inventory["retained"]:
        raise ValueError("retained member mismatch")
    original = inventory["original_members"]
    partition = inventory["retained"] + inventory["omitted"]
    names = [row["path"] for row in partition]
    if (len(names) != len(set(names))
            or sorted(partition, key=lambda row: row["path"]) != original
            or type(inventory["original_member_count"]) is not int
            or len(original) != inventory["original_member_count"]):
        raise ValueError("original-to-derivative partition mismatch")
    return {"verified": True, "retained_members": len(actual),
            "omitted_members": len(inventory["omitted"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("build")
    create.add_argument("--original", type=Path, required=True)
    create.add_argument("--hold", type=Path, required=True)
    create.add_argument("--hold-sha256", required=True)
    create.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--bundle", type=Path, required=True)
    check.add_argument("--inventory-sha256", required=True)
    args = parser.parse_args()
    if args.action == "build":
        hold_bytes = read_owned_path(args.hold)
        if digest(hold_bytes) != args.hold_sha256:
            raise ValueError("hold record identity mismatch")
        result = build(args.original, strict_json(hold_bytes), args.output)
        print(json.dumps({"created": True, "retained": len(result["retained"]),
                          "omitted": len(result["omitted"])}))
    else:
        print(json.dumps(verify(args.bundle, args.inventory_sha256)))


if __name__ == "__main__":
    main()
