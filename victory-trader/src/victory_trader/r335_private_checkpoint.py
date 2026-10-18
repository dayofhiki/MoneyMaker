"""Private R335 archive/restore bytes only; no network, credentials or qualification."""
from __future__ import annotations

import hashlib
import os
import tarfile
from pathlib import Path

from .offline_source_inventory import confined, strict_json


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_archive(root, output, *, day=None, secrets=()):
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("private_evidence_symlink_not_allowed")
    if day is None:
        names = [p.relative_to(root).as_posix() for p in sorted(root.rglob("*")) if p.is_file()]
    else:
        progress = strict_json(confined(root, "progress.json").read_bytes())
        dates = [d for d in progress["days"] if d["day"] == day]
        if len(dates) != 1 or "end_elapsed_seconds" not in dates[0]:
            raise ValueError("completed_date_checkpoint_required")
        names = ["contract.json", "attestation.json", "progress.json"]
        for pair in progress["pairs"]:
            if pair["request"]["trading_day"] == day:
                for page in pair["pages"]:
                    if page["wire_file"] is not None:
                        names.append(page["wire_file"])
                    names.append(f"attempts/{page['attempt']:05d}.json")
        names = sorted(set(names))
    if not names or any(not confined(root, name).is_file() for name in names):
        raise ValueError("all_private_checkpoint_files_required")
    for name in names:
        path = confined(root, name)
        if any(s and s.encode() in path.read_bytes() for s in secrets):
            raise ValueError("credential_not_allowed_in_private_checkpoint")
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        with tarfile.open(fileobj=handle, mode="w:gz", compresslevel=3) as archive:
            for name in names:
                archive.add(confined(root, name), arcname=name, recursive=False)
    return {"archive_sha256": file_sha(output), "archive_bytes": output.stat().st_size,
            "private_files": len(names)}


def restore_archive(archive, destination):
    if destination.exists():
        raise ValueError("fresh_private_restore_directory_required")
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        names = [m.name for m in members]
        if (not members or len(members) > 50000 or len(set(names)) != len(names) or
                sum(m.size for m in members) > 16 * 1024**3 or any(
                    not m.isfile() or m.size < 0 or Path(m.name).is_absolute() or
                    ".." in Path(m.name).parts or "\\" in m.name or
                    Path(m.name).as_posix() != m.name for m in members)):
            raise ValueError("bounded_regular_private_archive_required")
        destination.mkdir(parents=True, mode=0o700)
        handle.extractall(destination, filter="data")
    for path in destination.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    return names


def compare_restored(root, restored, names):
    if {p.relative_to(restored).as_posix() for p in restored.rglob("*") if p.is_file()} != set(names):
        raise ValueError("restored_private_file_set_changed")
    if any(file_sha(confined(root, name)) != file_sha(confined(restored, name)) for name in names):
        raise ValueError("restored_private_bytes_changed")
    return True
