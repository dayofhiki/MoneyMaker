"""Canonical private archives: close/fsync temporary bytes before atomic publication."""
from __future__ import annotations

import gzip
import io
import os
import tarfile

from .offline_source_inventory import confined, strict_json
from .r335_private_checkpoint import file_sha


def make_archive(root, output, *, day=None, secrets=()):
    if output.exists() or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("fresh_archive_and_regular_private_sources_required")
    if day is None:
        names = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
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
        raise ValueError("all_private_archive_sources_required")
    temp = output.with_name(output.name + ".writing")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            with gzip.GzipFile(filename="", fileobj=handle, mode="wb", compresslevel=3, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    for name in names:
                        path = confined(root, name)
                        body = path.read_bytes()
                        if any(s and s.encode() in body for s in secrets):
                            raise ValueError("credential_not_allowed_in_private_archive")
                        info = archive.gettarinfo(str(path), arcname=name)
                        info.mtime, info.uid, info.gid = 0, 0, 0
                        info.uname, info.gname, info.mode, info.pax_headers = "", "", 0o600, {}
                        info.size = len(body)
                        archive.addfile(info, io.BytesIO(body))
            handle.flush()
            os.fsync(handle.fileno())
        digest, size = file_sha(temp), temp.stat().st_size
        temp.rename(output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if file_sha(output) != digest or output.stat().st_size != size:
            raise ValueError("published_private_archive_changed")
        return {"archive_sha256": digest, "archive_bytes": size, "private_files": len(names)}
    except Exception:
        if temp.exists():
            temp.unlink()
        raise
