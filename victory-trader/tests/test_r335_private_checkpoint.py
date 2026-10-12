import io
import tarfile

import pytest

from victory_trader.r335_private_checkpoint import make_archive, restore_archive, compare_restored, file_sha


def test_private_archive_retrieval_bytes_and_file_modes(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "one.wire").write_bytes(b'{"exact":"1.00"}')
    (root / "two.json").write_bytes(b'{"time":123456789}')
    archive = tmp_path / "private.tar.gz"
    proof = make_archive(root, archive)
    assert proof["private_files"] == 2 and proof["archive_sha256"] == file_sha(archive)
    restored = tmp_path / "restored"
    names = restore_archive(archive, restored)
    assert compare_restored(root, restored, names)
    assert archive.stat().st_mode & 0o777 == 0o600
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in restored.rglob("*") if p.is_file())
    (restored / "one.wire").write_bytes(b"changed")
    with pytest.raises(ValueError, match="restored_private_bytes_changed"):
        compare_restored(root, restored, names)


def test_secrets_and_symlinks_block_archive_creation(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "one").write_bytes(b"synthetic-secret")
    with pytest.raises(ValueError, match="credential_not_allowed"):
        make_archive(root, tmp_path / "bad.tar.gz", secrets=("synthetic-secret",))
    (root / "alias").symlink_to(root / "one")
    with pytest.raises(ValueError, match="symlink_not_allowed"):
        make_archive(root, tmp_path / "bad.tar.gz")


@pytest.mark.parametrize("kind", ["traversal", "symlink", "duplicate"])
def test_unsafe_or_duplicate_archive_members_rejected(tmp_path, kind):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("../outside" if kind == "traversal" else "item")
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/outside"
            handle.addfile(member)
        else:
            member.size = 1
            handle.addfile(member, io.BytesIO(b"x"))
            if kind == "duplicate":
                handle.addfile(member, io.BytesIO(b"y"))
    with pytest.raises(ValueError, match="bounded_regular_private_archive_required"):
        restore_archive(archive, tmp_path / "restore")
