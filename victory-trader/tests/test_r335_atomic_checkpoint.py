import os
import threading

from victory_trader.r335_atomic_checkpoint import make_archive
from victory_trader.r335_private_checkpoint import restore_archive, compare_restored


def test_canonical_archive_identical_after_filename_and_filesystem_time_changes(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "page.wire").write_bytes(b'{"exact":"1.00","time":123456789}')
    one, two = tmp_path / "one.tar.gz", tmp_path / "two.tar.gz"
    a = make_archive(root, one)
    os.utime(root / "page.wire", (123, 123))
    b = make_archive(root, two)
    assert a == b and one.read_bytes() == two.read_bytes()
    restored = tmp_path / "restored"
    names = restore_archive(two, restored)
    assert compare_restored(root, restored, names)


def test_final_archive_not_visible_while_temporary_bytes_are_written(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "page.wire").write_bytes(b"exact")
    archive = tmp_path / "final.tar.gz"
    reached, release = threading.Event(), threading.Event()
    original = os.fsync
    def fsync(fd):
        if not reached.is_set():
            reached.set()
            assert release.wait(5)
        return original(fd)
    monkeypatch.setattr(os, "fsync", fsync)
    results = []
    worker = threading.Thread(target=lambda: results.append(make_archive(root, archive)))
    worker.start()
    assert reached.wait(5) and not archive.exists()
    assert archive.with_name(archive.name + ".writing").exists()
    release.set()
    worker.join(5)
    assert len(results) == 1 and archive.exists()
    assert not archive.with_name(archive.name + ".writing").exists()
