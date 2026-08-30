from __future__ import annotations

import os
from pathlib import Path

import pytest

from northstack.adapters import atomic_io


def _temps(path: Path) -> list[Path]:
    return list(path.parent.glob(f".{path.name}.*.tmp"))


def test_short_writes_are_completed(monkeypatch, tmp_path) -> None:
    target = tmp_path / "state.bin"
    target.write_bytes(b"old")
    real_write = os.write

    def short_write(fd: int, content: bytes | memoryview) -> int:
        return real_write(fd, content[:3])

    monkeypatch.setattr(atomic_io.os, "write", short_write)
    atomic_io.atomic_write_bytes(target, b"0123456789")
    assert target.read_bytes() == b"0123456789"
    assert not _temps(target)


@pytest.mark.parametrize("failure", ["write", "fsync", "replace"])
def test_pre_replace_failure_preserves_target_and_cleans_temp(
    monkeypatch, tmp_path, failure: str
) -> None:
    target = tmp_path / "state.bin"
    target.write_bytes(b"old")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError(failure)

    monkeypatch.setattr(atomic_io.os, failure, fail)
    with pytest.raises(OSError, match=failure):
        atomic_io.atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"old"
    assert not _temps(target)


def test_zero_byte_write_is_rejected(monkeypatch, tmp_path) -> None:
    target = tmp_path / "state.bin"
    target.write_bytes(b"old")
    monkeypatch.setattr(atomic_io.os, "write", lambda *_args: 0)
    with pytest.raises(OSError, match="zero"):
        atomic_io.atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"old"
    assert not _temps(target)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode semantics")
def test_existing_mode_is_preserved(tmp_path) -> None:
    target = tmp_path / "state.bin"
    target.write_bytes(b"old")
    target.chmod(0o640)
    atomic_io.atomic_write_bytes(target, b"new")
    assert target.stat().st_mode & 0o777 == 0o640


@pytest.mark.skipif(os.name == "nt", reason="directory fsync is POSIX-only")
def test_directory_fsync_failure_leaves_replaced_target_without_temp(monkeypatch, tmp_path) -> None:
    target = tmp_path / "state.bin"
    target.write_bytes(b"old")
    real_fsync, calls = os.fsync, 0

    def fail_directory(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync")
        real_fsync(fd)

    monkeypatch.setattr(atomic_io.os, "fsync", fail_directory)
    with pytest.raises(OSError, match="directory fsync"):
        atomic_io.atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"new"
    assert not _temps(target)
