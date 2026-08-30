"""Crash-safe atomic file replacement primitives."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


def _write_all(fd: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(fd, remaining)
        if written <= 0:
            raise OSError("atomic write returned zero bytes")
        remaining = remaining[written:]


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, content: bytes, *, create_parents: bool = True) -> None:
    """Write bytes durably to a same-directory temp file, then replace."""
    path = Path(path)
    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    fd: int | None = None
    temp_path = ""
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    try:
        fd, temp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        if mode is not None:
            os.chmod(temp_path, mode)
        _write_all(fd, content)
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temp_path, path)
        temp_path = ""
        _fsync_directory(path.parent)
    finally:
        if fd is not None:
            os.close(fd)
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, content.encode(encoding))
