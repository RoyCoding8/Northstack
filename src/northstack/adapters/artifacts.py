"""Content-addressed blob storage on the local filesystem."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

from northstack.adapters.atomic_io import atomic_write_bytes
from northstack.domain.outcome import ArtifactRef


class ArtifactCorruptionError(ValueError):
    pass


class ArtifactTooLargeError(ValueError):
    pass


class ArtifactStore:
    """Content-addressed blob store.

    Write computes sha256 and stores at ``<base>/<prefix>/<full_hash>``; read
    retrieves by digest; verify re-reads and recomputes.
    """

    def __init__(self, base_path: Path | str, *, max_read_bytes: int = 64 * 1024 * 1024) -> None:
        if max_read_bytes < 0:
            raise ValueError("max_read_bytes must be nonnegative")
        self._base = Path(base_path)
        self._max_read_bytes = max_read_bytes
        self._base.mkdir(parents=True, exist_ok=True)
        self._locks = tuple(threading.Lock() for _ in range(64))

    def write(self, content: bytes, *, media_type: str) -> ArtifactRef:
        """Write content, compute digest, return ArtifactRef."""
        digest_hash = hashlib.sha256(content).hexdigest()
        blob_path = self._path_for_digest(digest_hash)
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock(digest_hash):
            if not self._matches(blob_path, digest_hash):
                atomic_write_bytes(blob_path, content)
            if not self._matches(blob_path, digest_hash):
                raise ArtifactCorruptionError(
                    f"Artifact digest mismatch after write: sha256:{digest_hash}"
                )
        return ArtifactRef(
            digest=f"sha256:{digest_hash}",
            media_type=media_type,
            size_bytes=len(content),
        )

    def read(self, ref: ArtifactRef) -> bytes:
        """Read blob by ArtifactRef. Raises FileNotFoundError if missing."""
        digest_hash = ref.digest.split(":", 1)[1]
        with self._lock(digest_hash):
            try:
                with self._path_for_digest(digest_hash).open("rb") as stream:
                    content = stream.read(self._max_read_bytes + 1)
            except FileNotFoundError:
                raise FileNotFoundError(f"Artifact not found: {ref.digest}") from None
            if len(content) > self._max_read_bytes:
                raise ArtifactTooLargeError(
                    f"Artifact exceeds read limit of {self._max_read_bytes} bytes: {ref.digest}"
                )
            if hashlib.sha256(content).hexdigest() != digest_hash:
                raise ArtifactCorruptionError(f"Artifact digest mismatch: {ref.digest}")
        return content

    def read_by_digest(self, digest: str) -> bytes:
        """Read a blob by its bare ``sha256:...`` digest string.

        Raises FileNotFoundError if missing. Callers that track only digest
        strings (the evidence map) skip fabricating a full ArtifactRef.
        """
        return self.read(ArtifactRef(digest=digest, media_type="", size_bytes=0))

    def verify(self, ref: ArtifactRef) -> bool:
        """Verify stored blob matches its digest."""
        digest_hash = ref.digest.split(":", 1)[1]
        with self._lock(digest_hash):
            return self._matches(self._path_for_digest(digest_hash), digest_hash)

    def _lock(self, digest_hash: str) -> threading.Lock:
        return self._locks[int(digest_hash[:2], 16) % len(self._locks)]

    @staticmethod
    def _matches(path: Path, digest_hash: str) -> bool:
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1_048_576), b""):
                    digest.update(chunk)
            return digest.hexdigest() == digest_hash
        except OSError:
            return False

    def _path_for_digest(self, digest_hash: str) -> Path:
        return self._base / digest_hash[:2] / digest_hash
