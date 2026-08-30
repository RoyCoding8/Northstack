"""Long-term memory: what earlier runs learned, retrievable by a later one.

The ledger already records everything a run did, but it is a hash-chained
audit log scoped to one run -- the wrong shape to ask "has anyone here solved
this before".  This is the other shape: namespaced, deduplicated, and searched
by relevance rather than replayed in order.

Storage is SQLite FTS5 (BM25 relevance, shipped with CPython, no service to
run).  A record is keyed by the hash of its text within its namespace, so the
same lesson learned in ten runs is stored once with a rising ``hits`` count --
which doubles as the tiebreak that floats repeatedly-confirmed knowledge
above a one-off.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from northstack.domain.secrets_policy import looks_like_secret_value

MAX_MEMORY_BYTES = 8192

_SCHEMA = """
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        namespace TEXT NOT NULL,
        digest TEXT NOT NULL,
        text TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT '',
        hits INTEGER NOT NULL DEFAULT 1,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        UNIQUE(namespace, digest)
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        text, content='memories', content_rowid='id'
    );
"""


class MemoryRecord(BaseModel):
    """One remembered fact and where it came from."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(max_length=MAX_MEMORY_BYTES)
    source: str = ""
    hits: int = Field(default=1, ge=1)
    created_at: float = 0.0


def _digest(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _match_query(query: str) -> str:
    """Render free text as an FTS5 OR query.

    Callers pass an objective or an error message, not query syntax, so every
    token is quoted -- an unescaped ``"`` or a bare ``AND`` would otherwise be
    parsed as an operator and raise instead of searching.
    """
    tokens = re.findall(r"\w+", query.lower())
    return " OR ".join(f'"{t}"' for t in tokens)


class SqliteMemory:
    """Namespaced long-term memory over SQLite FTS5."""

    def __init__(self, path: Path | str, busy_timeout: float = 5.0) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False, isolation_level=None, timeout=busy_timeout
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
        except BaseException:
            self._conn.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def remember(self, namespace: str, text: str, *, source: str = "") -> MemoryRecord | None:
        """Store one fact. Returns None for text not worth keeping.

        Re-remembering a known fact bumps its hit count instead of storing a
        duplicate, so a namespace grows with what is *new*, not with how often
        the pipeline ran.

        Text carrying a credential shape is dropped whole rather than scrubbed
        in place: unlike the ledger this store has no audit obligation, so
        losing one lesson costs nothing next to persisting a key and then
        replaying it into a later run's prompt.
        """
        text = text.strip()[:MAX_MEMORY_BYTES]
        if not text or looks_like_secret_value(text):
            return None
        digest, now = _digest(text), time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._conn.execute(
                    "INSERT INTO memories (namespace, digest, text, source, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(namespace, digest) DO UPDATE SET"
                    " hits = hits + 1, updated_at = excluded.updated_at"
                    " RETURNING id, hits, created_at",
                    (namespace, digest, text, source, now, now),
                )
                row = cursor.fetchone()
                if row["hits"] == 1:
                    self._conn.execute(
                        "INSERT INTO memories_fts (rowid, text) VALUES (?, ?)", (row["id"], text)
                    )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return MemoryRecord(
            text=text, source=source, hits=row["hits"], created_at=row["created_at"]
        )

    def recall(self, namespace: str, query: str, *, limit: int = 5) -> list[MemoryRecord]:
        """The most relevant facts in ``namespace``, best first.

        An empty query recalls nothing rather than everything: a caller with
        nothing to ask about should not be handed the whole namespace.
        """
        match = _match_query(query)
        if not match:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT m.text, m.source, m.hits, m.created_at FROM memories_fts f"
                " JOIN memories m ON m.id = f.rowid"
                " WHERE f.memories_fts MATCH ? AND m.namespace = ?"
                " ORDER BY bm25(memories_fts), m.hits DESC LIMIT ?",
                (match, namespace, limit),
            ).fetchall()
        return [MemoryRecord(**dict(r)) for r in rows]

    def forget(self, namespace: str) -> int:
        """Drop a whole namespace; returns how many records went."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                ids = [
                    r["id"]
                    for r in self._conn.execute(
                        "SELECT id FROM memories WHERE namespace = ?", (namespace,)
                    )
                ]
                self._conn.executemany(
                    "INSERT INTO memories_fts (memories_fts, rowid, text)"
                    " SELECT 'delete', id, text FROM memories WHERE id = ?",
                    [(i,) for i in ids],
                )
                self._conn.execute("DELETE FROM memories WHERE namespace = ?", (namespace,))
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return len(ids)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
