"""Append-only event ledger backed by SQLite WAL with a hash chain.

Public seam:
  - Ledger(path) -> append(event), events(run_id), verify_integrity(run_id)

The ledger is the source of truth.  Everything else is a derived view -- and
this module knows nothing about those views: folding events into a ``RunState``
lives in ``application.replay``, so storage no longer depends on a read model.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Self, cast

from pydantic import BaseModel, ValidationError

from northstack.domain.secrets_policy import is_secret_field_name, looks_like_secret_value
from northstack.events.catalog import (
    CURRENT_SCHEMA_VERSION,
    EventKind,
    EventPayload,
    OutcomeEmitted,
    PayloadBase,
    StatusChanged,
    parse_payload,
)
from northstack.events.envelope import EventEnvelope
from northstack.events.errors import LedgerCorruption
from northstack.events.upcast import upcast


def _redact_recursive(value: Any) -> Any:
    """Recursively redact secrets in dicts, lists, and tuples."""
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if is_secret_field_name(str(k)) or (isinstance(v, str) and looks_like_secret_value(v)):
                result[k] = "[REDACTED]"
            else:
                result[k] = _redact_recursive(v)
        return result
    if isinstance(value, (list, tuple)):
        redacted = []
        for item in value:
            if isinstance(item, str) and looks_like_secret_value(item):
                redacted.append("[REDACTED]")
            else:
                redacted.append(_redact_recursive(item))
        if isinstance(value, tuple):
            return tuple(redacted)
        return redacted
    if isinstance(value, str) and looks_like_secret_value(value):
        return "[REDACTED]"
    return value


def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a recursively redacted copy of payload."""
    return cast(dict[str, Any], _redact_recursive(payload))


MAX_SCHEMA_VERSION = CURRENT_SCHEMA_VERSION  # payload-owned; see events.catalog
DATABASE_SCHEMA_VERSION = 1


class UnsupportedDatabaseSchema(ValueError):
    def __init__(self, found: int) -> None:
        self.found = found
        super().__init__(
            f"Unsupported database schema version {found}; maximum is {DATABASE_SCHEMA_VERSION}"
        )


class IntegrityResult(BaseModel):
    ok: bool
    events_checked: int = 0
    error_at_seq: int | None = None
    error_category: str | None = None
    error_message: str | None = None


class RunSummary(BaseModel):
    """Run-history row: lightweight projection of a run's ledger state.

    Derived from the events table without full replay so the run list page
    stays cheap.  ``status`` is the most recent ``status_changed`` payload
    (else "unknown"); ``outcome`` is the most recent ``outcome_emitted``
    payload (None until the run terminates).
    """

    run_id: str
    status: str = "unknown"
    outcome: str | None = None
    start_time: float
    last_event_time: float
    event_count: int


class Ledger:
    """Append-only event ledger backed by SQLite WAL mode.

    Thread-safe for append: uses BEGIN IMMEDIATE + an in-process lock to
    prevent concurrent writes from interleaving the read-modify-write cycle.

    Invariants:
      - Per-run sequence numbers are monotonically increasing.
      - Each event's prev_hash matches the previous event's hash_chain.
      - Payloads are redacted BEFORE hashing and storage.
      - The stored hash covers exactly the stored (redacted) representation.
      - Schema version is validated against MAX_SCHEMA_VERSION.
    """

    def __init__(self, path: Path | str, busy_timeout: float = 5.0) -> None:
        if busy_timeout < 0:
            raise ValueError("busy_timeout must be non-negative")
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            isolation_level=None,  # manual transaction control
            timeout=busy_timeout,
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._closed = False
        try:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version > DATABASE_SCHEMA_VERSION:
                raise UnsupportedDatabaseSchema(version)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._init_schema(version)
        except BaseException:
            self._closed = True
            self._conn.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _init_schema(self, version: int) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                schema_version INTEGER NOT NULL DEFAULT 1,
                timestamp REAL NOT NULL,
                prev_hash TEXT NOT NULL DEFAULT '',
                hash_chain TEXT NOT NULL,
                UNIQUE(run_id, seq)
            );
            CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id, seq);
            CREATE TABLE IF NOT EXISTS run_tombstones (
                run_id TEXT PRIMARY KEY,
                deleted_at REAL NOT NULL
            );
        """)
        if version < DATABASE_SCHEMA_VERSION:
            self._conn.execute(f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION}")
        self._conn.commit()

    def append(self, event: EventEnvelope) -> EventEnvelope:
        """Append an event to the ledger atomically.

        The event is redacted first, then hashed over the stored representation.
        Returns a fresh persisted EventEnvelope with the correct hash_chain.
        """
        with self._lock:
            self._ensure_open()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                return self._store(event)
            except BaseException:
                self._rollback()
                raise

    def append_next(self, run_id: str, payload: EventPayload) -> EventEnvelope:
        """Atomically allocate and persist the next sequence number."""
        with self._lock:
            self._ensure_open()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                prev = self._prev_event(run_id)
                event = EventEnvelope(
                    run_id=run_id,
                    seq=1 if prev is None else prev["seq"] + 1,
                    payload=payload,
                    prev_hash="" if prev is None else prev["hash_chain"],
                )
                return self._store(event)
            except BaseException:
                self._rollback()
                raise

    def _store(self, event: EventEnvelope) -> EventEnvelope:
        if event.schema_version > MAX_SCHEMA_VERSION:
            raise ValueError(
                f"Schema version {event.schema_version} exceeds max {MAX_SCHEMA_VERSION}"
            )
        payload = _redact_payload(event.payload.model_dump(mode="json"))
        prev = self._prev_event(event.run_id)
        if prev is None:
            if event.seq != 1:
                raise ValueError(
                    f"First event for run {event.run_id} must have seq=1, got {event.seq}"
                )
            prev_hash = ""
        else:
            if event.seq != prev["seq"] + 1:
                raise ValueError(
                    f"Sequence must be monotonic: expected {prev['seq'] + 1}, got {event.seq}"
                )
            if event.prev_hash and event.prev_hash != prev["hash_chain"]:
                raise ValueError(
                    f"Hash chain broken: expected prev_hash={prev['hash_chain'][:16]}..., "
                    f"got {event.prev_hash[:16]}..."
                )
            prev_hash = prev["hash_chain"]
        stored_hash = _compute_hash(
            event.run_id,
            event.seq,
            event.kind.value,
            payload,
            prev_hash,
            event.schema_version,
            event.timestamp,
        )
        persisted = EventEnvelope(
            run_id=event.run_id,
            seq=event.seq,
            payload=parse_payload(payload),
            timestamp=event.timestamp,
            prev_hash=prev_hash,
            hash_chain=stored_hash,
        )
        self._conn.execute(
            "INSERT INTO events "
            "(run_id, seq, kind, payload, schema_version, timestamp, prev_hash, hash_chain) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                persisted.run_id,
                persisted.seq,
                persisted.kind.value,
                json.dumps(payload, sort_keys=True),
                persisted.schema_version,
                persisted.timestamp,
                persisted.prev_hash,
                persisted.hash_chain,
            ),
        )
        self._conn.commit()
        return persisted

    def _rollback(self) -> None:
        if self._conn.in_transaction:
            self._conn.rollback()

    def events(self, run_id: str) -> list[EventEnvelope]:
        """Return all events for a run, ordered by sequence."""
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY seq",
                (run_id,),
            ).fetchall()
            return [self._row_to_event(r) for r in rows]

    def events_since(self, run_id: str, since: int = 0, limit: int = 500) -> list[EventEnvelope]:
        """Events with seq > since, ordered by seq, bounded by limit (SQL pushdown).

        Uses the (run_id, seq) index instead of loading the whole run into Python.
        """
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(
                "SELECT * FROM events WHERE run_id = ? AND seq > ? ORDER BY seq LIMIT ?",
                (run_id, since, limit),
            ).fetchall()
            return [self._row_to_event(r) for r in rows]

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[RunSummary]:
        with self._lock:
            self._ensure_open()
            return self._list_runs(limit, offset)

    def tombstone_run(self, run_id: str) -> None:
        """Hide a run from history listings. The append-only events are kept:
        replay, integrity verification, and CLI inspect keep working."""
        with self._lock:
            self._ensure_open()
            self._conn.execute(
                "INSERT OR IGNORE INTO run_tombstones (run_id, deleted_at) VALUES (?, ?)",
                (run_id, time.time()),
            )
            self._conn.commit()

    def tombstoned_run_ids(self) -> set[str]:
        with self._lock:
            self._ensure_open()
            return {row[0] for row in self._conn.execute("SELECT run_id FROM run_tombstones")}

    def _list_runs(self, limit: int, offset: int) -> list[RunSummary]:
        """Return a lightweight run-history list, newest first.

        One grouped query yields per-run start_time (MIN timestamp),
        last_event_time (MAX timestamp), and event_count (MAX seq).  Status
        and outcome then come from the single most recent event of each
        relevant kind per run (a cheap bounded scan), so this does NOT do a
        full replay per run.
        """
        grouped = self._conn.execute(
            "SELECT run_id, MIN(timestamp) AS start_time, "
            "MAX(timestamp) AS last_event_time, MAX(seq) AS n_events_max, "
            "MAX(id) AS last_rowid, COUNT(*) AS n_events "
            "FROM events WHERE run_id NOT IN (SELECT run_id FROM run_tombstones) GROUP BY run_id "
            "ORDER BY last_rowid DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        if not grouped:
            return []

        ids = [r["run_id"] for r in grouped]
        placeholders = ",".join("?" for _ in ids)

        status_rows = self._conn.execute(
            f"SELECT run_id, seq, kind, payload FROM events WHERE kind = ? AND run_id "  # noqa: S608
            f"IN ({placeholders}) ORDER BY seq DESC",
            (EventKind.STATUS_CHANGED.value, *ids),
        ).fetchall()
        latest_status: dict[str, str] = {}
        for row in status_rows:
            rid = row["run_id"]
            if rid not in latest_status:
                latest_status[rid] = _typed_payload(row, StatusChanged).status.value

        outcome_rows = self._conn.execute(
            f"SELECT run_id, seq, kind, payload FROM events WHERE kind = ? AND run_id "  # noqa: S608
            f"IN ({placeholders}) ORDER BY seq DESC",
            (EventKind.OUTCOME_EMITTED.value, *ids),
        ).fetchall()
        latest_outcome: dict[str, str | None] = {}
        for row in outcome_rows:
            rid = row["run_id"]
            if rid not in latest_outcome:
                latest_outcome[rid] = _typed_payload(row, OutcomeEmitted).outcome.value

        summaries: list[RunSummary] = []
        for row in grouped:
            rid = row["run_id"]
            summaries.append(
                RunSummary(
                    run_id=rid,
                    status=latest_status.get(rid, "unknown"),
                    outcome=latest_outcome.get(rid),
                    start_time=row["start_time"],
                    last_event_time=row["last_event_time"],
                    event_count=row["n_events"],
                )
            )
        return summaries

    def verify_integrity(self, run_id: str) -> IntegrityResult:
        with self._lock:
            self._ensure_open()
            return self._verify_integrity(run_id)

    def _verify_integrity(self, run_id: str) -> IntegrityResult:
        """Verify the hash chain: recompute every event hash AND check prev links.

        This catches both chain-link corruption and payload/timestamp/kind
        tampering at any position.
        """
        rows = self._conn.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()

        if not rows:
            return IntegrityResult(ok=True, events_checked=0)

        prev_hash = ""
        for i, row in enumerate(rows):
            if row["prev_hash"] != prev_hash:
                return IntegrityResult(
                    ok=False,
                    events_checked=i,
                    error_at_seq=row["seq"],
                    error_category="prev_hash",
                    error_message=(
                        f"Prev-hash link broken at seq {row['seq']}: "
                        f"expected {prev_hash[:16] if prev_hash else '(empty)'}..., "
                        f"got {row['prev_hash'][:16] if row['prev_hash'] else '(empty)'}..."
                    ),
                )

            try:
                payload = json.loads(row["payload"])
            except UnicodeDecodeError as exc:
                return IntegrityResult(
                    ok=False,
                    events_checked=i,
                    error_at_seq=row["seq"],
                    error_category="payload_encoding",
                    error_message=f"Payload encoding is invalid at seq {row['seq']}: {exc}",
                )
            except (json.JSONDecodeError, TypeError) as exc:
                return IntegrityResult(
                    ok=False,
                    events_checked=i,
                    error_at_seq=row["seq"],
                    error_category="payload_json",
                    error_message=f"Payload is not JSON at seq {row['seq']}: {exc}",
                )
            if not isinstance(payload, dict):
                return IntegrityResult(
                    ok=False,
                    events_checked=i,
                    error_at_seq=row["seq"],
                    error_category="payload_type",
                    error_message=(
                        f"Payload at seq {row['seq']} is {type(payload).__name__}, not an object"
                    ),
                )
            recomputed = _compute_hash(
                row["run_id"],
                row["seq"],
                row["kind"],
                payload,
                row["prev_hash"],
                row["schema_version"],
                row["timestamp"],
            )
            if recomputed != row["hash_chain"]:
                return IntegrityResult(
                    ok=False,
                    events_checked=i,
                    error_at_seq=row["seq"],
                    error_category="hash_mismatch",
                    error_message=(
                        f"Hash mismatch at seq {row['seq']}: "
                        f"stored={row['hash_chain'][:16]}..., "
                        f"recomputed={recomputed[:16]}..."
                    ),
                )

            try:
                _row_payload(row)
            except LedgerCorruption as exc:
                return IntegrityResult(
                    ok=False,
                    events_checked=i,
                    error_at_seq=row["seq"],
                    error_category="event_schema",
                    error_message=str(exc),
                )

            prev_hash = row["hash_chain"]

        return IntegrityResult(ok=True, events_checked=len(rows))

    def _prev_event(self, run_id: str) -> sqlite3.Row | None:
        """Get the last event for a run."""
        row: sqlite3.Row | None = self._conn.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY seq DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return row

    def _row_to_event(self, row: sqlite3.Row) -> EventEnvelope:
        return EventEnvelope(
            run_id=row["run_id"],
            seq=row["seq"],
            payload=_row_payload(row),
            timestamp=row["timestamp"],
            prev_hash=row["prev_hash"],
            hash_chain=row["hash_chain"],
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Ledger is closed")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()


def _row_payload(row: sqlite3.Row) -> EventPayload:
    """Parse one stored row into its catalog payload, or refuse it.

    Every failure names the seq and kind: a renamed field, an unknown kind, or
    a payload written before the typed catalog must stop the read rather than
    degrade into plausible-looking state.
    """
    seq, kind_str = row["seq"], row["kind"]
    try:
        kind = EventKind(kind_str)
    except ValueError as exc:
        raise LedgerCorruption(seq, kind_str, "unknown event kind") from exc
    try:
        data = json.loads(row["payload"])
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise LedgerCorruption(seq, kind_str, f"payload is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LedgerCorruption(seq, kind_str, f"payload is {type(data).__name__}, not an object")
    if "kind" not in data:
        raise LedgerCorruption(
            seq, kind_str, "payload predates the typed event catalog; recreate the ledger"
        )
    try:
        return parse_payload(upcast(kind, data, seq=seq))
    except ValidationError as exc:
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:3]
        )
        raise LedgerCorruption(seq, kind_str, detail) from exc


def _typed_payload[T: PayloadBase](row: sqlite3.Row, expected: type[T]) -> T:
    """Parse a row known by its ``kind`` column to be of one payload type."""
    payload = _row_payload(row)
    if not isinstance(payload, expected):
        raise LedgerCorruption(
            row["seq"], row["kind"], f"expected {expected.__name__}, got {type(payload).__name__}"
        )
    return payload


def _compute_hash(
    run_id: str,
    seq: int,
    kind: str,
    payload: dict[str, Any],
    prev_hash: str,
    schema_version: int,
    timestamp: float,
) -> str:
    """Deterministic SHA-256 hash over event fields."""
    canonical = json.dumps(
        {
            "run_id": run_id,
            "seq": seq,
            "kind": kind,
            "payload": payload,
            "prev_hash": prev_hash,
            "schema_version": schema_version,
            "timestamp": timestamp,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
