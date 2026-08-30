"""Tests for ledger.py at the public seams.

Seams tested:
  1. Ledger.append -- serialized append with monotonic sequence + hash chain
  2. Ledger.append -- rejects hash tampering (prev_hash mismatch)
  3. Ledger.append -- recursive payload redaction for secret fields/values
  4. Ledger.append -- returned envelope has redacted payload
  5. Ledger.append -- schema/event-version checks
  6. Ledger.events -- ordered retrieval by run_id
  7. Ledger.replay -- projection to RunState with transition validation
  8. Ledger.verify_integrity -- recomputes every hash + prev links
  9. Ledger.verify_integrity -- detects payload/kind/timestamp/hash tampering
  10. Concurrent append -- no duplicate/gapped sequences
  11. Context manager support
  12. ArtifactStore.write / read / verify -- content-addressed blobs
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from pathlib import Path
from threading import Event

import pytest
from pydantic import ValidationError

from northstack.adapters import sqlite_ledger as ledger_module
from northstack.adapters import artifacts as artifacts_module
from northstack.adapters.artifacts import ArtifactStore
from northstack.adapters.sqlite_ledger import Ledger
from northstack.application.replay import replay_run
from northstack.domain import (
    ArtifactRef,
    Budget,
    CellStatus,
    GraphCell,
    GraphVersion,
    RunOutcome,
    RunStatus,
    WorkContract,
)
from northstack.domain.budget import Spend
from northstack.events.catalog import (
    PAYLOAD_BY_KIND,
    AnalysisRequested,
    CellCompleted,
    CellCreated,
    CellFailed,
    CellStarted,
    EventKind,
    EvidenceRecorded,
    GraphAccepted,
    RouteSelected,
    RunCreated,
    StatusChanged,
)
from northstack.events.envelope import EventEnvelope
from tests.helpers.events import env

# Helpers


def _tamper_db(db_path: Path, run_id: str, seq: int, column: str, value: object) -> None:
    """Directly tamper with a column in the events table (test helper, not production)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        f"UPDATE events SET {column} = ? WHERE run_id = ? AND seq = ?", (value, run_id, seq)
    )
    conn.commit()
    conn.close()


def _replace_payload_with_valid_hash(db_path: Path, run_id: str, seq: int, payload: object) -> None:
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT * FROM events WHERE run_id = ? AND seq = ?", (run_id, seq)
    ).fetchone()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = ledger_module._compute_hash(
        row["run_id"],
        row["seq"],
        row["kind"],
        payload,
        row["prev_hash"],
        row["schema_version"],
        row["timestamp"],
    )
    connection.execute(
        "UPDATE events SET payload = ?, hash_chain = ? WHERE run_id = ? AND seq = ?",
        (encoded, digest, run_id, seq),
    )
    connection.commit()
    connection.close()


def _contract() -> WorkContract:
    return WorkContract(
        id="wc-c1",
        objective="build",
        budget=Budget(token_limit=100, cost_limit_usd=1.0),
    )


def _artifact() -> ArtifactRef:
    return ArtifactRef(
        digest="sha256:" + "a" * 64,
        media_type="application/json",
        size_bytes=42,
    )


# A payload that legitimately carries free-form content. Redaction tests put
# secret-bearing dicts here rather than relaxing extra="forbid" on any model.
def _analysis(analysis: dict[str, object]) -> AnalysisRequested:
    return AnalysisRequested(profile="test", analysis=analysis)


# Fixtures


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    """A fresh ledger for each test, with automatic cleanup."""
    with Ledger(path=tmp_path / "test.db") as lg:
        yield lg


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    """A fresh artifact store for each test."""
    return ArtifactStore(base_path=tmp_path / "artifacts")


# Event ordering & hash chain


class TestEventOrdering:
    def test_append_assigns_monotonic_seq(self, ledger: Ledger):
        e1 = env(1, run_id="r1")
        e2 = env(2, StatusChanged(status=RunStatus.PLANNED), run_id="r1")
        a1 = ledger.append(e1)
        a2 = ledger.append(e2)
        assert a1.seq == 1
        assert a2.seq == 2

    def test_append_enforces_monotonic_seq(self, ledger: Ledger):
        e1 = env(1, run_id="r1")
        e2 = env(1, StatusChanged(status=RunStatus.PLANNED), run_id="r1")
        ledger.append(e1)
        with pytest.raises(ValueError, match="[Ss]equence"):
            ledger.append(e2)

    def test_append_stores_hash_chain(self, ledger: Ledger):
        e1 = env(1, _analysis({"x": 1}), run_id="r1")
        a1 = ledger.append(e1)
        events = ledger.events("r1")
        assert len(events) == 1
        assert events[0].hash_chain == a1.hash_chain
        assert len(events[0].hash_chain) == 64

    def test_events_ordered_by_seq(self, ledger: Ledger):
        for i in range(1, 6):
            e = env(i, run_id="r1")
            ledger.append(e)
        events = ledger.events("r1")
        seqs = [e.seq for e in events]
        assert seqs == [1, 2, 3, 4, 5]

    def test_events_since_filters_and_bounds_at_sql(self, ledger: Ledger):
        # 10 events; events_since must push seq>? + LIMIT to SQL.
        for i in range(1, 11):
            ledger.append(env(i, run_id="r1"))
        tail = ledger.events_since("r1", since=4, limit=3)
        assert [e.seq for e in tail] == [5, 6, 7]
        # since past the end -> empty
        assert ledger.events_since("r1", since=10, limit=5) == []
        # limit larger than remaining tail returns the rest, never seq<=since
        rest = ledger.events_since("r1", since=8, limit=50)
        assert [e.seq for e in rest] == [9, 10]
        # unknown run -> empty (no error)
        assert ledger.events_since("nope", since=0, limit=5) == []

    def test_append_returns_redacted_payload(self, ledger: Ledger):
        """The returned envelope must have redacted payload, not the original."""
        e = env(
            1,
            _analysis({"api_key": "sk-secret", "task": "build"}),
            run_id="r1",
        )
        returned = ledger.append(e)
        assert returned.payload.analysis["api_key"] == "[REDACTED]"
        assert returned.payload.analysis["task"] == "build"
        assert "sk-secret" not in returned.payload.analysis["api_key"]


# Hash tamper detection


class TestHashTamperDetection:
    def test_tampered_prev_hash_rejected(self, ledger: Ledger):
        e1 = env(1, run_id="r1")
        ledger.append(e1)
        e2 = env(
            2,
            StatusChanged(status=RunStatus.PLANNED),
            run_id="r1",
            prev_hash="bad_hash_value" + "0" * 50,
        )
        with pytest.raises(ValueError, match="[Hh]ash chain"):
            ledger.append(e2)

    def test_verify_integrity_passes_on_clean_chain(self, ledger: Ledger):
        for i in range(1, 4):
            ledger.append(env(i, run_id="r1"))
        changes = ledger._conn.total_changes
        result = ledger.verify_integrity("r1")
        assert result.ok is True
        assert result.events_checked == 3
        assert ledger._conn.total_changes == changes

    def test_verify_detects_prev_hash_tamper(self, ledger: Ledger, tmp_path: Path):
        """DB-level corruption of prev_hash is detected by verify_integrity."""
        for i in range(1, 4):
            ledger.append(env(i, run_id="r1"))
        _tamper_db(tmp_path / "test.db", "r1", 2, "prev_hash", "CORRUPTED")
        result = ledger.verify_integrity("r1")
        assert result.ok is False
        assert result.error_at_seq == 2

    def test_verify_detects_payload_tamper(self, ledger: Ledger, tmp_path: Path):
        """DB-level corruption of stored payload is detected by hash recomputation."""
        for i in range(1, 3):
            ledger.append(env(i, _analysis({"data": f"value-{i}"}), run_id="r1"))
        _tamper_db(tmp_path / "test.db", "r1", 2, "payload", '{"data":"TAMPERED"}')
        result = ledger.verify_integrity("r1")
        assert result.ok is False
        assert result.error_at_seq == 2

    def test_verify_detects_kind_tamper(self, ledger: Ledger, tmp_path: Path):
        """DB-level corruption of event kind is detected."""
        ledger.append(env(1, run_id="r1"))
        ledger.append(env(2, StatusChanged(status=RunStatus.PLANNED), run_id="r1"))
        _tamper_db(tmp_path / "test.db", "r1", 2, "kind", "claim_recorded")
        result = ledger.verify_integrity("r1")
        assert result.ok is False
        assert result.error_at_seq == 2

    def test_verify_detects_timestamp_tamper(self, ledger: Ledger, tmp_path: Path):
        """DB-level corruption of timestamp is detected."""
        ledger.append(env(1, run_id="r1"))
        ledger.append(env(2, StatusChanged(status=RunStatus.PLANNED), run_id="r1"))
        _tamper_db(tmp_path / "test.db", "r1", 2, "timestamp", "9999999999.0")
        result = ledger.verify_integrity("r1")
        assert result.ok is False
        assert result.error_at_seq == 2

    def test_verify_detects_hash_chain_tamper(self, ledger: Ledger, tmp_path: Path):
        """DB-level corruption of stored hash_chain is detected."""
        ledger.append(env(1, run_id="r1"))
        ledger.append(env(2, StatusChanged(status=RunStatus.PLANNED), run_id="r1"))
        _tamper_db(tmp_path / "test.db", "r1", 2, "hash_chain", "a" * 64)
        result = ledger.verify_integrity("r1")
        assert result.ok is False
        assert result.error_at_seq == 2

    @pytest.mark.parametrize(
        ("payload", "category"),
        [
            (b'\xff{"kind":"run_created"}', "payload_encoding"),
            ("{", "payload_json"),
            ("[]", "payload_type"),
        ],
    )
    def test_verify_reports_malformed_payload_without_raising(
        self, ledger: Ledger, tmp_path: Path, payload: object, category: str
    ) -> None:
        ledger.append(env(1, run_id="r1"))
        _tamper_db(tmp_path / "test.db", "r1", 1, "payload", payload)
        result = ledger.verify_integrity("r1")
        assert not result.ok
        assert result.error_at_seq == 1
        assert result.error_category == category

    def test_verify_rejects_hash_valid_schema_invalid_payload(
        self, ledger: Ledger, tmp_path: Path
    ) -> None:
        ledger.append(env(1, run_id="r1"))
        _replace_payload_with_valid_hash(
            tmp_path / "test.db",
            "r1",
            1,
            {"kind": "run_created", "schema_version": 1, "unexpected": True},
        )
        result = ledger.verify_integrity("r1")
        assert not result.ok
        assert result.error_at_seq == 1
        assert result.error_category == "event_schema"


# Payload redaction


class TestPayloadRedaction:
    def test_common_secret_keys_redacted(self, ledger: Ledger):
        e = env(
            1,
            _analysis(
                {
                    "api_key": "sk-secret-123",
                    "password": "hunter2",
                    "authorization": "Bearer token",
                    "normal_field": "visible",
                }
            ),
            run_id="r1",
        )
        ledger.append(e)
        events = ledger.events("r1")
        p = events[0].payload.analysis
        assert p["api_key"] == "[REDACTED]"
        assert p["password"] == "[REDACTED]"
        assert p["authorization"] == "[REDACTED]"
        assert p["normal_field"] == "visible"

    @pytest.mark.parametrize(
        "key",
        ["client_secret", "webhook-token", "x-api-key", "session_cookie", "signature", "sig"],
    )
    def test_sensitive_key_variants_redacted(self, ledger: Ledger, key: str):
        ledger.append(env(1, _analysis({key: "plaintext-secret"}), run_id="r1"))
        assert ledger.events("r1")[0].payload.analysis[key] == "[REDACTED]"

    def test_bearer_token_redacted(self, ledger: Ledger):
        e = env(1, _analysis({"header": "Bearer sk-abc123"}), run_id="r1")
        ledger.append(e)
        events = ledger.events("r1")
        assert events[0].payload.analysis["header"] == "[REDACTED]"

    def test_no_redaction_when_no_secrets(self, ledger: Ledger):
        e = env(1, _analysis({"task": "build feature X", "count": 42}), run_id="r1")
        ledger.append(e)
        events = ledger.events("r1")
        assert events[0].payload.analysis["task"] == "build feature X"
        assert events[0].payload.analysis["count"] == 42

    @pytest.mark.parametrize(
        "key", ["token_limit", "total_tokens", "key_status", "signature_algorithm"]
    )
    def test_nonsecret_metadata_names_are_not_redacted(self, ledger: Ledger, key: str):
        ledger.append(env(1, _analysis({key: 42}), run_id="r1"))
        assert ledger.events("r1")[0].payload.analysis[key] == 42

    def test_nested_dict_redaction(self, ledger: Ledger):
        """Secrets inside nested dicts are redacted recursively."""
        e = env(
            1,
            _analysis(
                {
                    "config": {
                        "api_key": "sk-nested-secret",
                        "name": "my-config",
                    },
                    "task": "build",
                }
            ),
            run_id="r1",
        )
        ledger.append(e)
        events = ledger.events("r1")
        p = events[0].payload.analysis
        assert p["config"]["api_key"] == "[REDACTED]"
        assert p["config"]["name"] == "my-config"
        assert p["task"] == "build"

    def test_nested_list_redaction(self, ledger: Ledger):
        """Secret strings inside lists are redacted recursively."""
        e = env(
            1,
            _analysis({"headers": ["Bearer token123", "normal-header"], "task": "build"}),
            run_id="r1",
        )
        ledger.append(e)
        events = ledger.events("r1")
        p = events[0].payload.analysis
        assert p["headers"][0] == "[REDACTED]"
        assert p["headers"][1] == "normal-header"

    def test_deeply_nested_redaction(self, ledger: Ledger):
        """Secrets at arbitrary nesting depth are redacted."""
        e = env(
            1,
            _analysis(
                {
                    "level1": {
                        "level2": {
                            "level3": {
                                "password": "deep-secret",
                                "data": "visible",
                            }
                        }
                    }
                }
            ),
            run_id="r1",
        )
        ledger.append(e)
        events = ledger.events("r1")
        p = events[0].payload.analysis
        assert p["level1"]["level2"]["level3"]["password"] == "[REDACTED]"
        assert p["level1"]["level2"]["level3"]["data"] == "visible"

    def test_tuple_values_redacted(self, ledger: Ledger):
        """Secret strings inside tuples are redacted."""
        e = env(
            1,
            _analysis({"mixed": ["normal", "Bearer sk-tuple-secret"]}),
            run_id="r1",
        )
        ledger.append(e)
        events = ledger.events("r1")
        p = events[0].payload.analysis
        assert p["mixed"][1] == "[REDACTED]"

    def test_openai_style_key_in_value_redacted(self, ledger: Ledger):
        """Strings matching secret patterns are redacted regardless of field name."""
        e = env(
            1,
            _analysis({"config": "use sk-proj-abc123 for auth"}),
            run_id="r1",
        )
        ledger.append(e)
        events = ledger.events("r1")
        assert events[0].payload.analysis["config"] == "[REDACTED]"


# Schema/event-version checks


class TestSchemaVersionChecks:
    def test_rejects_unknown_event_kind(self):
        with pytest.raises(ValidationError):
            EventEnvelope(run_id="r1", seq=1, payload={"kind": "unknown_kind"})

    def test_accepts_current_schema_version(self, ledger: Ledger):
        e = env(1, run_id="r1")
        a = ledger.append(e)
        assert a.schema_version == 1

    def test_rejects_future_schema_version(self, ledger: Ledger):
        e = env(1, RunCreated(schema_version=999), run_id="r1")
        with pytest.raises(ValueError, match="[Ss]chema"):
            ledger.append(e)


# Artifact store


class TestArtifactStore:
    def test_write_and_read(self, store: ArtifactStore):
        content = b'{"result": "all tests pass"}'
        ref = store.write(content, media_type="application/json")
        assert ref.size_bytes == len(content)
        assert ref.media_type == "application/json"
        assert ref.digest.startswith("sha256:")
        read_back = store.read(ref)
        assert read_back == content

    def test_verify_digest_pass(self, store: ArtifactStore):
        content = b"hello world"
        ref = store.write(content, media_type="text/plain")
        assert store.verify(ref) is True

    def test_verify_digest_fails_on_tamper(self, store: ArtifactStore):
        content = b"original content"
        ref = store.write(content, media_type="text/plain")
        # Tamper via direct path
        digest_hash = ref.digest.split(":", 1)[1]
        blob_path = store._path_for_digest(digest_hash)
        blob_path.write_bytes(b"tampered content")
        assert store.verify(ref) is False

    def test_content_addressed_dedup(self, store: ArtifactStore):
        content = b"same content twice"
        ref1 = store.write(content, media_type="text/plain")
        ref2 = store.write(content, media_type="text/plain")
        assert ref1.digest == ref2.digest

    def test_write_never_writes_directly_to_final_path(
        self, store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
    ):
        def fail(path: Path, content: bytes):
            raise AssertionError(f"direct final-path write: {path} ({len(content)} bytes)")

        monkeypatch.setattr(Path, "write_bytes", fail)
        ref = store.write(b"atomically published", media_type="text/plain")
        assert store.read(ref) == b"atomically published"

    def test_corrupt_existing_digest_blob_is_replaced(self, store: ArtifactStore):
        content = b"canonical content"
        digest = hashlib.sha256(content).hexdigest()
        path = store._path_for_digest(digest)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"corrupt")
        ref = store.write(content, media_type="text/plain")
        assert store.read(ref) == content
        assert store.verify(ref)

    def test_read_rejects_corrupt_blob(self, store: ArtifactStore):
        ref = store.write(b"original", media_type="text/plain")
        store._path_for_digest(ref.digest.split(":", 1)[1]).write_bytes(b"tampered")
        with pytest.raises(ValueError, match="digest"):
            store.read(ref)

    def test_concurrent_same_content_writers_converge(self, store: ArtifactStore):
        content = b"shared" * 10_000
        with ThreadPoolExecutor(max_workers=8) as pool:
            refs = list(
                pool.map(lambda _: store.write(content, media_type="text/plain"), range(32))
            )
        assert len({ref.digest for ref in refs}) == 1
        assert store.read(refs[0]) == content

    def test_concurrent_different_content_writers_remain_distinct(self, store: ArtifactStore):
        contents = [f"artifact-{index}".encode() * 1000 for index in range(32)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            refs = list(
                pool.map(lambda value: store.write(value, media_type="text/plain"), contents)
            )
        assert len({ref.digest for ref in refs}) == len(contents)
        assert [store.read(ref) for ref in refs] == contents

    def test_failed_atomic_publication_leaves_no_final_blob(
        self, store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
    ):
        content = b"never published"
        path = store._path_for_digest(hashlib.sha256(content).hexdigest())

        def fail(target: Path, value: bytes) -> None:
            raise OSError(f"injected failure for {target} ({len(value)} bytes)")

        monkeypatch.setattr(artifacts_module, "atomic_write_bytes", fail)
        with pytest.raises(OSError, match="injected"):
            store.write(content, media_type="text/plain")
        assert not path.exists()

    def test_read_enforces_store_byte_limit(self, tmp_path: Path):
        store = ArtifactStore(tmp_path / "bounded-artifacts", max_read_bytes=3)
        ref = store.write(b"four", media_type="text/plain")
        with pytest.raises(ValueError, match="read limit"):
            store.read(ref)

    def test_empty_content(self, store: ArtifactStore):
        ref = store.write(b"", media_type="application/octet-stream")
        assert ref.size_bytes == 0
        assert store.read(ref) == b""

    def test_large_content(self, store: ArtifactStore):
        content = b"x" * (1024 * 1024)
        ref = store.write(content, media_type="application/octet-stream")
        assert ref.size_bytes == len(content)
        assert store.verify(ref) is True


# Replay to RunState


class TestReplay:
    def test_replay_empty_run(self, ledger: Ledger):
        state = replay_run(ledger, "nonexistent")
        assert state.run_id == "nonexistent"
        assert state.status == RunStatus.INTAKE
        assert state.events_replayed == 0

    def test_replay_run_created(self, ledger: Ledger):
        ledger.append(env(1, run_id="r1"))
        state = replay_run(ledger, "r1")
        assert state.run_id == "r1"
        assert state.events_replayed == 1

    def test_replay_with_valid_status_changes(self, ledger: Ledger):
        """Valid state machine transitions replay cleanly."""
        events = [
            env(1, run_id="r1"),
            env(2, StatusChanged(status=RunStatus.CONTRACTED), run_id="r1"),
            env(3, StatusChanged(status=RunStatus.PLANNED), run_id="r1"),
            env(4, StatusChanged(status=RunStatus.EXECUTING), run_id="r1"),
        ]
        for e in events:
            ledger.append(e)
        state = replay_run(ledger, "r1")
        assert state.status == RunStatus.EXECUTING
        assert state.events_replayed == 4

    def test_replay_rejects_illegal_transition(self, ledger: Ledger):
        """STATUS_CHANGED events that violate the state machine raise ValueError."""
        events = [
            env(1, run_id="r1"),
            # Skip contracted/planned, jump straight to verifying (illegal)
            env(2, StatusChanged(status=RunStatus.VERIFYING), run_id="r1"),
        ]
        for e in events:
            ledger.append(e)
        with pytest.raises(ValueError, match="[Ii]llegal.*replay"):
            replay_run(ledger, "r1")

    def test_replay_rejects_terminal_transition(self, ledger: Ledger):
        """Transitioning out of a terminal state during replay raises ValueError."""
        events = [
            env(1, run_id="r1"),
            env(2, StatusChanged(status=RunStatus.CONTRACTED), run_id="r1"),
            env(3, StatusChanged(status=RunStatus.PLANNED), run_id="r1"),
            env(4, StatusChanged(status=RunStatus.EXECUTING), run_id="r1"),
            env(5, StatusChanged(status=RunStatus.VERIFYING), run_id="r1"),
            env(6, StatusChanged(status=RunStatus.VERIFIED), run_id="r1"),
            # Try to leave terminal state
            env(7, StatusChanged(status=RunStatus.EXECUTING), run_id="r1"),
        ]
        for e in events:
            ledger.append(e)
        with pytest.raises(ValueError, match="[Ii]llegal.*replay"):
            replay_run(ledger, "r1")

    def test_replay_cell_created(self, ledger: Ledger):
        events = [
            env(1, run_id="r1"),
            env(
                2,
                CellCreated(cell=GraphCell(id="c1", name="Feature A", contract=_contract())),
                run_id="r1",
            ),
        ]
        for e in events:
            ledger.append(e)
        state = replay_run(ledger, "r1")
        assert len(state.cells) == 1
        assert state.cells[0].id == "c1"

    def test_replay_ignores_other_runs(self, ledger: Ledger):
        ledger.append(env(1, run_id="r1"))
        ledger.append(env(1, run_id="r2"))
        state = replay_run(ledger, "r1")
        assert state.run_id == "r1"
        assert state.events_replayed == 1

    def test_replay_evidence_recorded_preserves_partial_state(self, ledger: Ledger):
        """A recorded evidence event sets the manifest without blocking later projection.

        The typed catalog refuses an invalid outcome at the boundary, so every
        EvidenceRecorded that reaches the ledger is valid; partial state after
        it is preserved (here: a subsequent RouteSelected still projects).
        """
        events = [
            env(1, run_id="r1"),
            env(2, EvidenceRecorded(outcome=RunOutcome.ABSTAINED), run_id="r1"),
            env(3, RouteSelected(cell_id="c1", profile_name="builder"), run_id="r1"),
        ]
        for event in events:
            ledger.append(event)

        state = replay_run(ledger, "r1")

        assert state.evidence_manifest is not None
        assert state.evidence_manifest.outcome == RunOutcome.ABSTAINED
        assert state.routes == {"c1": "builder"}
        assert state.events_replayed == 3

    def test_replay_projects_modern_graph_cell_lifecycle(self, ledger: Ledger):
        cell = GraphCell(id="c1", name="Build", contract=_contract())
        graph = GraphVersion(version=1, cells=[cell], current_horizon=0)
        events = [
            env(1, run_id="r1"),
            env(
                2,
                GraphAccepted(version=1, cells=graph.cells, edges=graph.edges),
                run_id="r1",
            ),
            env(3, CellStarted(cell_id="c1"), run_id="r1"),
        ]
        for event in events:
            ledger.append(event)
        assert replay_run(ledger, "r1").graph.cells[0].status == CellStatus.RUNNING

        ledger.append_next(
            "r1", CellCompleted(cell_id="c1", output_artifact=_artifact(), usage=Spend())
        )
        state = replay_run(ledger, "r1")

        assert state.graph is not None
        assert state.graph.cells[0].status == CellStatus.COMPLETED
        assert state.cells == []

    def test_replay_projects_modern_graph_cell_failure(self, ledger: Ledger):
        cell = GraphCell(id="c1", contract=_contract())
        ledger.append_next("r1", GraphAccepted(version=1, cells=[cell]))
        ledger.append_next("r1", CellFailed(cell_id="c1", error="boom", error_kind="provider"))

        state = replay_run(ledger, "r1")

        assert state.graph is not None
        assert state.graph.cells[0].status == CellStatus.FAILED


class TestProjectionExhaustiveness:
    def test_every_event_kind_has_a_payload_model(self):
        """The catalog's import-time guard: no EventKind without a payload model.

        This replaces the old RunProjection.handled_kinds assertion. With the
        pure ``fold`` + ``assert_never``, exhaustiveness is a type error; the
        catalog itself guarantees the kind/model bijection at import time.
        """
        assert set(PAYLOAD_BY_KIND) == set(EventKind)


# Concurrent append


class TestConcurrentAppend:
    def test_concurrent_append_next_same_run_is_contiguous(self, tmp_path: Path):
        with Ledger(path=tmp_path / "same-run.db") as ledger:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [
                    pool.submit(ledger.append_next, "shared", RunCreated()) for _ in range(100)
                ]
                for future in as_completed(futures):
                    future.result()

            events = ledger.events("shared")
            assert [event.seq for event in events] == list(range(1, 101))
            assert ledger.verify_integrity("shared").ok is True

    def test_concurrent_append_next_across_connections_is_contiguous(self, tmp_path: Path) -> None:
        db_path = tmp_path / "multi-connection.db"
        with Ledger(db_path) as first, Ledger(db_path) as second:
            start = Event()

            def append(index: int) -> EventEnvelope:
                assert start.wait(2)
                return (first, second)[index % 2].append_next("shared", RunCreated())

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(append, i) for i in range(100)]
                start.set()
                for future in as_completed(futures):
                    future.result()
            events = first.events("shared")
            assert [event.seq for event in events] == list(range(1, 101))
            assert first.verify_integrity("shared").ok

    def test_concurrent_appends_separate_runs(self, tmp_path: Path):
        """Multiple threads appending to separate runs produce no corruption.

        Each thread appends to its own run, exercising the shared lock
        without seq conflicts.
        """
        db_path = tmp_path / "concurrent.db"
        num_threads = 8
        events_per_thread = 20

        with Ledger(path=db_path) as lg:
            errors: list[Exception] = []

            def worker(thread_id: int) -> None:
                try:
                    run_id = f"run-{thread_id}"
                    for j in range(events_per_thread):
                        lg.append(env(j + 1, run_id=run_id))
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                futures = [pool.submit(worker, t) for t in range(num_threads)]
                for f in as_completed(futures):
                    f.result()

            assert errors == [], f"Concurrent append errors: {errors}"

            # Each run should have exactly events_per_thread events
            for t in range(num_threads):
                events = lg.events(f"run-{t}")
                assert len(events) == events_per_thread
                seqs = [e.seq for e in events]
                assert seqs == list(range(1, events_per_thread + 1))


# Context manager


class TestContextManager:
    def test_close_is_idempotent(self, tmp_path: Path):
        ledger = Ledger(path=tmp_path / "close.db")

        ledger.close()
        ledger.close()

    def test_context_manager(self, tmp_path: Path):
        with Ledger(path=tmp_path / "cm.db") as lg:
            lg.append(env(1, run_id="r1"))
            assert lg.events("r1")  # still open, can read
        # After exit, connection should be closed
        # Attempting operations would raise (but we just verify the context exits cleanly)

    def test_append_next_after_close_raises_runtime_error(self, tmp_path: Path):
        # A closed Ledger must reject append_next with a clean RuntimeError
        # ("Ledger is closed"), not a raw sqlite3.OperationalError.  Callers
        # (e.g. the web run task's finally) that race a close against a
        # concurrent append must get a typed, catchable error.
        ledger = Ledger(path=tmp_path / "closed.db")
        ledger.append_next("r1", RunCreated())
        ledger.close()
        with pytest.raises(RuntimeError, match="closed"):
            ledger.append_next("r1", RunCreated())

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("events", ("r1",)),
            ("events_since", ("r1",)),
            ("list_runs", ()),
            ("verify_integrity", ("r1",)),
        ],
    )
    def test_reads_after_close_raise_runtime_error(
        self, tmp_path: Path, method: str, args: tuple[object, ...]
    ) -> None:
        ledger = Ledger(path=tmp_path / f"{method}.db")
        ledger.close()
        with pytest.raises(RuntimeError, match="Ledger is closed"):
            getattr(ledger, method)(*args)

    def test_close_waits_for_active_reader(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ledger = Ledger(path=tmp_path / "reader-close.db")
        ledger.append_next("r1", RunCreated())
        entered, release = Event(), Event()
        original = Ledger._row_to_event

        def blocking_row_to_event(self: Ledger, row: sqlite3.Row) -> EventEnvelope:
            if self is ledger:
                entered.set()
                assert release.wait(2)
            return original(self, row)

        monkeypatch.setattr(Ledger, "_row_to_event", blocking_row_to_event)
        with ThreadPoolExecutor(max_workers=2) as pool:
            reader = pool.submit(ledger.events, "r1")
            assert entered.wait(2)
            closer = pool.submit(ledger.close)
            with pytest.raises(TimeoutError):
                closer.result(0.05)
            release.set()
            assert len(reader.result(2)) == 1
            closer.result(2)


class TestSQLiteLifecycle:
    def test_new_database_records_schema_version_and_wal(self, tmp_path: Path) -> None:
        db_path = tmp_path / "versioned.db"
        with Ledger(db_path):
            pass
        with sqlite3.connect(db_path) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    def test_future_database_schema_is_rejected_without_mutation(self, tmp_path: Path) -> None:
        db_path = tmp_path / "future.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA user_version = 2")
        before = db_path.read_bytes()
        with pytest.raises(ValueError, match="database schema version 2"):
            Ledger(db_path)
        assert db_path.read_bytes() == before
        assert not db_path.with_name(f"{db_path.name}-wal").exists()

    def test_reader_remains_available_during_other_connection_write(self, tmp_path: Path) -> None:
        db_path = tmp_path / "reader-writer.db"
        with Ledger(db_path) as writer, Ledger(db_path) as reader:
            writer.append_next("r1", RunCreated())
            writer._conn.execute("BEGIN IMMEDIATE")
            try:
                assert len(reader.events("r1")) == 1
            finally:
                writer._conn.execute("ROLLBACK")

    def test_busy_writer_waits_then_appends(self, tmp_path: Path) -> None:
        db_path = tmp_path / "busy.db"
        with Ledger(db_path) as holder, Ledger(db_path, busy_timeout=1) as contender:
            holder._conn.execute("BEGIN IMMEDIATE")
            entered = Event()

            def append() -> EventEnvelope:
                entered.set()
                return contender.append_next("r1", RunCreated())

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(append)
                assert entered.wait(2)
                with pytest.raises(TimeoutError):
                    future.result(0.05)
                holder._conn.execute("ROLLBACK")
                assert future.result(2).seq == 1

    def test_interrupted_insert_rolls_back_and_next_append_succeeds(self, tmp_path: Path) -> None:
        with Ledger(tmp_path / "rollback.db") as ledger:
            ledger._conn.execute(
                "CREATE TRIGGER fail_insert BEFORE INSERT ON events "
                "BEGIN SELECT RAISE(ABORT, 'injected'); END"
            )
            with pytest.raises(sqlite3.IntegrityError, match="injected"):
                ledger.append_next("r1", RunCreated())
            ledger._conn.execute("DROP TRIGGER fail_insert")
            assert ledger.append_next("r1", RunCreated()).seq == 1
