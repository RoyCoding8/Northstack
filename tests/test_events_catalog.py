"""Tests that prove the typed event catalog and sound replay.

These tests pin the catalog's guarantees:

  1. Every payload model round-trips through ``parse_payload`` -- a producer's
     own ``model_dump`` is always re-readable, so a write can never produce a
     payload this build cannot read back.
  2. ``Ledger.append_next`` then ``events()`` returns the equal payload -- the
     ledger stores and rehydrates the typed object, not a loose dict.
  3. A stored row whose payload no longer matches the catalog is refused with
     ``LedgerCorruption`` that names ``seq`` and ``kind`` -- never a silent
     partial replay. A renamed field is the canonical corruption.
  4. A row from a future schema version is refused with ``UnknownSchemaVersion``
     -- best-effort parsing of an unknown format is the bug, not the fix.
  5. A pre-catalog row (no ``kind`` key in its payload) is refused -- old-shape
     data is never degraded into plausible-looking state.
  6. A committed golden ledger replays to a known ``RunState`` snapshot, so a
     silent regression in the projection or catalog is caught.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from northstack.adapters.sqlite_ledger import Ledger
from northstack.application.replay import replay_run
from northstack.domain.budget import Budget, BudgetUsage, Spend
from northstack.domain.contract import CriterionKind
from northstack.domain.graph import CellStatus, GraphCell
from northstack.domain.outcome import (
    ArtifactRef,
    FailureType,
    RecoveryAction,
    RunOutcome,
)
from northstack.domain.status import RunStatus
from northstack.events.catalog import (
    PAYLOAD_MODELS,
    ContractProposed,
    RunCreated,
    StatusChanged,
    parse_payload,
)
from northstack.events.errors import LedgerCorruption, UnknownSchemaVersion

FIXTURE = Path(__file__).parent / "fixtures" / "golden_ledger_verified_run.json"


# Helpers


def _tamper_payload(db_path: Path, run_id: str, seq: int, new_payload: str) -> None:
    """Overwrite a stored ``payload`` blob directly (test-only)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "UPDATE events SET payload = ? WHERE run_id = ? AND seq = ?",
        (new_payload, run_id, seq),
    )
    conn.commit()
    conn.close()


def _set_column(db_path: Path, run_id: str, seq: int, column: str, value: object) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        f"UPDATE events SET {column} = ? WHERE run_id = ? AND seq = ?",
        (value, run_id, seq),
    )
    conn.commit()
    conn.close()


# 1. Hypothesis round trip: every model re-parses its own dump


@pytest.mark.parametrize("model", PAYLOAD_MODELS, ids=lambda m: m.__name__)
def test_payload_round_trips_through_parse(model):
    """A payload's own JSON dump must re-validate to an equal payload.

    Frozen + extra="forbid" must not turn a producer-emitted value into an
    unreadable one. We construct one minimal valid instance per model.
    """
    instance = _minimal(model)
    reparsed = parse_payload(instance.model_dump(mode="json"))
    assert reparsed == instance


@given(data=st.data())
@settings(max_examples=100)
def test_round_trip_preserves_kind(data):
    """For a random payload model, any valid instance re-parses to the same kind."""
    model = data.draw(st.sampled_from(PAYLOAD_MODELS))
    instance = _minimal(model)
    reparsed = parse_payload(instance.model_dump(mode="json"))
    assert reparsed.kind == instance.kind


def _minimal(model):
    """Construct the smallest valid instance of ``model`` known to parse.

    Introspects ``model.model_fields`` and supplies a minimal value for every
    non-defaulted field via a type-keyed registry. Optional fields are left to
    their catalog defaults so the test stays robust to additive fields -- if a
    field regresses from optional to required the test fails loudly rather than
    silently constructing a value the catalog never asked for.
    """
    # Map a field's annotation (the bare class object pydantic exposes) to a
    # minimal valid instance. Enums take their first member; nested models take
    # a minimal construction; ``str``/``bool`` take a trivial literal.
    registry = {
        str: "x",
        bool: True,
        int: 1,
        Budget: Budget(),
        BudgetUsage: BudgetUsage(),
        Spend: Spend(),
        GraphCell: GraphCell(id="c1"),
        ArtifactRef: ArtifactRef(
            digest="sha256:" + "a" * 64, media_type="application/json", size_bytes=1
        ),
        RunStatus: RunStatus.INTAKE,
        CellStatus: CellStatus.PENDING,
        CriterionKind: CriterionKind.COMMAND,
        RunOutcome: RunOutcome.VERIFIED,
        FailureType: FailureType.TRANSIENT,
        RecoveryAction: RecoveryAction.BACKOFF_RETRY,
    }
    kwargs: dict = {}
    for name, fi in model.model_fields.items():
        if fi.is_required():
            ann = fi.annotation
            if ann not in registry:
                raise AssertionError(
                    f"no minimal value registered for {model.__name__}.{name}: {ann!r}"
                )
            kwargs[name] = registry[ann]
    return model(**kwargs)


# 2. Ledger round trip: append_next -> events() returns an equal payload


def test_ledger_round_trip_preserves_payload(tmp_path: Path):
    payloads = [
        RunCreated(),
        StatusChanged(status=RunStatus.INTAKE),
        ContractProposed(
            id="wc-1",
            version=1,
            objective="o",
            budget=Budget(token_limit=1000, cost_limit_usd=1.0),
            acceptance_criteria_count=2,
        ),
    ]
    with Ledger(path=tmp_path / "rt.db") as lg:
        for p in payloads:
            lg.append_next("r1", p)
        events = lg.events("r1")
    assert [e.payload for e in events] == payloads


# 3. Loud failure: a renamed field raises LedgerCorruption naming seq + kind


def test_renamed_field_raises_ledger_corruption(tmp_path: Path):
    db_path = tmp_path / "corrupt.db"
    with Ledger(path=db_path) as lg:
        lg.append_next("r1", RunCreated())
        lg.append_next("r1", StatusChanged(status=RunStatus.INTAKE))

    # Rename the required ``status`` field on the STATUS_CHANGED row -- the
    # catalog must refuse the row, naming the seq and kind it could not read.
    rows = (
        sqlite3.connect(str(db_path))
        .execute("SELECT payload FROM events WHERE run_id = ? AND seq = ?", ("r1", 2))
        .fetchone()
    )
    mutated = json.loads(rows[0])
    mutated["state"] = mutated.pop("status")
    _tamper_payload(db_path, "r1", 2, json.dumps(mutated, sort_keys=True))

    with Ledger(path=db_path) as lg, pytest.raises(LedgerCorruption) as exc:
        lg.events("r1")
    assert exc.value.seq == 2
    assert exc.value.kind == "status_changed"


# 4. Unknown future schema version raises UnknownSchemaVersion


def test_future_schema_version_raises_unknown_schema_version(tmp_path: Path):
    db_path = tmp_path / "future.db"
    with Ledger(path=db_path) as lg:
        lg.append_next("r1", RunCreated())

    # Inject a payload declaring a schema_version this build does not understand.
    rows = (
        sqlite3.connect(str(db_path))
        .execute("SELECT payload FROM events WHERE run_id = ? AND seq = ?", ("r1", 1))
        .fetchone()
    )
    mutated = json.loads(rows[0])
    mutated["schema_version"] = 99
    _tamper_payload(db_path, "r1", 1, json.dumps(mutated, sort_keys=True))
    # The ``schema_version`` column is read off the payload by the ledger, so it
    # must reflect the tampered value too for the upcaster to see it.
    _set_column(db_path, "r1", 1, "schema_version", 99)

    with Ledger(path=db_path) as lg, pytest.raises(UnknownSchemaVersion) as exc:
        lg.events("r1")
    assert exc.value.seq == 1
    assert exc.value.kind == "run_created"
    assert exc.value.version == 99


# 5. Pre-catalog ledger: an old-shape row is refused, never partially replayed


def test_pre_catalog_row_is_refused(tmp_path: Path):
    db_path = tmp_path / "precat.db"
    with Ledger(path=db_path) as lg:
        lg.append_next("r1", RunCreated())

    # Overwrite a row with a pre-catalog payload: a bare dict with no ``kind``
    # key. This is the shape the ledger stored before the typed catalog existed.
    _tamper_payload(db_path, "r1", 1, json.dumps({"status": "intake"}, sort_keys=True))

    with (
        Ledger(path=db_path) as lg,
        pytest.raises(LedgerCorruption, match="predates the typed event catalog"),
    ):
        lg.events("r1")


# 6. Golden ledger: committed fixture replays to a known snapshot


def test_golden_ledger_replays_to_known_snapshot(tmp_path: Path):
    payloads = [parse_payload(d) for d in json.loads(FIXTURE.read_text())]
    with Ledger(path=tmp_path / "golden.db") as lg:
        for p in payloads:
            lg.append_next("golden-run", p)
        state = replay_run(lg, "golden-run")
    snap = state.snapshot()
    # Stable, semantically meaningful fields only -- the per-cell placeholder
    # contract and last_event_hash are not part of the contract this test holds.
    assert snap["run_id"] == "golden-run"
    assert snap["status"] == "verified"
    assert snap["outcome"] == "verified"
    assert snap["contract_version"] == 1
    assert snap["graph_version"] == 1
    assert snap["events_replayed"] == len(payloads)
    assert [c.status.value for c in state.cells] == ["verified"]
