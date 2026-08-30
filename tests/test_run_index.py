"""Run index: the supervisor-maintained map that kills the fs scan.

Without a run index, ``_workspace_db`` would open every candidate
``.northstack/ledger.db`` and call ``ledger.events(run_id)`` on each to ask
"do you contain this run?".  The run index the supervisor maintains
(``run_id -> workspace``, populated at run start, retained after release)
removes that probe.  This file pins that the resolution of a known run's db
path NEVER opens a Ledger to probe -- the index answers directly.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


def _deterministic_config(name: str = "TestCo"):
    from northstack.config import ModelProfile, NorthStackConfig, Protocol, Role, RunConfig

    reviewer = ModelProfile(
        name="reviewer-1",
        roles={Role.REVIEWER},
        protocol=Protocol.OPENAI_CHAT,
        base_url="http://localhost:8080/v1",
        model="test",
        max_concurrency=4,
        requests_per_minute=1000,
        input_price_per_million_usd=1.0,
        output_price_per_million_usd=5.0,
        max_output_tokens=4096,
    )
    planner = ModelProfile(
        name="planner-1",
        roles={Role.PLANNER},
        protocol=Protocol.OPENAI_CHAT,
        base_url="http://localhost:8080/v1",
        model="test",
        max_concurrency=4,
        requests_per_minute=1000,
        input_price_per_million_usd=1.0,
        output_price_per_million_usd=5.0,
        max_output_tokens=4096,
    )
    return NorthStackConfig(
        name=name, profiles=[reviewer, planner], commands=[], run=RunConfig(), routing=[]
    )


def _count_ledger_opens(fn):
    """Count Ledger constructions during ``fn()`` by patching sqlite3.connect.

    The fs-scan probe opened a Ledger per candidate. We count sqlite
    connections opened by Ledger.__init__'s sqlite3.connect -- if the run
    index resolves the run directly, no probe Ledger is opened -> zero.
    """
    import sqlite3

    count = {"n": 0}
    real_connect = sqlite3.connect

    def _spy_connect(*args, **kwargs):
        count["n"] += 1
        return real_connect(*args, **kwargs)

    with patch("northstack.adapters.sqlite_ledger.sqlite3.connect", side_effect=_spy_connect):
        result = fn()
    return result, count["n"]


class TestRunIndexNoFilesystemProbe:
    def test_resolving_a_finished_run_db_does_not_open_ledgers_to_probe(
        self, tmp_path: Path
    ) -> None:
        from northstack.adapters.config_toml import save_config_to_toml
        from northstack.interfaces.web.server import create_app

        config_path = tmp_path / "northstack.toml"
        save_config_to_toml(_deterministic_config(), config_path)
        app = create_app(config_path)
        app.state.files_base_root = str(tmp_path)
        ws = tmp_path / "ws"
        ws.mkdir()

        with TestClient(app) as client:
            # Start a run so its id enters the run index, then wait for it
            # to finish so the ledger is flushed and the supervisor released.
            r = client.post("/api/runs", json={"goal": "g", "workspace_root": str(ws)})
            assert r.status_code == 200, r.text
            run_id = r.json()["run_id"]

            deadline = time.time() + 20.0
            snap = None
            while time.time() < deadline:
                snap = client.get(f"/api/runs/{run_id}").json()
                if snap.get("status") in ("verified", "abstained", "failed"):
                    break
                time.sleep(0.05)
            assert snap is not None and snap.get("status") in (
                "verified",
                "abstained",
                "failed",
            )

            # The run is finished. A SECOND, separate server process started
            # against the same workspace would have NO live supervisor for
            # this run -- its only way to find the db is the run index. We
            # simulate that by dropping the supervisor (as a fresh process
            # would have no in-memory handles) while keeping the index.
            app.state.supervisors.pop(run_id, None)

            from starlette.requests import Request

            from northstack.interfaces.web.routes_runs import _workspace_db

            scope = {
                "type": "http",
                "app": app,
                "headers": [],
                "method": "GET",
                "path": "/",
                "query_string": b"",
            }
            request = Request(scope)

            # Resolving the finished run's db must NOT open candidate Ledgers
            # to probe -- the run index answers directly. We count sqlite
            # connections opened while resolving; if the index answers, zero.
            def _resolve() -> Path | None:
                return _workspace_db(request, run_id)

            db, opened = _count_ledger_opens(_resolve)

            assert db is not None, "index must resolve the finished run's db"
            assert db.name == "ledger.db"
            assert opened == 0, (
                f"resolving a finished run must not open Ledgers to probe; opened {opened}"
            )


def test_historical_load_crosses_every_page(tmp_path: Path) -> None:
    from northstack.adapters.sqlite_ledger import Ledger
    from northstack.application.run_index import RunIndex
    from northstack.events.catalog import RunCreated

    workspace = tmp_path / "history"
    db = workspace / ".northstack" / "ledger.db"
    with Ledger(path=db) as ledger:
        for run_id in ("run-a", "run-b", "run-c"):
            ledger.append_next(run_id, RunCreated())
    index = RunIndex()
    index.load_historical([str(workspace)], page_size=1)
    assert {index.workspace_of(run_id) for run_id in ("run-a", "run-b", "run-c")} == {
        str(workspace.resolve())
    }
    assert index.database_paths() == [db]


def test_historical_load_skips_future_database_schema_without_mutation(tmp_path: Path) -> None:
    import sqlite3

    from northstack.application.run_index import RunIndex

    workspace = tmp_path / "future"
    db = workspace / ".northstack" / "ledger.db"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA user_version = 2")
    before = db.read_bytes()
    index = RunIndex()
    index.load_historical([str(workspace)])
    assert index.known_workspaces() == []
    assert db.read_bytes() == before


def test_historical_load_marks_duplicate_run_ids_ambiguous(tmp_path: Path) -> None:
    from northstack.adapters.sqlite_ledger import Ledger
    from northstack.application.run_index import RunIndex
    from northstack.events.catalog import RunCreated

    workspaces = [tmp_path / "first", tmp_path / "second"]
    for workspace in workspaces:
        with Ledger(workspace / ".northstack" / "ledger.db") as ledger:
            ledger.append_next("run-duplicate", RunCreated())
    index = RunIndex()
    index.load_historical([str(workspace) for workspace in workspaces])
    assert index.workspace_of("run-duplicate") is None
    assert index.is_ambiguous("run-duplicate")
    assert index.ambiguous_run_ids() == ["run-duplicate"]
    assert index.database_paths() == [
        workspace / ".northstack" / "ledger.db" for workspace in workspaces
    ]
