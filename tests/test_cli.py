"""Tests for CLI at the public seam.

Seams tested:
  1. config validate -- loads TOML and reports valid/invalid
  2. run -- config error handling (missing/invalid config produces clean error)
  3. ledger replay -- replays events to reconstruct RunState
  4. ledger verify -- checks hash chain integrity
  5. ledger events -- lists events for a run
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

from typer.testing import CliRunner

from northstack.adapters.sqlite_ledger import Ledger
from northstack.interfaces.cli import app
from tests.helpers.events import env

runner = CliRunner()


# Helpers


def _write_valid_config(tmp_path: Path) -> Path:
    toml_content = textwrap.dedent("""\
        [northstack]
        name = "TestCo"

        [[northstack.profiles]]
        name = "cheap-worker"
        protocol = "openai_chat"
        base_url = "http://localhost:8080/v1"
        model = "mimo-v2.5"
        max_concurrency = 8
        api_key_env = "MIMO_API_KEY"
        roles = ["worker"]
        capabilities = ["tool_use"]
    """)
    config_path = tmp_path / "northstack.toml"
    config_path.write_text(toml_content)
    return config_path


def _write_invalid_config(tmp_path: Path) -> Path:
    toml_content = textwrap.dedent("""\
        [northstack]
        name = "BadCo"

        [[northstack.profiles]]
        name = "bad"
        protocol = "bogus"
        base_url = "http://x"
        model = "m"
        max_concurrency = 1
    """)
    config_path = tmp_path / "bad.toml"
    config_path.write_text(toml_content)
    return config_path


def _seed_ledger(ledger_path: Path, run_id: str = "run-1") -> None:
    """Seed a ledger with a few events for testing."""
    with Ledger(path=ledger_path) as lg:
        for i in range(1, 4):
            lg.append(env(i, run_id=run_id))


def _tamper_db(db_path: Path, run_id: str, seq: int, column: str, value: str) -> None:
    """Directly tamper with a column in the events table."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        f"UPDATE events SET {column} = ? WHERE run_id = ? AND seq = ?", (value, run_id, seq)
    )
    conn.commit()
    conn.close()


# config validate


class TestConfigValidate:
    def test_valid_config(self, tmp_path: Path):
        config_path = _write_valid_config(tmp_path)
        result = runner.invoke(app, ["config", "validate", str(config_path)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower() or "ok" in result.output.lower()

    def test_validate_loads_env_file_beside_config(self, tmp_path: Path, monkeypatch):
        config_path = _write_valid_config(tmp_path)
        (tmp_path / ".env").write_text("MIMO_API_KEY=test-key-123\n")
        monkeypatch.delenv("MIMO_API_KEY", raising=False)
        result = runner.invoke(app, ["config", "validate", str(config_path)])
        assert result.exit_code == 0
        assert "env:MIMO_API_KEY OK" in result.output
        assert os.environ["MIMO_API_KEY"] == "test-key-123"

    def test_invalid_config(self, tmp_path: Path):
        config_path = _write_invalid_config(tmp_path)
        result = runner.invoke(app, ["config", "validate", str(config_path)])
        assert result.exit_code != 0
        assert "error" in result.output.lower() or "invalid" in result.output.lower()

    def test_missing_config_file(self, tmp_path: Path):
        result = runner.invoke(app, ["config", "validate", str(tmp_path / "nope.toml")])
        assert result.exit_code != 0


# run: config error handling


class TestRunProjectConfigError:
    """C-1: run_project must handle missing/invalid config cleanly, not traceback.

    Uses subprocess to invoke the real CLI (Typer's test runner swallows
    exceptions, hiding the traceback that real operators see).
    """

    def test_run_missing_config_file(self, tmp_path: Path):
        """A missing --config file should produce a clean error, not a traceback."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "northstack.interfaces.cli",
                "run",
                "--config",
                str(tmp_path / "nonexistent.toml"),
                "--workspace",
                str(tmp_path),
                "--goal",
                "test",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert result.returncode != 0, "expected non-zero exit on missing config"
        combined = result.stdout + result.stderr
        assert "Traceback" not in combined, f"raw traceback leaked to user output:\n{combined}"

    def test_run_invalid_config(self, tmp_path: Path):
        """A malformed/invalid config should produce a clean error, not a traceback."""
        config_path = _write_invalid_config(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "northstack.interfaces.cli",
                "run",
                "--config",
                str(config_path),
                "--workspace",
                str(tmp_path),
                "--goal",
                "test",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert result.returncode != 0, "expected non-zero exit on invalid config"
        combined = result.stdout + result.stderr
        assert "Traceback" not in combined, f"raw traceback leaked to user output:\n{combined}"


# ledger replay


class TestLedgerReplay:
    def test_replay_shows_state(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        _seed_ledger(db_path)
        result = runner.invoke(app, ["ledger", "replay", str(db_path), "run-1"])
        assert result.exit_code == 0
        assert "run-1" in result.output

    def test_replay_unknown_run(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        _seed_ledger(db_path)
        result = runner.invoke(app, ["ledger", "replay", str(db_path), "unknown"])
        assert result.exit_code == 0
        assert "unknown" in result.output


# ledger verify


class TestLedgerVerify:
    def test_verify_clean_chain(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        _seed_ledger(db_path)
        result = runner.invoke(app, ["ledger", "verify", str(db_path), "run-1"])
        assert result.exit_code == 0
        assert "ok" in result.output.lower() or "integrity" in result.output.lower()

    def test_verify_tampered_chain(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        _seed_ledger(db_path)
        _tamper_db(db_path, "run-1", 2, "prev_hash", "CORRUPTED")
        result = runner.invoke(app, ["ledger", "verify", str(db_path), "run-1"])
        assert result.exit_code != 0
        assert "fail" in result.output.lower() or "error" in result.output.lower()


# ledger events


class TestLedgerEvents:
    def test_list_events(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        _seed_ledger(db_path)
        result = runner.invoke(app, ["ledger", "events", str(db_path), "run-1"])
        assert result.exit_code == 0
        assert "3" in result.output or "run-1" in result.output

    def test_list_events_empty_run(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        _seed_ledger(db_path)
        result = runner.invoke(app, ["ledger", "events", str(db_path), "empty"])
        assert result.exit_code == 0


# The sync entrypoint refuses to run inside a live event loop


class TestSyncEntrypointRefusesLiveLoop:
    """``Company.run`` is an explicit sync entrypoint that *refuses* to run
    when an event loop is already running, surfacing a clear error rather than
    smuggling a second loop onto a worker thread (which would be a
    sync-over-async hack with no clear failure mode).
    """

    def test_run_inside_a_live_loop_raises_clear_error(self, tmp_path: Path) -> None:
        import asyncio
        from typing import Any

        from northstack.adapters.artifacts import ArtifactStore
        from northstack.adapters.sqlite_ledger import Ledger
        from northstack.adapters.workspace.restricted import RestrictedWorkspace
        from northstack.application.contracting import (
            ContractCompiler,
            DeterministicAnalysisRunner,
        )
        from northstack.application.orchestrator import Company
        from northstack.application.worker import WorkerResult
        from northstack.config import NorthStackConfig
        from northstack.domain.request import ProjectRequest

        class _NoopWorker:
            async def run(
                self,
                cell: Any,
                profile_name: str,
                tool_defs: list[Any],
                *,
                system_prompt: str = "",
                output_json_schema: dict[str, Any] | None = None,
                resume_from_messages: list[Any] | None = None,
                on_progress: Any = None,
                on_checkpoint: Any = None,
                on_event: Any = None,
            ) -> WorkerResult:
                return WorkerResult(ok=True, text="{}", total_input_tokens=0, total_output_tokens=0)

        class _NoopFactory:
            def create(self, workspace: Any) -> _NoopWorker:
                return _NoopWorker()

        config = NorthStackConfig(name="test")
        ledger = Ledger(path=tmp_path / "test.db")
        store = ArtifactStore(tmp_path / "artifacts")
        ws = RestrictedWorkspace(tmp_path / "workspace")
        company = Company(
            config=config,
            ledger=ledger,
            artifact_store=store,
            workspace=ws,
            gateway=None,
            worker_factory=_NoopFactory(),
            compiler=ContractCompiler(analysis_runner=DeterministicAnalysisRunner()),
        )
        request = ProjectRequest(goal="test", workspace_root=str(tmp_path))
        captured: list[BaseException] = []

        async def _call_sync_from_loop() -> None:
            # Inside a running loop, the sync entrypoint must refuse -- not
            # spawn a ThreadPoolExecutor + nested asyncio.run.
            try:
                company.run(request)
            except BaseException as exc:  # noqa: BLE001 - capture for assertion
                captured.append(exc)

        asyncio.run(_call_sync_from_loop())
        ledger.close()

        assert captured, "Company.run did not raise when called inside a live loop"
        err = captured[0]
        assert isinstance(err, RuntimeError), (
            f"expected RuntimeError, got {type(err).__name__}: {err}"
        )
        assert "loop" in str(err).lower() or "async" in str(err).lower(), (
            f"error message does not name the live-loop cause: {err}"
        )
