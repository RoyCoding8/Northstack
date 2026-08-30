"""Run index: the run-id -> workspace map the supervisor maintains.

``run_id -> workspace`` is populated when a run starts and retained after
release, so resolving a run is a dict lookup -- no candidate ``Ledger`` is
opened to probe. Historical runs (on disk before this process started) are
loaded once at startup by scanning the known workspace roots' ledgers.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

from northstack.adapters.sqlite_ledger import Ledger, UnsupportedDatabaseSchema
from northstack.events.errors import LedgerCorruption

logger = logging.getLogger(__name__)


class DuplicateRunIdError(ValueError):
    pass


_PRUNED_DIRS = {".git", ".venv", "node_modules", "__pycache__"}


def discover_workspaces(base_root: str, *, max_depth: int = 4) -> list[str]:
    """Workspace dirs under ``base_root`` holding a ledger (best-effort).

    Workspaces are arbitrary subdirectories of the files base root, so a
    plain listing of the root misses them; walk the tree shallowly instead
    and stop at run state directories once found.
    """
    base = Path(base_root)
    base_depth = len(base.parts)
    found: list[str] = []
    for dirpath, dirnames, _filenames in os.walk(base):
        if len(Path(dirpath).parts) - base_depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in _PRUNED_DIRS]
        if ".northstack" in dirnames and (Path(dirpath) / ".northstack" / "ledger.db").is_file():
            found.append(dirpath)
            dirnames.remove(".northstack")
    return found


class RunIndex:
    """run_id -> workspace path, the single source for ``_workspace_db``."""

    __slots__ = ("_ambiguous", "_runs", "_workspaces")

    def __init__(self) -> None:
        self._runs: dict[str, str] = {}
        self._workspaces: set[str] = set()
        self._ambiguous: set[str] = set()

    def register(self, run_id: str, workspace: str) -> None:
        root = str(Path(workspace).resolve())
        existing = self._runs.get(run_id)
        if run_id in self._ambiguous or (existing is not None and existing != root):
            raise DuplicateRunIdError(f"run id is ambiguous across workspaces: {run_id}")
        self._runs[run_id] = root
        self._workspaces.add(root)

    def forget(self, run_id: str) -> None:
        self._runs.pop(run_id, None)

    def workspace_of(self, run_id: str) -> str | None:
        return self._runs.get(run_id)

    def db_path(self, run_id: str) -> Path | None:
        ws = self._runs.get(run_id)
        if ws is None:
            return None
        return Path(ws) / ".northstack" / "ledger.db"

    def known_workspaces(self) -> list[str]:
        """Distinct workspace roots the index has seen (for history listing)."""
        return sorted(self._workspaces)

    def ambiguous_run_ids(self) -> list[str]:
        return sorted(self._ambiguous)

    def is_ambiguous(self, run_id: str) -> bool:
        return run_id in self._ambiguous

    def database_paths(self) -> list[Path]:
        return [Path(root) / ".northstack" / "ledger.db" for root in self.known_workspaces()]

    def load_historical(self, roots: list[str], *, page_size: int = 500) -> None:
        """One-time startup load: scan ``roots`` for ledgers and index their run ids.

        Each root is a workspace dir; its ledger is ``root/.northstack/ledger.db``.
        Unreadable ledgers are skipped (best-effort).
        """
        for root in roots:
            resolved = str(Path(root).resolve())
            db = Path(resolved) / ".northstack" / "ledger.db"
            if not db.is_file():
                continue
            try:
                ledger = Ledger(path=db)
                self._workspaces.add(resolved)
                try:
                    offset = 0
                    while page := ledger.list_runs(limit=page_size, offset=offset):
                        for summary in page:
                            existing = self._runs.get(summary.run_id)
                            if summary.run_id in self._ambiguous:
                                continue
                            if existing is not None and existing != resolved:
                                self._runs.pop(summary.run_id)
                                self._ambiguous.add(summary.run_id)
                                logger.warning(
                                    "run index: duplicate run id %s in %s and %s",
                                    summary.run_id,
                                    existing,
                                    resolved,
                                )
                            else:
                                self._runs[summary.run_id] = resolved
                        offset += len(page)
                finally:
                    ledger.close()
            except (sqlite3.Error, OSError, LedgerCorruption, UnsupportedDatabaseSchema):
                logger.debug("run index: skipping unreadable ledger %s", db, exc_info=True)
