"""Run lifecycle + history + comparison endpoints for the web control surface.

Run start (CRITICAL lifecycle, do not regress):
  - ``POST /api/runs`` builds a FRESH ``Company`` per run via ``build_company``
    (fresh ``ModelGateway`` whose httpx client is bound to the server's event
    loop).  It pre-generates ``run_id`` so the handler can return it
    synchronously, then wraps ``company.run_async(request, run_id=run_id)``
    + ``ledger.close()`` in an inner ``async def _run()`` and
    ``asyncio.create_task(_run())`` it on the server's own loop.  Never
    ``asyncio.run``.  ``run_async`` closes the gateway itself in its own
    finally (same loop) -- the handler must NOT close the gateway.

Each run is owned by a :class:`RunSupervisor` held in
``app.state.supervisors``: the supervisor holds the per-run task, ledger
handle and workspace path, and releases them exactly once. Request handlers
open their own ledger handles, so supervisor release never races with a web
read. ``release()`` closes the run-owned ledger when the run finishes.
"""

from __future__ import annotations

import asyncio
import csv
import heapq
import io
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from northstack.adapters.sqlite_ledger import Ledger, RunSummary
from northstack.application.build import build_company
from northstack.application.projection_cache import ProjectionCache
from northstack.application.replay import replay_run
from northstack.application.run_index import RunIndex
from northstack.application.run_supervisor import RunSupervisor
from northstack.domain import Budget, ProjectRequest
from northstack.domain.request import GoalText, MAX_WAVES, WorkspaceRootText
from northstack.events.envelope import EventEnvelope
from northstack.interfaces.web.config_store import ConfigStore
from northstack.interfaces.web.routes_files import _base_root, _is_within

router = APIRouter(tags=["runs"])
logger = logging.getLogger(__name__)


def _store(request: Request) -> ConfigStore:
    store: ConfigStore = request.app.state.store
    return store


def _supervisors(request: Request) -> dict[str, RunSupervisor]:
    supervisors: dict[str, RunSupervisor] = request.app.state.supervisors
    return supervisors


def _run_index(request: Request) -> RunIndex:
    run_index: RunIndex = request.app.state.run_index
    return run_index


class RunStartBody(BaseModel):
    goal: GoalText
    workspace_root: WorkspaceRootText
    budget_tokens: int | None = Field(default=None, ge=0)
    budget_cost_usd: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    max_waves: int = Field(default=3, ge=1, le=MAX_WAVES)


def _budget(body: RunStartBody) -> Budget | None:
    if body.budget_tokens is None and body.budget_cost_usd is None:
        return None
    try:
        return Budget(
            token_limit=body.budget_tokens,
            cost_limit_usd=body.budget_cost_usd,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


def _workspace_db(request: Request, run_id: str | None) -> Path | None:
    """Resolve the ledger.db path for a run (or for history listing).

    The run index (``app.state.run_index``) maps run_id -> workspace and is
    the single source: a live run's supervisor registers its workspace at
    start, historical runs are loaded once at lifespan start, so this is a
    dict lookup -- no candidate Ledger is opened to probe.  Returns None if
    unknown.
    """
    index = _run_index(request)
    if run_id is not None:
        if index.is_ambiguous(run_id):
            raise HTTPException(
                status_code=409,
                detail={
                    "message": f"run id is ambiguous across workspaces: {run_id}",
                    "category": "ambiguous_run_id",
                },
            )
        return index.db_path(run_id)
    for ws in index.known_workspaces():
        db = Path(ws) / ".northstack" / "ledger.db"
        if db.exists():
            return db
    return None


def _resolve_ledger(request: Request, run_id: str) -> Ledger:
    """Open a request-owned Ledger for ``run_id``."""
    db = _workspace_db(request, run_id)
    if db is None:
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
    return Ledger(path=db)


def _is_active(request: Request, run_id: str) -> bool:
    sup = _supervisors(request).get(run_id)
    return sup is not None and sup.is_active


def _db_for(request: Request) -> Path:
    """A ledger.db to list run history from (any known workspace)."""
    db = _workspace_db(request, None)
    return db or Path(".northstack") / "ledger.db"


def _history_runs(
    request: Request,
    *,
    limit: int,
    status: str | None = None,
    outcome: str | None = None,
) -> tuple[list[RunSummary], int]:
    paths = _run_index(request).database_paths()
    fallback = Path(".northstack") / "ledger.db"
    if not paths and fallback.is_file():
        paths = [fallback]

    total = 0

    def summaries():
        nonlocal total
        for db in paths:
            if not db.is_file():
                continue
            ledger = Ledger(path=db)
            try:
                offset = 0
                while page := ledger.list_runs(limit=500, offset=offset):
                    for run in page:
                        if (not status or run.status == status) and (
                            not outcome or (run.outcome or "") == outcome
                        ):
                            total += 1
                            yield run
                    offset += len(page)
            finally:
                ledger.close()

    runs = heapq.nsmallest(limit, summaries(), key=lambda run: (-run.last_event_time, run.run_id))
    return runs, total


def _snapshot_for_run(request: Request, run_id: str) -> dict[str, Any]:
    ledger = _resolve_ledger(request, run_id)
    try:
        if not ledger.events(run_id):
            raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
        return replay_run(ledger, run_id).snapshot()
    finally:
        ledger.close()


@router.post("/runs")
async def start_run(body: RunStartBody, request: Request) -> dict[str, Any]:
    workspace = await asyncio.to_thread(lambda: Path(body.workspace_root).expanduser().resolve())
    if not _is_within(workspace, _base_root(request)):
        raise HTTPException(
            status_code=403,
            detail=f"workspace '{workspace}' is outside the allowed root",
        )
    if not workspace.is_dir():
        raise HTTPException(status_code=400, detail=f"workspace not a directory: {workspace}")

    config = _store(request).get()
    db_path = workspace / ".northstack" / "ledger.db"
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    request_obj = ProjectRequest(
        goal=body.goal,
        workspace_root=str(workspace),
        budget=_budget(body),
        max_waves=body.max_waves,
    )
    components = build_company(config, workspace, db_path=db_path)
    sup = RunSupervisor(
        run_id=run_id,
        ledger=components.ledger,
        workspace=str(workspace),
        task=None,
    )

    async def _run() -> None:
        try:
            await components.company.run_async(request_obj, run_id=run_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("run crashed run_id=%s", run_id)
        finally:
            sup.release()

    task: asyncio.Task[None] | None = None
    registered = False
    try:
        _supervisors(request)[run_id] = sup
        _run_index(request).register(run_id, str(workspace))
        registered = True
        task = asyncio.create_task(_run(), name=run_id)
        sup.bind_task(task)
    except BaseException:
        _supervisors(request).pop(run_id, None)
        if registered:
            _run_index(request).forget(run_id)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.debug("startup task cancelled during rollback")
        sup.release()
        try:
            await components.gateway.close()
        except Exception:
            logger.warning("gateway.close() failed during startup rollback", exc_info=True)
        raise
    return {"run_id": run_id, "workspace_root": str(workspace)}


@router.post("/runs/{run_id}/resume")
async def resume_run(run_id: str, request: Request) -> dict[str, Any]:
    """Continue a stopped/failed run: replay its contract/graph, seed
    completed cells, and execute the rest as a fresh run id."""
    if _is_active(request, run_id):
        raise HTTPException(status_code=409, detail="run is active; stop it before resuming")
    db_path = _workspace_db(request, run_id)
    if db_path is None or not db_path.is_file():
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
    workspace = db_path.parent.parent
    config = _store(request).get()
    components = build_company(config, workspace, db_path=db_path)
    new_run_id = f"run-{uuid.uuid4().hex[:12]}"
    sup = RunSupervisor(
        run_id=new_run_id,
        ledger=components.ledger,
        workspace=str(workspace),
        task=None,
    )

    async def _run() -> None:
        try:
            await components.company.resume_async(run_id, run_id=new_run_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("resumed run crashed run_id=%s", new_run_id)
        finally:
            sup.release()

    task: asyncio.Task[None] | None = None
    registered = False
    try:
        _supervisors(request)[new_run_id] = sup
        _run_index(request).register(new_run_id, str(workspace))
        registered = True
        task = asyncio.create_task(_run(), name=new_run_id)
        sup.bind_task(task)
    except BaseException:
        _supervisors(request).pop(new_run_id, None)
        if registered:
            _run_index(request).forget(new_run_id)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.debug("resume task cancelled during rollback")
        sup.release()
        try:
            await components.gateway.close()
        except Exception:
            logger.warning("gateway.close() failed during resume rollback", exc_info=True)
        raise
    return {"run_id": new_run_id, "resumed_from": run_id, "workspace_root": str(workspace)}


@router.get("/runs")
def list_runs(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    status: str | None = None,
    outcome: str | None = None,
) -> dict[str, Any]:
    runs, total = _history_runs(request, limit=offset + limit + 1, status=status, outcome=outcome)
    return {
        "runs": [r.model_dump() for r in runs[offset : offset + limit]],
        "truncated": total > offset + limit,
        "next_offset": offset + limit if total > offset + limit else None,
        "total": total,
    }


@router.get("/runs/active")
def list_active_runs(request: Request) -> dict[str, Any]:
    supervisors = _supervisors(request)
    return {"active": [rid for rid, s in supervisors.items() if s.is_active]}


@router.get("/runs/export")
def export_runs(
    request: Request,
    format: str = Query("csv"),
    limit: int = Query(10_000, ge=1, le=10_000),
    offset: int = Query(0, ge=0),
) -> StreamingResponse:
    if format != "csv":
        raise HTTPException(status_code=400, detail="only csv supported")
    history, _ = _history_runs(request, limit=offset + limit + 1)
    page = history[offset:]
    runs, truncated = page[:limit], len(page) > limit
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["run_id", "status", "outcome", "start_time", "last_event_time", "event_count"])
    for r in runs:
        writer.writerow(
            [r.run_id, r.status, r.outcome or "", r.start_time, r.last_event_time, r.event_count]
        )
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=runs.csv",
            "X-NorthStack-Truncated": str(truncated).lower(),
            "X-NorthStack-Next-Offset": str(offset + len(runs)),
        },
    )


# Compare two runs (registered BEFORE /runs/{run_id} so the static "compare"
# path segment wins route matching over the {run_id} path parameter).
@router.get("/runs/compare")
def compare_runs(request: Request, a: str = Query(...), b: str = Query(...)) -> dict[str, Any]:
    sa, sb = _snapshot_for_run(request, a), _snapshot_for_run(request, b)

    def _usage(snap):
        u = snap.get("usage") or {}
        return {
            "total_calls": u.get("total_calls", 0),
            "total_cost_usd": u.get("total_cost_usd", 0.0),
            "total_input_tokens": u.get("total_input_tokens", 0),
            "total_output_tokens": u.get("total_output_tokens", 0),
        }

    ua, ub = _usage(sa), _usage(sb)
    return {
        "a": a,
        "b": b,
        "goal": {"a": sa.get("goal"), "b": sb.get("goal")},
        "outcome": {"a": sa.get("outcome"), "b": sb.get("outcome")},
        "failure_type": {"a": sa.get("failure_type"), "b": sb.get("failure_type")},
        "status": {"a": sa.get("status"), "b": sb.get("status")},
        "events_replayed": {"a": sa.get("events_replayed"), "b": sb.get("events_replayed")},
        "usage": {"a": ua, "b": ub},
        "delta": {
            "total_calls": ub["total_calls"] - ua["total_calls"],
            "total_cost_usd": round(ub["total_cost_usd"] - ua["total_cost_usd"], 6),
            "total_input_tokens": ub["total_input_tokens"] - ua["total_input_tokens"],
            "total_output_tokens": ub["total_output_tokens"] - ua["total_output_tokens"],
        },
    }


def _run_ledger_or_404(request: Request, run_id: str) -> Ledger:
    ledger = _resolve_ledger(request, run_id)
    return ledger


def _event_dict(e: EventEnvelope) -> dict[str, Any]:
    """Serialize a ledger event for JSON responses. Single source."""
    return {
        "run_id": e.run_id,
        "seq": e.seq,
        "kind": e.kind.value,
        "payload": e.payload.model_dump(mode="json"),
        "timestamp": e.timestamp,
        "prev_hash": e.prev_hash,
        "hash_chain": e.hash_chain,
    }


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict[str, Any]:
    ledger = _run_ledger_or_404(request, run_id)
    active = _is_active(request, run_id)
    try:
        if active:
            cache: ProjectionCache = request.app.state.projections
            tail = ledger.events_since(run_id, since=cache.cursor(run_id))
            state = cache.project(run_id, ledger, tail)
        else:
            state = replay_run(ledger, run_id)
        snap = state.snapshot()
        snap["active"] = active
        return snap
    finally:
        ledger.close()


@router.get("/runs/{run_id}/events")
def get_events(
    run_id: str,
    request: Request,
    since: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
) -> dict[str, Any]:
    ledger = _run_ledger_or_404(request, run_id)
    try:
        tail = ledger.events_since(run_id, since=since, limit=limit)
        next_seq = tail[-1].seq if tail else since
        return {
            "events": [_event_dict(e) for e in tail],
            "next_seq": next_seq,
        }
    finally:
        ledger.close()


@router.get("/runs/{run_id}/integrity")
def get_integrity(run_id: str, request: Request) -> dict[str, Any]:
    ledger = _run_ledger_or_404(request, run_id)
    try:
        result = ledger.verify_integrity(run_id)
        return result.model_dump()
    finally:
        ledger.close()


@router.post("/runs/{run_id}/stop")
async def stop_run(run_id: str, request: Request) -> dict[str, Any]:
    sup = _supervisors(request).get(run_id)
    task = sup.task if sup is not None else None
    if task is None:
        raise HTTPException(status_code=404, detail=f"no active run: {run_id}")
    task.cancel()
    request.app.state.projections.invalidate(run_id)
    return {"run_id": run_id, "stopping": True}


@router.delete("/runs/{run_id}")
def delete_run(run_id: str, request: Request) -> dict[str, Any]:
    """Tombstone a finished run: hidden from history, ledger events kept."""
    if _is_active(request, run_id):
        raise HTTPException(status_code=409, detail="run is active; stop it before deleting")
    ledger = _resolve_ledger(request, run_id)
    try:
        if not ledger.events(run_id):
            raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
        ledger.tombstone_run(run_id)
    finally:
        ledger.close()
    return {"run_id": run_id, "deleted": True}


@router.get("/runs/{run_id}/ledger.json")
def export_ledger(
    run_id: str,
    request: Request,
    since: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=5000),
) -> dict[str, Any]:
    ledger = _run_ledger_or_404(request, run_id)
    try:
        page = ledger.events_since(run_id, since=since, limit=limit + 1)
        events, truncated = page[:limit], len(page) > limit
        return {
            "run_id": run_id,
            "events": [_event_dict(e) for e in events],
            "next_seq": events[-1].seq if events else since,
            "truncated": truncated,
        }
    finally:
        ledger.close()
