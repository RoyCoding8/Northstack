"""Workspace file browser endpoints (workspace-relative paths only).

All path access goes through ``RestrictedWorkspace`` (path containment, NOT
a security sandbox -- documented).  Reads are byte-bounded and truncated by
the workspace config; the response carries the ``truncated``/``total_bytes``
flags so the UI can show "read more."

Workspace discovery: the operator picks a base directory; this endpoint lists
sub-directories that contain a ``.northstack/ledger.db`` (i.e. have been
operated on by northstack).  The base is resolved from a query param under
a configurable allowlist root (default: the cwd).  Traversal outside the base
is rejected.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from northstack.adapters.workspace.restricted import RestrictedWorkspace, is_sensitive_path

router = APIRouter(tags=["files"])
_MAX_WORKSPACE_SCAN = 10_000


def _base_root(request: Request) -> Path:
    """Allowed base directory for workspace discovery (default cwd)."""
    return Path(getattr(request.app.state, "files_base_root", ".") or ".").resolve()


def _is_within(path: Path, root: Path) -> bool:
    """True iff ``path`` is ``root`` itself or a descendant of ``root``.

    Both arguments must already be resolved.  Uses ``Path.is_relative_to``
    (Python 3.9+; our floor is 3.12).
    """
    try:
        return path.is_relative_to(root)
    except (ValueError, TypeError):
        return False


@router.get("/files/workspaces")
def list_workspaces(
    request: Request,
    base: str | None = Query(None, description="Base dir to scan for workspaces"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0, le=10_000),
) -> dict[str, Any]:
    """List workspace dirs under ``base`` that carry a .northstack/ledger.db.

    ``base`` is confined to the configured allowlist root
    (``files_base_root``, default cwd): a caller-supplied base is resolved and
    rejected if it escapes that root.  This honours the module's documented
    invariant -- "traversal outside the base is rejected" -- which previously
    held only when ``base`` was omitted.
    """
    allow_root = _base_root(request)
    if base:
        root = Path(base).expanduser().resolve()
        if not _is_within(root, allow_root):
            raise HTTPException(
                status_code=400,
                detail=f"base '{base}' resolves outside the allowlist root {allow_root}",
            )
    else:
        root = allow_root
    try:
        if not root.is_dir():
            raise HTTPException(status_code=400, detail=f"not a directory: {root}")
        found, scanned, scan_truncated = [], 0, False
        with os.scandir(root) as entries:
            for entry in entries:
                scanned += 1
                if scanned > _MAX_WORKSPACE_SCAN:
                    scan_truncated = True
                    break
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    child = Path(entry.path)
                    if (child / ".northstack" / "ledger.db").is_file():
                        found.append({"path": str(child), "name": entry.name})
                except OSError:
                    continue
    except HTTPException:
        raise
    except PermissionError as exc:
        raise HTTPException(
            status_code=403,
            detail={"category": "workspace_base_unreadable", "path": str(root)},
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail={"category": "workspace_base_unreadable", "path": str(root)},
        ) from exc
    found.sort(key=lambda item: (item["name"].casefold(), item["path"]))
    page = found[offset : offset + limit]
    truncated = scan_truncated or offset + limit < len(found)
    return {
        "base": str(root),
        "workspaces": page,
        "truncated": truncated,
        "next_offset": offset + len(page) if truncated and page else None,
        "scanned": min(scanned, _MAX_WORKSPACE_SCAN),
    }


def _open_workspace(request: Request, workspace: str) -> RestrictedWorkspace:
    ws = Path(workspace).expanduser().resolve()
    if not _is_within(ws, _base_root(request)):
        raise HTTPException(status_code=403, detail="workspace is outside the allowed root")
    if not ws.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {ws}")
    return RestrictedWorkspace(ws)


def _decode_entries(result) -> list[str]:
    """Decode the structured filename array returned by list()."""
    if not result.ok:
        raise HTTPException(status_code=400, detail=result.error or "list failed")
    return json.loads(result.data) if result.data else []


@router.get("/files/tree")
def get_tree(
    request: Request,
    workspace: str = Query(..., description="Workspace root"),
    path: str = Query(".", description="Workspace-relative dir to list"),
) -> dict[str, Any]:
    ws = _open_workspace(request, workspace)
    path = path or "."
    entries = _decode_entries(ws.list(path))
    root = Path(workspace).resolve()
    base = (root / path).resolve() if path and path != "." else root
    items = []
    for name in entries:
        full = base / name
        try:
            is_dir = full.is_dir()
        except OSError:
            is_dir = False
        items.append({"name": name, "type": "dir" if is_dir else "file"})
    return {"workspace": str(root), "path": path, "entries": items}


@router.get("/files/read")
def read_file(
    request: Request,
    workspace: str = Query(...),
    path: str = Query(..., description="Workspace-relative file path"),
) -> dict[str, Any]:
    if is_sensitive_path(path):
        raise HTTPException(status_code=403, detail="sensitive file reads are denied")
    ws = _open_workspace(request, workspace)
    result = ws.read(path)
    if not result.ok:
        if result.error_kind == "sensitive_denied":
            raise HTTPException(status_code=403, detail="sensitive file reads are denied")
        raise HTTPException(status_code=400, detail=result.error or "read failed")
    content = (result.data or b"").decode("utf-8", errors="replace")
    return {
        "workspace": str(Path(workspace).resolve()),
        "path": path,
        "content": content,
        "truncated": bool(result.truncated),
        "total_bytes": int(result.total_bytes),
    }


@router.get("/files/artifacts")
def list_artifacts(
    request: Request,
    run_id: str = Query(...),
    workspace: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0, le=10_000),
    cursor: str | None = Query(None, max_length=512),
) -> dict[str, Any]:
    """List artifact blobs stored for a run under the workspace's artifacts dir.

    Artifacts live in a per-WORKSPACE dir (``.northstack/artifacts``), not a
    per-run dir.  When ``workspace`` is supplied explicitly (rather than looked
    up from the active run map), we must still confirm ``run_id`` actually ran
    in that workspace's ledger -- otherwise a caller who passes a *nonexistent*
    run_id plus some real workspace would receive THAT workspace's artifacts
    under the wrong run_id.  404 on unknown run; ``artifacts: []`` on a known
    run whose workspace has no artifacts dir yet.
    """
    index = getattr(request.app.state, "run_index", None)
    indexed = index.workspace_of(run_id) if index is not None else None
    if indexed is None:
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
    resolved_ws = Path(indexed).expanduser().resolve()
    if workspace is not None:
        requested = Path(workspace).expanduser().resolve()
        if os.path.normcase(str(requested)) != os.path.normcase(str(resolved_ws)):
            raise HTTPException(status_code=404, detail=f"run not found in workspace: {run_id}")
    if not _is_within(resolved_ws, _base_root(request)):
        raise HTTPException(status_code=403, detail="workspace is outside the allowed root")

    art_dir = resolved_ws / ".northstack" / "artifacts"
    if not art_dir.is_dir():
        return {
            "run_id": run_id,
            "artifacts": [],
            "root": str(art_dir),
            "truncated": False,
            "next_offset": None,
            "next_cursor": None,
        }
    items: list[dict[str, Any]] = []
    stack = [art_dir]
    scanned = 0
    scan_truncated = False
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        child_dirs: list[Path] = []
        for entry in entries:
            scanned += 1
            if scanned > 10_000:
                scan_truncated = True
                stack.clear()
                break
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    child_dirs.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=False) or entry.name.startswith("."):
                    continue
                size = entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
            items.append({"path": str(Path(entry.path).relative_to(art_dir)), "size_bytes": size})
        stack.extend(sorted(child_dirs, reverse=True))
    items.sort(key=lambda item: item["path"])
    eligible = [item for item in items if cursor is None or item["path"] > cursor]
    start = 0 if cursor is not None else offset
    page = eligible[start : start + limit]
    truncated = scan_truncated or start + limit < len(eligible)
    next_offset = offset + len(page) if truncated and page else None
    return {
        "run_id": run_id,
        "artifacts": page,
        "root": str(art_dir),
        "truncated": truncated,
        "next_offset": next_offset,
        "next_cursor": page[-1]["path"] if truncated and page else None,
    }
