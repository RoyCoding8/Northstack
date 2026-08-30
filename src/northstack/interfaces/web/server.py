"""FastAPI control surface for the northstack (localhost only).

Composition + lifecycle:
  - ``lifespan`` loads ``northstack.toml`` via ``NorthStackConfig.from_toml``,
    loads secrets from the config's ``.env`` (mirrors ``tui.load_secrets``),
    and constructs a single shared ``ConfigStore`` (the in-memory editable
    config).  Each run is owned by a :class:`RunSupervisor` held in
    ``app.state.supervisors`` keyed by run_id -- the supervisor replaces the
    former hand-synced ``active_runs`` / ``run_ledgers`` / ``run_workspaces``
    dicts.  On shutdown every still-running task is cancelled and its
    supervisor released (each task's ``run_async`` finally closes its own
    loop-bound gateway).
  - A run is started via ``POST /api/runs``: build a fresh ``Company`` per run
    (fresh gateway -- see ``build.py``), pre-generate ``run_id``, wrap
    ``company.run_async(request, run_id=run_id)`` + ``ledger.close()`` in an
    inner ``async def _run()``, and ``asyncio.create_task(_run())`` it on the
    server's event loop.  The handler returns ``{"run_id": run_id}``
    synchronously.  NEVER ``asyncio.run``.

Security posture (documented, not a sandbox):
  - localhost only (bound to 127.0.0.1 by the entry point).
  - ``RestrictedWorkspace`` is path-containment, NOT a security sandbox.
  - Secrets are env-var references only; this module never reads or emits a
    secret value -- only the env-var name + a resolved OK/UNSET status.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from northstack.adapters.sqlite_ledger import UnsupportedDatabaseSchema
from northstack.application.projection_cache import ProjectionCache
from northstack.application.run_index import RunIndex, discover_workspaces
from northstack.config import NorthStackConfig
from northstack.domain.url_policy import is_loopback_host
from northstack.events.errors import LedgerCorruption
from northstack.interfaces.tui import load_secrets
from northstack.interfaces.web.config_store import ConfigStore
from northstack.interfaces.web.routes_config import router as config_router
from northstack.interfaces.web.routes_files import router as files_router
from northstack.interfaces.web.routes_runs import router as runs_router

logger = logging.getLogger(__name__)


def create_app(
    config_path: Path,
    *,
    static_dir: Path | None = None,
    api_token: str | None = None,
    files_base_root: Path | None = None,
) -> FastAPI:
    """Build the FastAPI app bound to a config file path.

    ``config_path``: the TOML config that backs the ``ConfigStore`` (edits
    persist here on explicit save).  ``static_dir``: the frontend assets;
    defaults to the ``static/`` directory next to this module.
    """
    config_path = Path(config_path).resolve()
    config = NorthStackConfig.from_toml(config_path)
    load_secrets(config_path)
    store = ConfigStore(config, config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store
        app.state.config_path = config_path
        app.state.supervisors = {}
        app.state.projections = ProjectionCache()
        app.state.run_index = RunIndex()
        resolved_root = await asyncio.to_thread(
            lambda: str(Path(app.state.files_base_root).resolve())
        )
        roots = [resolved_root, *discover_workspaces(resolved_root)]
        app.state.run_index.load_historical(roots)
        try:
            yield
        finally:
            supervisors = list(app.state.supervisors.values())
            for sup in supervisors:
                task = sup.task
                if task is not None:
                    task.cancel()
            for sup in supervisors:
                task = sup.task
                if task is not None:
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110 - shutdown: swallow task errors
                        pass
                sup.release()
            app.state.supervisors.clear()

    app = FastAPI(
        title="NorthStack Control Surface",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.config_path = config_path
    app.state.files_base_root = str((files_base_root or Path.cwd()).resolve())
    app.state.api_token = api_token

    @app.exception_handler(LedgerCorruption)
    async def ledger_corruption(_request: Request, exc: LedgerCorruption) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "category": "ledger_corruption"},
        )

    @app.exception_handler(UnsupportedDatabaseSchema)
    async def unsupported_database_schema(
        _request: Request, exc: UnsupportedDatabaseSchema
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "category": "unsupported_database_schema"},
        )

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        correlation_id = uuid.uuid4().hex
        logger.exception(
            "Unhandled web error correlation_id=%s path=%s",
            correlation_id,
            request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "internal server error",
                "category": "internal_error",
                "correlation_id": correlation_id,
            },
            headers={"X-Correlation-ID": correlation_id},
        )

    if api_token:

        @app.middleware("http")
        async def authenticate_api(request: Request, call_next):
            if request.url.path.startswith("/api/"):
                authorization = request.headers.get("authorization", "")
                scheme, _, supplied = authorization.partition(" ")
                valid = (
                    scheme.lower() == "bearer"
                    and bool(supplied)
                    and hmac.compare_digest(supplied, api_token)
                )
                if not valid:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "invalid or missing bearer token"},
                        headers={"WWW-Authenticate": "Bearer"},
                    )
            return await call_next(request)

    app.include_router(config_router, prefix="/api")
    app.include_router(runs_router, prefix="/api")
    app.include_router(files_router, prefix="/api")

    if static_dir is None:
        static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def root() -> object:
            from fastapi.responses import FileResponse

            return FileResponse(str(static_dir / "index.html"))

    return app


def _ensure_web_assets() -> None:
    """The Tailwind bundle is a build artifact, not committed source.

    In a source checkout, build it with npm on startup. An installed wheel
    cannot rebuild, so there we only point at the fix.
    """
    import shutil
    import subprocess
    import sys

    bundle = Path(__file__).parent / "static" / "components.css"
    if bundle.is_file():
        return
    root = next(
        (p for p in Path(__file__).resolve().parents if (p / "package.json").is_file()), None
    )
    npm = shutil.which("npm") if root else None
    if npm is None:
        print(
            "northstack-web: static/components.css is missing; "
            "run `npm install && npm run build:css` in the repository root",
            file=sys.stderr,
        )
        return
    print("northstack-web: building web assets (npm run build:css)...")
    subprocess.run([npm, "run", "build:css"], cwd=root, check=False)  # noqa: S603 - fixed argv


def main() -> None:
    """Entry point ``northstack-web``. Binds 127.0.0.1 only."""
    import argparse

    import uvicorn

    _ensure_web_assets()
    parser = argparse.ArgumentParser(prog="northstack-web")
    parser.add_argument("--config", type=Path, default=Path("northstack.toml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--dangerous-allow-non-loopback", action="store_true")
    parser.add_argument("--dangerous-no-auth", action="store_true")
    parser.add_argument("--token-env", default="NORTHSTACK_WEB_TOKEN")
    parser.add_argument("--files-base-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    api_token = os.environ.get(args.token_env, "").strip() or None
    if not is_loopback_host(args.host):
        if not args.dangerous_allow_non_loopback:
            parser.error("non-loopback binding requires --dangerous-allow-non-loopback")
        if api_token is None and not args.dangerous_no_auth:
            parser.error(f"non-loopback binding requires a token in {args.token_env}")

    app = create_app(
        args.config,
        api_token=api_token,
        files_base_root=args.files_base_root,
    )
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
