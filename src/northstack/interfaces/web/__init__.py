"""Web control surface for northstack (FastAPI, localhost only).

Optional dependency group: ``uv sync --extra web``.  Imports of FastAPI are
deferred to the server module so the package imports cleanly without the web
extras installed (the core CLI/tests never pull in fastapi).
"""
