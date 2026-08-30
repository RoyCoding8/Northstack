"""Schema evolution for stored payloads.

A stored payload is brought up to ``CURRENT_SCHEMA_VERSION`` by applying the
registered upcaster for each intermediate version.  Two rules make this sound:

  - a version newer than this build understands is *refused*, never
    best-effort parsed;
  - a gap in the ladder is refused too, so "we forgot to write the upcaster"
    cannot degrade into a silent partial parse.

The registry is empty at v1.  It exists from day one so the next format change
is additive rather than a migration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from northstack.events.catalog import CURRENT_SCHEMA_VERSION, EventKind
from northstack.events.errors import LedgerCorruption, UnknownSchemaVersion

Upcaster = Callable[[dict[str, Any]], dict[str, Any]]

UPCASTERS: dict[tuple[EventKind, int], Upcaster] = {}


def register(kind: EventKind, from_version: int) -> Callable[[Upcaster], Upcaster]:
    """Register the step that lifts ``kind`` from ``from_version`` to the next."""

    def decorate(fn: Upcaster) -> Upcaster:
        UPCASTERS[(kind, from_version)] = fn
        return fn

    return decorate


def upcast(kind: EventKind, data: Mapping[str, Any], *, seq: int) -> dict[str, Any]:
    """Return ``data`` lifted to the current schema version for ``kind``."""
    current = dict(data)
    version = current.get("schema_version", 1)
    if not isinstance(version, int):
        raise LedgerCorruption(seq, kind.value, f"schema_version is not an int: {version!r}")
    if version > CURRENT_SCHEMA_VERSION:
        raise UnknownSchemaVersion(seq, kind.value, version, CURRENT_SCHEMA_VERSION)
    while version < CURRENT_SCHEMA_VERSION:
        step = UPCASTERS.get((kind, version))
        if step is None:
            raise LedgerCorruption(seq, kind.value, f"no upcaster from schema_version {version}")
        current = step(current)
        version += 1
        current["schema_version"] = version
    return current
