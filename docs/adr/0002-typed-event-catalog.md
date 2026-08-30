# ADR 0002: Typed Event Catalog (KD1)

Accepted.
## Context

The control plane persists every state transition as an event in a hash-chain ledger. For most of the project's life an event was an `EventKind` string plus an opaque, untyped dict. Projection reached into that dict with `p.get(...)` and `p["..."]`; a typo or a renamed field was a silent runtime wrong-answer, not a type error. The ledger, the projection, and the test fixtures each spelled out independently what an event "looked like", so they could — and did — drift.

## Decision

One pydantic model per `EventKind`, in a single catalog (`src/northstack/events/catalog.py`):

- `EventKind` is a `str` enum of every event the control plane emits (`RUN_CREATED`, `STATUS_CHANGED`, `REQUEST_ACCEPTED`, `WORKSPACE_SNAPSHOT`, `ANALYSIS_REQUESTED`, `ANALYSIS_COMPLETED`, `CONTRACT_PROPOSED`, `CONTRACT_VALIDATED`, `CONTRACT_AMENDED`, `GRAPH_PROPOSED`, `GRAPH_ACCEPTED`, `CELL_CREATED`, `CELL_ADVANCED`, `ROUTE_SELECTED`, `CELL_STARTED`, `CELL_COMPLETED`, `CELL_FAILED`, `CLAIM_RECORDED`, `ARTIFACT_STORED`, `BUDGET_UPDATED`, `VERIFICATION_CHECK`, `EVIDENCE_RECORDED`, `RECOVERY_TRANSITION`, `OUTCOME_EMITTED`, `STALL_DETECTED`).
- Each payload model pins its `kind` with a `Literal[EventKind.…]`, so the catalog is self-documenting and a wrong-kind assignment is a type error.
- `PAYLOAD_BY_KIND: dict[EventKind, type[PayloadBase]]` is the registry, and the module self-checks at import: `set(PAYLOAD_BY_KIND) != set(EventKind)` raises. A new `EventKind` with no payload model fails loud, at import, not silently at first replay.
- The projection is `events/projection.py:fold(state, event) -> RunState`, not a method on `Ledger`. The ledger's job is append plus integrity; the projection's job is to fold envelopes into a `RunState`. The two concerns no longer share a class (`application/replay.py` drives `Ledger` → `fold_events` → `fold`).

## Why

- A renamed field is now a compile-time error in every consumer at once, instead of a runtime wrong-answer somewhere.
- The import-time check closes the "added an `EventKind` but forgot the model" gap, which under an opaque-dict design stays open until a run hits that branch.
- Keeping the projection out of `Ledger` keeps the storage adapter a pure append/integrity seam (ADR 0001, decision 1): the ledger can't depend on a read model, because it doesn't hold one.

## Consequences

- Adding an event is three coordinated edits — enum member, payload model, registry entry — which is the point: the three can't drift silently.
- Legacy on-disk events are reconciled by `events/upcast.py` before folding, so the typed catalog landed without a ledger migration.
