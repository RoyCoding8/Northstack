"""Shared event-construction helpers for tests.

Tests that only care about sequence numbers, ordering, or hash chaining pick
the placeholder branch, and tests that assert on event content name the model
they mean.  No test should build an ``EventEnvelope`` with a hand-rolled
``payload`` dict or a bare ``kind`` kwarg -- both are the seams the catalog
closes.
"""

from __future__ import annotations

from northstack.events.catalog import EventPayload, RunCreated
from northstack.events.envelope import EventEnvelope

__all__ = ["env"]


def env(
    seq: int = 1,
    payload: EventPayload | None = None,
    *,
    run_id: str = "r1",
    prev_hash: str = "",
    hash_chain: str = "",
) -> EventEnvelope:
    """Envelope for tests that care about ``seq``/chaining, not payload shape.

    ``payload`` defaults to a placeholder ``RunCreated()`` so callers that only
    need sequence numbers stay terse; tests asserting on content pass the model
    they mean.  Tests needing a crafted chain pass ``prev_hash``/``hash_chain``.
    """
    return EventEnvelope(
        run_id=run_id,
        seq=seq,
        payload=payload or RunCreated(),
        prev_hash=prev_hash,
        hash_chain=hash_chain,
    )
