"""Fold a run's ledger events into a projected ``RunState``.

This lives above both storage and the projection so neither depends on the
other: ``adapters.sqlite_ledger`` yields envelopes, ``events.projection``
folds them, and only this module knows about both.
"""

from __future__ import annotations

from collections.abc import Iterable

from northstack.adapters.sqlite_ledger import Ledger
from northstack.domain.run_state import RunState
from northstack.events.envelope import EventEnvelope
from northstack.events.projection import fold


def fold_events(state: RunState, events: Iterable[EventEnvelope]) -> RunState:
    """Apply events onto ``state`` in order, returning the folded result.

    Incremental by construction: callers holding a cached state pass only the
    events past their cursor instead of re-folding from seq 1.
    """
    for event in events:
        state = fold(state, event).model_copy(
            update={
                "events_replayed": state.events_replayed + 1,
                "last_event_hash": event.hash_chain,
            }
        )
    return state


def replay_run(ledger: Ledger, run_id: str) -> RunState:
    """Project a run's full event history into a ``RunState``.

    STATUS_CHANGED events are validated against the state machine: illegal
    transitions raise ``ValueError``.
    """
    return fold_events(RunState(run_id=run_id), ledger.events(run_id))
