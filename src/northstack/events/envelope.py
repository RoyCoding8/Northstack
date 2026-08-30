"""The hash-chained envelope carrying one typed payload.

``kind`` and ``schema_version`` are read off the payload rather than stored
beside it: two independently-set copies of the same fact can disagree, and the
one on the payload is the one the catalog validates.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field

from northstack.events.catalog import EventKind, EventPayload

__all__ = ["EventEnvelope", "EventKind"]


class EventEnvelope(BaseModel):
    """Versioned, hash-chained event envelope.

    ``hash_chain`` is computed by the ledger after redaction, not by the model;
    on construction it defaults to empty and the ledger populates it.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    run_id: str
    seq: int = Field(ge=1)
    payload: EventPayload
    timestamp: float = Field(default_factory=time.time)
    prev_hash: str = Field(default="")
    hash_chain: str = Field(
        default="", description="SHA-256 hex; computed by ledger after redaction"
    )

    @property
    def kind(self) -> EventKind:
        return self.payload.kind

    @property
    def schema_version(self) -> int:
        return self.payload.schema_version
