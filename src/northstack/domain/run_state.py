"""Projected run state.

``RunState`` is a read model folded from the ledger; it is never a store.
Nothing writes it except the projection.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from northstack.domain.budget import Budget, BudgetUsage
from northstack.domain.contract import WorkContract
from northstack.domain.graph import GraphCell, GraphVersion
from northstack.domain.outcome import (
    AttemptSignature,
    CalibrationRecord,
    EvidenceManifest,
    FailureType,
    RunOutcome,
)
from northstack.domain.status import RunStatus


class RunState(BaseModel):
    """Projected state from event replay.

    Tracks contract version, graph version, evidence, usage, and outcome
    summaries sufficient for ``inspect``.
    """

    run_id: str
    status: RunStatus = Field(default=RunStatus.INTAKE)
    cells: list[GraphCell] = Field(default_factory=list)
    events_replayed: int = Field(default=0, ge=0)
    last_event_hash: str = Field(default="")

    goal: str = Field(default="")
    workspace_root: str = Field(default="")
    contract_version: int = Field(default=0, ge=0)
    graph_version: int = Field(default=0, ge=0)
    graph: GraphVersion | None = None
    current_contract: WorkContract | None = None
    routes: dict[str, str] = Field(default_factory=dict, description="cell_id -> profile_name")
    evidence_manifest: EvidenceManifest | None = None
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    run_budget: Budget | None = Field(default=None)
    outcome: RunOutcome | None = None
    failure_type: FailureType | None = None
    attempt_signatures: list[AttemptSignature] = Field(default_factory=list)
    recovery_events: list[dict[str, Any]] = Field(default_factory=list)
    calibration_records: list[CalibrationRecord] = Field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        """Return a plain-dict snapshot for inspection."""
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "cells": [c.model_dump() for c in self.cells],
            "events_replayed": self.events_replayed,
            "last_event_hash": self.last_event_hash,
            "goal": self.goal,
            "workspace_root": self.workspace_root,
            "contract_version": self.contract_version,
            "graph_version": self.graph_version,
            "routes": self.routes,
            "outcome": self.outcome.value if self.outcome else None,
            "failure_type": self.failure_type.value if self.failure_type else None,
            "usage": self.usage.model_dump(),
            "budget": (
                {
                    "token_limit": self.run_budget.token_limit,
                    "cost_limit_usd": self.run_budget.cost_limit_usd,
                }
                if self.run_budget is not None
                else None
            ),
            "recovery_events": self.recovery_events,
        }
