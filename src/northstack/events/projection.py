"""Fold typed events into an inspectable ``RunState``.

Pure: every handler returns a new state rather than mutating in place, and no
handler swallows a parse error -- the payload arrived validated, so there is
nothing left to guess.  The ``match`` closes with ``assert_never``, which makes
"added an event kind but forgot to project it" a *type* error at check time
instead of a runtime surprise on the one run that emits it.
"""

from __future__ import annotations

from typing import assert_never

from northstack.domain import (
    CellStatus,
    EvidenceManifest,
    GraphVersion,
    RunOutcome,
    RunState,
    RunStatus,
    WorkContract,
)
from northstack.events.catalog import (
    AnalysisCompleted,
    AnalysisRequested,
    ArtifactStored,
    BudgetUpdated,
    CellAdvanced,
    CellCompleted,
    CellCreated,
    CellFailed,
    CellProgress,
    CellStarted,
    ClaimRecorded,
    ContractAmended,
    ContractProposed,
    ContractValidated,
    EvidenceRecorded,
    GraphAccepted,
    GraphProposed,
    OutcomeEmitted,
    RecoveryTransition,
    RequestAccepted,
    RouteSelected,
    RunCreated,
    StallDetected,
    StatusChanged,
    VerificationCheck,
    WorkspaceSnapshot,
)
from northstack.events.envelope import EventEnvelope

__all__ = ["fold"]


def fold(state: RunState, event: EventEnvelope) -> RunState:
    """Return the state after applying one event."""
    match event.payload:
        case StatusChanged() as p:
            return _status_changed(state, p, event.seq)
        case RequestAccepted() as p:
            return state.model_copy(
                update={
                    "goal": p.goal,
                    "workspace_root": p.workspace_root,
                    "run_budget": p.budget,
                }
            )
        case ContractProposed() as p:
            return state.model_copy(
                update={
                    "contract_version": p.version,
                    "current_contract": WorkContract(
                        id=p.id,
                        version=p.version,
                        objective=p.objective,
                        scope=p.scope,
                        deliverables=p.deliverables,
                        constraints=p.constraints,
                        allowed_tools=p.allowed_tools,
                        workspace_scope=p.workspace_scope,
                        budget=p.budget,
                        acceptance_criteria=p.acceptance_criteria,
                        unresolved_ambiguity=p.unresolved_ambiguity,
                    ),
                }
            )
        case ContractAmended() as p:
            return state.model_copy(update={"contract_version": p.version})
        case GraphAccepted() as p:
            return state.model_copy(
                update={
                    "graph_version": p.version,
                    "graph": GraphVersion(
                        version=p.version,
                        cells=p.cells,
                        edges=p.edges,
                        milestones=p.milestones,
                    ),
                }
            )
        case CellCreated() as p:
            return state.model_copy(update={"cells": [*state.cells, p.cell]})
        case RouteSelected() as p:
            return state.model_copy(update={"routes": {**state.routes, p.cell_id: p.profile_name}})
        case CellStarted() as p:
            return _set_cell_status(state, p.cell_id, CellStatus.RUNNING)
        case CellCompleted() as p:
            return _set_cell_status(state, p.cell_id, CellStatus.COMPLETED)
        case CellFailed() as p:
            return _set_cell_status(state, p.cell_id, CellStatus.FAILED)
        case CellAdvanced() as p:
            return _set_cell_status(state, p.cell_id, p.status)
        case BudgetUpdated() as p:
            return state.model_copy(update={"usage": p.usage})
        case EvidenceRecorded() as p:
            return _evidence_recorded(state, p)
        case OutcomeEmitted() as p:
            return state.model_copy(update={"outcome": p.outcome})
        case RecoveryTransition() as p:
            return _recovery_transition(state, p)
        case StallDetected():
            return state.model_copy(update={"outcome": RunOutcome.ABSTAINED})
        case (
            RunCreated()
            | WorkspaceSnapshot()
            | AnalysisRequested()
            | AnalysisCompleted()
            | ContractValidated()
            | GraphProposed()
            | ClaimRecorded()
            | ArtifactStored()
            | VerificationCheck()
            | CellProgress()
        ):
            return state
        case _ as unreachable:
            assert_never(unreachable)


def _status_changed(state: RunState, payload: StatusChanged, seq: int) -> RunState:
    if not RunStatus.can_transition(state.status, payload.status):
        raise ValueError(
            "Illegal transition during replay: "
            f"{state.status.value} -> {payload.status.value} (at seq {seq})"
        )
    return state.model_copy(update={"status": payload.status})


def _set_cell_status(state: RunState, cell_id: str, status: CellStatus) -> RunState:
    """One status vocabulary, applied to both cell views of the same id."""
    return state.model_copy(
        update={
            "cells": [
                c.model_copy(update={"status": status}) if c.id == cell_id else c
                for c in state.cells
            ],
            "graph": state.graph.with_cell_status(cell_id, status) if state.graph else None,
        }
    )


def _evidence_recorded(state: RunState, payload: EvidenceRecorded) -> RunState:
    manifest = EvidenceManifest(
        outcome=payload.outcome,
        records=payload.records,
        tools_used=payload.tools_used,
        usage=payload.usage,
        hard_gate_failures=payload.hard_gate_failures,
        material_disagreement=payload.material_disagreement,
    )
    cells = state.cells
    if manifest.outcome == RunOutcome.VERIFIED:
        cells = [
            c.model_copy(update={"status": CellStatus.VERIFIED})
            if c.status == CellStatus.COMPLETED
            else c
            for c in cells
        ]
    return state.model_copy(update={"evidence_manifest": manifest, "cells": cells})


def _recovery_transition(state: RunState, payload: RecoveryTransition) -> RunState:
    return state.model_copy(
        update={
            "failure_type": payload.failure_type,
            "recovery_events": [
                *state.recovery_events,
                {
                    "cell_id": payload.cell_id,
                    "failure_type": payload.failure_type.value,
                    "action": payload.action.value,
                    "attempt_number": payload.attempt_number,
                },
            ],
        }
    )
