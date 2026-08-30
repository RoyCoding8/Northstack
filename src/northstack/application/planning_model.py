"""Model-proposed graph planning with deterministic hardening.

The model PROPOSES a decomposition; the control plane owns every structural
fact. Nothing the model emits is trusted:

  - cell ids are rewritten deterministically (``cell-<run>-<i>``);
  - waves are recomputed from topological levels of the dependency DAG --
    the model's own wave numbers, if any, are ignored;
  - at most one mutating cell per wave is enforced by bumping surplus
    mutating cells into fresh waves;
  - per-cell budgets are allocated from the contract's budget by the model's
    *shares*, normalized so the sum can never exceed the contract;
  - acceptance-criterion links are validated in-range and must cover every
    criterion;
  - the finished graph must pass :meth:`GraphPlanner.validate` -- otherwise
    the planner falls back to the canonical single-cell graph.

Fail-open by design: any gateway error, unparseable proposal, or invalid
graph degrades to the single-cell plan (with a logged reason). Decomposition
is an optimization, not a safety property -- a run must never die because a
planner endpoint hiccuped. The fallback is visible in the ledger: the
accepted graph has exactly one cell.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from northstack.adapters.providers.wire import MessageRole, ModelMessage, ModelRequest
from northstack.application.json_extraction import extract_first_json_object
from northstack.application.planning import GraphPlanner, single_cell_graph
from northstack.config import Role
from northstack.domain.contract import WorkContract
from northstack.ports.protocols import GatewayPort
from northstack.domain.graph import (
    CellMode,
    CellStatus,
    GraphCell,
    GraphEdge,
    GraphVersion,
)

logger = logging.getLogger(__name__)

MAX_PROPOSED_CELLS = 8
MAX_WAVES = 6

_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cells": {
            "type": "array",
            "maxItems": MAX_PROPOSED_CELLS,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "mode": {"type": "string", "enum": ["read_only", "mutating"]},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "acceptance_criterion_indices": {"type": "array", "items": {"integer"}},
                    "budget_share": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["name", "mode"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["cells"],
    "additionalProperties": False,
}


class _ProposedCell(BaseModel):
    """Lenient parse target for one proposed cell."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str = ""
    mode: str = "mutating"
    depends_on: list[str] = Field(default_factory=list)
    acceptance_criterion_indices: list[int] = Field(default_factory=list)
    budget_share: float = Field(default=0.5, ge=0.0, le=1.0)


class _Proposal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    cells: list[_ProposedCell] = Field(default_factory=list)


_PROPOSAL_ADAPTER: TypeAdapter[_Proposal] = TypeAdapter(_Proposal)


class ModelBackedPlanner(GraphPlanner):
    """Proposes a multi-cell decomposition through the PLANNER-role profile.

    Subclasses :class:`GraphPlanner` so the orchestrator's single
    ``planner.plan()/validate()`` seam is unchanged; only ``plan`` is
    overridden, and every proposal is hardened or discarded before it can
    become a graph.
    """

    def __init__(
        self,
        gateway: GatewayPort,
        profile_name: str,
        role_map: dict[Role, list[str]] | None = None,
        *,
        max_output_tokens: int = 2048,
    ) -> None:
        super().__init__(role_map)
        self._gateway = gateway
        self._profile_name = profile_name
        self._max_output_tokens = max_output_tokens

    @property
    def profile_name(self) -> str:
        return self._profile_name

    async def plan(self, contract: WorkContract, run_id: str) -> GraphVersion:
        proposal = await self._request_proposal(contract)
        if proposal is None:
            return self._fallback(contract, run_id, "no parseable proposal")
        graph = self._harden(proposal, contract, run_id)
        if graph is None:
            return self._fallback(contract, run_id, "proposal failed validation")
        errors = self.validate(graph, run_budget=contract.budget)
        if errors:
            return self._fallback(contract, run_id, f"graph invalid: {errors[0]}")
        return graph

    async def _request_proposal(self, contract: WorkContract) -> _Proposal | None:
        criteria_lines = "\n".join(
            f"  [{i}] {c.kind}: {c.description}" for i, c in enumerate(contract.acceptance_criteria)
        )
        prompt = (
            "You are a work decomposition planner. Split the following contract "
            "into a small DAG of execution cells (at most "
            f"{MAX_PROPOSED_CELLS}). Read-only cells (analysis, search) run "
            "concurrently; mutating cells (writing deliverables) run one per "
            "wave. Each criterion index must be covered by at least one cell. "
            'Respond with JSON only: {"cells": [{"name": str, "mode": '
            '"read_only"|"mutating", "depends_on": [names of earlier '
            'cells], "acceptance_criterion_indices": [ints], '
            '"budget_share": float in [0,1]}]}.\n\n'
            f"Objective: {contract.objective}\n"
            f"Scope: {contract.scope or '(none)'}\n"
            f"Deliverables: {', '.join(contract.deliverables)}\n"
            f"Acceptance criteria:\n{criteria_lines}"
        )
        request = ModelRequest(
            profile_name=self._profile_name,
            messages=[ModelMessage(role=MessageRole.USER, content=prompt)],
            output_json_schema=_PROPOSAL_SCHEMA,
            max_output_tokens=self._max_output_tokens,
            temperature=0.0,
        )
        try:
            response = await self._gateway.complete(request)
        except Exception:  # noqa: BLE001
            logger.warning("model planner gateway failure profile=%s", self._profile_name)
            return None
        raw = extract_first_json_object(getattr(response, "text", "") or "")
        if raw is None:
            logger.warning("model planner unparseable response profile=%s", self._profile_name)
            return None
        try:
            return _PROPOSAL_ADAPTER.validate_python(raw)
        except ValidationError:
            logger.warning("model proposal failed schema profile=%s", self._profile_name)
            return None

    def _harden(
        self, proposal: _Proposal, contract: WorkContract, run_id: str
    ) -> GraphVersion | None:
        """Turn a proposal into a legal graph, or None when impossible."""
        proposed = [c for c in proposal.cells if c.name.strip()][:MAX_PROPOSED_CELLS]
        if len(proposed) < 2:
            return None

        ids = [f"cell-{run_id}-{i}" for i in range(len(proposed))]
        name_to_id = {c.name.strip(): ids[i] for i, c in enumerate(proposed)}

        deps: list[set[str]] = []
        for i, cell in enumerate(proposed):
            earlier = set(ids[:i])
            resolved = set()
            for dep in cell.depends_on:
                target = name_to_id.get(dep.strip())
                if target in earlier:
                    resolved.add(target)
            deps.append(resolved)

        waves = self._topological_waves(ids, deps)
        if waves is None:
            return None

        waves = self._separate_mutating(proposed, waves)
        if waves is None or max(waves) >= MAX_WAVES:
            return None

        covered: set[int] = set()
        per_cell_indices: list[list[int]] = []
        for cell in proposed:
            valid = sorted(
                {
                    i
                    for i in cell.acceptance_criterion_indices
                    if isinstance(i, int) and 0 <= i < len(contract.acceptance_criteria)
                }
            )
            per_cell_indices.append(valid)
            covered.update(valid)
        if covered != set(range(len(contract.acceptance_criteria))):
            return None

        shares = [max(cell.budget_share, 0.0) for cell in proposed]
        total_share = sum(shares)
        if total_share <= 0:
            return None
        shares = [s / total_share for s in shares]

        cells: list[GraphCell] = []
        for i, cell in enumerate(proposed):
            mode = CellMode.READ_ONLY if cell.mode == "read_only" else CellMode.MUTATING
            child_budget = contract.budget.model_copy(
                update={
                    "token_limit": (
                        int(contract.budget.token_limit * shares[i])
                        if contract.budget.token_limit is not None
                        else None
                    ),
                    "cost_limit_usd": (
                        round(contract.budget.cost_limit_usd * shares[i], 6)
                        if contract.budget.cost_limit_usd is not None
                        else None
                    ),
                }
            )
            cells.append(
                GraphCell(
                    id=ids[i],
                    name=cell.name.strip()[:50],
                    wave=waves[i],
                    mode=mode,
                    contract=contract.model_copy(
                        update={"id": f"{contract.id}-c{i}", "budget": child_budget}
                    ),
                    status=CellStatus.PENDING,
                    dependencies=sorted(deps[i]),
                    acceptance_criterion_indices=per_cell_indices[i],
                    required_profile_roles=[Role.WORKER.value],
                )
            )

        graph_edges = [
            GraphEdge(from_id=dep, to_id=ids[i], kind="blocks")
            for i, cell_deps in enumerate(deps)
            for dep in sorted(cell_deps)
        ]
        milestones = [ids[-1]]
        return GraphVersion(
            version=1,
            cells=cells,
            edges=graph_edges,
            milestones=milestones,
            current_horizon=max(waves),
        )

    @staticmethod
    def _topological_waves(ids: list[str], deps: list[set[str]]) -> list[int] | None:
        """Wave index per cell = longest dependency chain (Kahn levels)."""
        waves: dict[str, int] = {}
        remaining = list(zip(ids, deps, strict=True))
        progressed = True
        while remaining and progressed:
            progressed = False
            next_round: list[tuple[str, set[str]]] = []
            for cell_id, cell_deps in remaining:
                if all(d in waves for d in cell_deps):
                    waves[cell_id] = 1 + max((waves[d] for d in cell_deps), default=-1)
                    progressed = True
                else:
                    next_round.append((cell_id, cell_deps))
            remaining = next_round
        if remaining:  # cycle
            return None
        return [waves[cid] for cid in ids]

    @staticmethod
    def _separate_mutating(proposed: list[_ProposedCell], waves: list[int]) -> list[int] | None:
        """Bump surplus mutating cells into fresh waves; None if unbounded."""
        adjusted = list(waves)
        mutating_in_wave: dict[int, int] = {}
        for i, cell in enumerate(proposed):
            w = adjusted[i]
            is_mutating = cell.mode != "read_only"
            if is_mutating:
                while w in mutating_in_wave:
                    w += 1
                    if w >= MAX_WAVES:
                        return None
                mutating_in_wave[w] = i
            adjusted[i] = max(w, adjusted[i])
        return adjusted

    def _fallback(self, contract: WorkContract, run_id: str, reason: str) -> GraphVersion:
        logger.info("model planner fell back to single-cell graph: %s", reason)
        cell = GraphCell(
            id=f"cell-{run_id}",
            name=contract.objective[:50],
            wave=0,
            mode=CellMode.MUTATING,
            contract=contract,
            status=CellStatus.PENDING,
            acceptance_criterion_indices=list(range(len(contract.acceptance_criteria))),
            required_profile_roles=[Role.WORKER.value],
        )
        return single_cell_graph(cell)
