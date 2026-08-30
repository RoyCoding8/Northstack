"""Hermetic tests for the model-proposed, deterministically hardened planner.

No network: a fake gateway plays scripted proposals. Pinned invariants:

  - the model PROPOSES, the control plane owns structure: ids are rewritten,
    waves recomputed topologically, budgets clamped to the contract's;
  - at most one mutating cell per wave, enforced by bumping;
  - every criterion must be covered, or the proposal is discarded;
  - any failure (gateway error, unparseable, invalid, single-cell, wave
    explosion) degrades to the canonical single-cell graph -- never an
    exception, never a dead run.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from northstack.adapters.providers.wire import ModelRequest
from northstack.application.planning import GraphPlanner
from northstack.application.planning_model import MAX_WAVES, ModelBackedPlanner
from northstack.application.scheduling import Scheduler
from northstack.config import Role
from northstack.domain.budget import Budget
from northstack.domain.contract import (
    FileDiffCriterion,
    SoftRubricCriterion,
    WorkContract,
)
from northstack.domain.graph import CellMode


class _Resp(BaseModel):
    text: str = ""


class _FakeGateway:
    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self._text = text
        self._error = error
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> _Resp:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return _Resp(text=self._text)


def _contract(token_limit: int = 100_000, cost: float = 5.0) -> WorkContract:
    return WorkContract(
        id="wc-1",
        objective="ship the widget",
        scope="the widget module",
        deliverables=["widget.py", "tests"],
        budget=Budget(token_limit=token_limit, cost_limit_usd=cost),
        acceptance_criteria=[
            FileDiffCriterion(description="widget exists", path="widget.py"),
            SoftRubricCriterion(description="quality"),
        ],
    )


def _planner(gateway: Any) -> ModelBackedPlanner:
    return ModelBackedPlanner(gateway, "planner-mid", {})


_GOOD_PROPOSAL = """
{"cells": [
  {"name": "inspect", "mode": "read_only", "depends_on": [],
   "acceptance_criterion_indices": [1], "budget_share": 0.25},
  {"name": "write widget", "mode": "mutating", "depends_on": ["inspect"],
   "acceptance_criterion_indices": [0], "budget_share": 0.75}
]}
"""


async def test_valid_proposal_hardens_into_legal_multi_cell_graph():
    planner = _planner(_FakeGateway(_GOOD_PROPOSAL))
    graph = await planner.plan(_contract(), "run-1")

    assert len(graph.cells) == 2
    # Deterministic rewritten ids, not the model's names.
    assert [c.id for c in graph.cells] == ["cell-run-1-0", "cell-run-1-1"]
    # Waves from the dependency DAG: dependent cell strictly later.
    waves = {c.id: c.wave for c in graph.cells}
    assert waves["cell-run-1-1"] > waves["cell-run-1-0"]
    assert graph.cells[1].dependencies == ["cell-run-1-0"]
    assert [cell.id for cell in Scheduler().ready_cells(graph)] == ["cell-run-1-0"]
    # Read-only then mutating: exactly one mutating cell per wave.
    mutating_waves = [c.wave for c in graph.cells if c.mode == CellMode.MUTATING]
    assert len(mutating_waves) == len(set(mutating_waves))
    # Every cell requires the worker role (routing stays in charge).
    assert all(c.required_profile_roles == [Role.WORKER.value] for c in graph.cells)
    # Budget shares are normalized and can never exceed the contract.
    total_tokens = sum(c.contract.budget.token_limit or 0 for c in graph.cells)
    assert total_tokens <= 100_000
    total_cost = sum(c.contract.budget.cost_limit_usd or 0 for c in graph.cells)
    assert total_cost <= 5.0 + 1e-9
    # Every criterion covered.
    covered = {i for c in graph.cells for i in c.acceptance_criterion_indices}
    assert covered == {0, 1}
    # The hardened graph passes the production validator.
    assert GraphPlanner().validate(graph) == []


async def test_two_mutating_cells_in_one_wave_are_bumped_apart():
    proposal = """
    {"cells": [
      {"name": "a", "mode": "mutating", "depends_on": [], "acceptance_criterion_indices": [0, 1]},
      {"name": "b", "mode": "mutating", "depends_on": [], "acceptance_criterion_indices": [0]},
      {"name": "c", "mode": "read_only", "depends_on": ["a"], "acceptance_criterion_indices": [1]}
    ]}
    """
    planner = _planner(_FakeGateway(proposal))
    graph = await planner.plan(_contract(), "run-2")
    assert len(graph.cells) == 3
    mutating_waves = [c.wave for c in graph.cells if c.mode == CellMode.MUTATING]
    assert len(mutating_waves) == len(set(mutating_waves))
    assert GraphPlanner().validate(graph) == []


async def test_dependency_on_a_later_cell_is_dropped_not_trusted():
    # "late" depends on "later", which does not exist yet at its position --
    # a forward reference. The hardener drops it instead of creating a cycle.
    proposal = """
    {"cells": [
      {"name": "late", "mode": "read_only", "depends_on": ["later"],
       "acceptance_criterion_indices": [0, 1]},
      {"name": "later", "mode": "mutating", "depends_on": [],
       "acceptance_criterion_indices": []}
    ]}
    """
    planner = _planner(_FakeGateway(proposal))
    graph = await planner.plan(_contract(), "run-3")
    assert len(graph.cells) == 2
    assert graph.edges == []  # forward edge dropped
    # Criterion coverage still holds ("later" covers none, "late" covers both).
    covered = {i for c in graph.cells for i in c.acceptance_criterion_indices}
    assert covered == {0, 1}


async def test_gateway_error_falls_back_to_single_cell():
    planner = _planner(_FakeGateway(error=RuntimeError("endpoint down")))
    graph = await planner.plan(_contract(), "run-4")
    assert len(graph.cells) == 1
    assert graph.cells[0].mode == CellMode.MUTATING


async def test_unparseable_proposal_falls_back_to_single_cell():
    planner = _planner(_FakeGateway("I could not decompose this, sorry."))
    graph = await planner.plan(_contract(), "run-5")
    assert len(graph.cells) == 1


async def test_uncovered_criterion_discards_proposal():
    proposal = """
    {"cells": [
      {"name": "a", "mode": "mutating", "depends_on": [], "acceptance_criterion_indices": [0]},
      {"name": "b", "mode": "read_only", "depends_on": ["a"], "acceptance_criterion_indices": [0]}
    ]}
    """
    planner = _planner(_FakeGateway(proposal))
    graph = await planner.plan(_contract(), "run-6")  # criterion 1 uncovered
    assert len(graph.cells) == 1


async def test_single_cell_proposal_is_not_a_decomposition():
    proposal = (
        '{"cells": [{"name": "only", "mode": "mutating", "depends_on": [], '
        '"acceptance_criterion_indices": [0, 1]}]}'
    )
    planner = _planner(_FakeGateway(proposal))
    graph = await planner.plan(_contract(), "run-7")
    assert len(graph.cells) == 1  # canonical fallback, not the proposal


async def test_out_of_range_criterion_indices_are_ignored():
    proposal = """
    {"cells": [
      {"name": "a", "mode": "mutating", "depends_on": [],
       "acceptance_criterion_indices": [0, 7, -1]},
      {"name": "b", "mode": "read_only", "depends_on": ["a"],
       "acceptance_criterion_indices": [1]}
    ]}
    """
    planner = _planner(_FakeGateway(proposal))
    graph = await planner.plan(_contract(), "run-8")
    assert len(graph.cells) == 2
    covered = {i for c in graph.cells for i in c.acceptance_criterion_indices}
    assert covered == {0, 1}


async def test_request_goes_to_the_planner_profile_with_schema():
    gw = _FakeGateway(_GOOD_PROPOSAL)
    await _planner(gw).plan(_contract(), "run-9")
    request = gw.requests[0]
    assert request.profile_name == "planner-mid"
    assert request.temperature == 0.0
    assert request.output_json_schema is not None
    body = request.messages[0].content
    assert "ship the widget" in body
    assert "widget exists" in body  # criteria with indices in the prompt


async def test_wave_explosion_discards_proposal():
    # More mutually-independent mutating cells than MAX_WAVES allows.
    cells = ", ".join(
        f'{{"name": "m{i}", "mode": "mutating", "depends_on": [], '
        f'"acceptance_criterion_indices": [{i % 2}]}}'
        for i in range(MAX_WAVES + 2)
    )
    proposal = '{"cells": [' + cells + "]}"
    planner = _planner(_FakeGateway(proposal))
    graph = await planner.plan(_contract(), "run-10")
    assert len(graph.cells) == 1  # fallback: cannot separate legally
