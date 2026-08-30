"""Regression lock: single-cell GraphVersion construction is centralized.

Both GraphPlanner.plan and Company.run_async wrap a single GraphCell into a
GraphVersion(version=1, cells=[cell], edges=[], milestones=[cell.id],
current_horizon=0). This test guards the dedup against silent regression.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from northstack.application.planning import single_cell_graph as _single_cell_graph
from northstack.domain import (
    Budget,
    CellMode,
    CellStatus,
    CommandCriterion,
    GraphCell,
    GraphVersion,
    WorkContract,
)

SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "northstack" / "application" / "orchestrator.py"
)


@pytest.fixture
def sample_contract() -> WorkContract:
    return WorkContract(
        id="wc-test-1",
        version=1,
        objective="Implement a greeting function",
        scope="functions/ directory",
        deliverables=["greeting.py", "test_greeting.py"],
        constraints=["Python 3.12+"],
        allowed_tools=["read_file", "write_file", "run_command"],
        workspace_scope="/tmp/workspace",
        budget=Budget(token_limit=10_000, cost_limit_usd=2.0),
        acceptance_criteria=[
            CommandCriterion(
                description="pytest|tests pass",
                command_name="pytest",
                exit_code=0,
            ),
        ],
    )


def _bare_single_cell_graph_versions(source: str) -> list[int]:
    """GraphVersion(...) calls whose kwargs match the single-cell shape exactly."""
    tree = ast.parse(source)
    hits: list[int] = []
    target = {"version", "cells", "edges", "milestones", "current_horizon"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "GraphVersion"):
            continue
        kwargs = {kw.arg for kw in node.keywords if kw.arg is not None}
        if kwargs != target:
            continue
        cells_kw = next(kw for kw in node.keywords if kw.arg == "cells")
        if not (isinstance(cells_kw.value, ast.List) and len(cells_kw.value.elts) == 1):
            hits.append(node.lineno)
            continue
        miles_kw = next(kw for kw in node.keywords if kw.arg == "milestones")
        if not isinstance(miles_kw.value, ast.List) or len(miles_kw.value.elts) != 1:
            hits.append(node.lineno)
            continue
        edge_kw = next(kw for kw in node.keywords if kw.arg == "edges")
        if not (isinstance(edge_kw.value, ast.List) and len(edge_kw.value.elts) == 0):
            hits.append(node.lineno)
        horiz = next(kw for kw in node.keywords if kw.arg == "current_horizon")
        if not (isinstance(horiz.value, ast.Constant) and horiz.value.value == 0):
            hits.append(node.lineno)
    return hits


def test_no_bare_single_cell_graph_version_literals() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    hits = _bare_single_cell_graph_versions(source)
    assert not hits, (
        f"orchestration.py constructs single-cell GraphVersion literals at "
        f"lines {hits}; use _single_cell_graph(cell) instead."
    )


def test_single_cell_graph_wraps_a_cell(sample_contract: WorkContract) -> None:
    cell = GraphCell(
        id="cell-x",
        name="objective",
        wave=0,
        mode=CellMode.MUTATING,
        contract=sample_contract,
        status=CellStatus.PENDING,
        acceptance_criterion_indices=list(range(len(sample_contract.acceptance_criteria))),
        required_profile_roles=["worker"],
    )
    g = _single_cell_graph(cell)
    assert isinstance(g, GraphVersion)
    assert g.version == 1
    assert g.cells == [cell]
    assert g.edges == []
    assert g.milestones == [cell.id]
    assert g.current_horizon == 0
