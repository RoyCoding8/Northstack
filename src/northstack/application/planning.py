"""Graph planning: synthesize and validate a GraphVersion from a contract."""

from __future__ import annotations

from northstack.config import Role
from northstack.domain.budget import Budget
from northstack.domain.contract import WorkContract
from northstack.domain.graph import CellMode, CellStatus, GraphCell, GraphVersion

MAX_TOTAL_TOKEN_BUDGET = 10_000_000
MAX_TOTAL_COST_USD = 100.0


def single_cell_graph(cell: GraphCell) -> GraphVersion:
    """Wrap one cell into the canonical single-cell GraphVersion."""
    return GraphVersion(version=1, cells=[cell], edges=[], milestones=[cell.id], current_horizon=0)


class GraphPlanner:
    """Synthesizes a GraphVersion from a WorkContract."""

    def __init__(self, role_map: dict[Role, list[str]] | None = None) -> None:
        self._role_map = role_map or {}

    async def plan(self, contract: WorkContract, run_id: str) -> GraphVersion:
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

    def validate(
        self,
        graph: GraphVersion,
        *,
        run_budget: Budget | None = None,
        max_waves: int | None = None,
    ) -> list[str]:
        return [
            *self._structural_errors(graph),
            *self._budget_errors(graph, run_budget),
            *self._concurrency_errors(graph),
            *self._wave_budget_errors(graph, max_waves),
        ]

    @staticmethod
    def _structural_errors(graph: GraphVersion) -> list[str]:
        errors: list[str] = []
        cells_by_id = {cell.id: cell for cell in graph.cells}
        cell_ids = set(cells_by_id)
        if not graph.cells:
            errors.append("graph contains no cells")
        if len(cell_ids) != len(graph.cells):
            errors.append("duplicate cell IDs in graph")
        if len(set(graph.milestones)) != len(graph.milestones):
            errors.append("duplicate milestones in graph")
        errors.extend(
            f"unknown milestone '{milestone}'"
            for milestone in sorted(set(graph.milestones) - cell_ids)
        )
        max_wave = max((cell.wave for cell in graph.cells), default=0)
        if graph.cells and graph.current_horizon != max_wave:
            errors.append(
                f"current horizon {graph.current_horizon} does not match maximum cell wave "
                f"{max_wave}"
            )

        adj: dict[str, list[str]] = {c.id: [] for c in graph.cells}
        in_degree: dict[str, int] = {c.id: 0 for c in graph.cells}
        for cell in graph.cells:
            if len(set(cell.dependencies)) != len(cell.dependencies):
                errors.append(f"cell '{cell.id}' has duplicate dependencies")
            indices = cell.acceptance_criterion_indices
            if len(set(indices)) != len(indices):
                errors.append(f"cell '{cell.id}' has duplicate criterion indices")
            invalid = sorted(
                {
                    index
                    for index in indices
                    if not 0 <= index < len(cell.contract.acceptance_criteria)
                }
            )
            if invalid:
                errors.append(f"cell '{cell.id}' has invalid criterion indices: {invalid}")
            for dependency in cell.dependencies:
                if dependency == cell.id:
                    errors.append(f"cell '{cell.id}' depends on itself")
                if dependency not in cell_ids:
                    errors.append(f"cell '{cell.id}' depends on unknown cell '{dependency}'")
                    continue
                if cells_by_id[dependency].wave >= cell.wave:
                    errors.append(
                        f"dependency '{dependency}' of cell '{cell.id}' must be in an earlier wave"
                    )
                adj[dependency].append(cell.id)
                in_degree[cell.id] += 1

        dependency_pairs = {
            (dependency, cell.id)
            for cell in graph.cells
            for dependency in cell.dependencies
            if dependency in cell_ids
        }
        edge_keys = [(edge.from_id, edge.to_id, edge.kind) for edge in graph.edges]
        if len(set(edge_keys)) != len(edge_keys):
            errors.append("duplicate edges in graph")
        for edge in graph.edges:
            if edge.from_id not in cell_ids or edge.to_id not in cell_ids:
                errors.append(f"edge {edge.from_id!r}->{edge.to_id!r} has unknown edge endpoint")
        blocking_pairs = {
            (edge.from_id, edge.to_id) for edge in graph.edges if edge.kind == "blocks"
        }
        if blocking_pairs != dependency_pairs:
            errors.append("blocking edges disagree with cell dependencies")

        queue = [cid for cid, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited != len(graph.cells):
            errors.append("cycle detected in graph")
        return errors

    @staticmethod
    def _budget_errors(graph: GraphVersion, run_budget: Budget | None) -> list[str]:
        axes = (
            ("token", "token_limit", MAX_TOTAL_TOKEN_BUDGET, ""),
            ("cost", "cost_limit_usd", MAX_TOTAL_COST_USD, "$"),
        )
        errors: list[str] = []
        for name, field, cap, unit in axes:
            authority = getattr(run_budget, field, None)
            limits = [
                value
                for c in graph.cells
                if (value := getattr(c.contract.budget, field)) is not None
            ]
            if len(limits) != len(graph.cells):
                if authority is not None:
                    errors.append(f"graph contains an unlimited {name} budget")
            elif sum(limits) > cap:
                errors.append(
                    f"total {name} budget {unit}{sum(limits):g} exceeds {unit}{cap:g} limit"
                )
        return errors

    @staticmethod
    def _concurrency_errors(graph: GraphVersion) -> list[str]:
        wave_mutating: dict[int, list[str]] = {}
        for c in graph.cells:
            if c.mode == CellMode.MUTATING:
                wave_mutating.setdefault(c.wave, []).append(c.id)
        return [
            f"wave {wave} has {len(cells)} mutating cells: {cells}"
            for wave, cells in wave_mutating.items()
            if len(cells) > 1
        ]

    @staticmethod
    def _wave_budget_errors(graph: GraphVersion, max_waves: int | None) -> list[str]:
        if max_waves is None:
            return []
        mutating = [c.id for c in graph.cells if c.mode == CellMode.MUTATING]
        if len(mutating) > max_waves:
            return [
                f"graph serializes {len(mutating)} mutating cells "
                f"({mutating[:3]}...) but max_waves={max_waves} allows {max_waves}"
            ]
        return []
