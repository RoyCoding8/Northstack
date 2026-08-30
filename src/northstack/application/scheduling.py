"""Ready-cell scheduling: parallel read-only cells, at most one mutating."""

from __future__ import annotations

from collections.abc import Iterable

from northstack.domain.graph import CellMode, CellStatus, GraphCell, GraphVersion


class Scheduler:
    """Computes ready cells without mutating the immutable graph model."""

    def ready_cells(
        self,
        graph: GraphVersion,
        completed_ids: Iterable[str] | None = None,
        failed_ids: Iterable[str] | None = None,
    ) -> list[GraphCell]:
        """Cells whose dependencies are met, capped at one mutating cell.

        ``completed_ids``/``failed_ids`` carry progress the orchestrator holds
        outside the frozen graph.  When omitted, cell status on the graph is
        authoritative -- which is what a replayed graph gives you.
        """
        done = set(completed_ids) if completed_ids is not None else set()
        failed = set(failed_ids) if failed_ids is not None else set()
        track_externally = completed_ids is not None or failed_ids is not None
        if not track_externally:
            done = {c.id for c in graph.cells if c.status == CellStatus.COMPLETED}

        ready: list[GraphCell] = []
        has_mutating = False
        for cell in graph.cells:
            if cell.id in done or cell.id in failed:
                continue
            if not track_externally and cell.status != CellStatus.PENDING:
                continue
            if not all(dep in done for dep in cell.dependencies):
                continue
            if cell.mode == CellMode.MUTATING:
                if has_mutating:
                    continue
                has_mutating = True
            ready.append(cell)
        return ready
