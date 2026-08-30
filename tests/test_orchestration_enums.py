"""Regression lock: the application layer uses CellMode/CellStatus, not bare strings.

A prior pass left 10 raw "mutating"/"pending"/"completed" string literals in the
orchestrator despite the str-enums existing in the domain.  This test guards the
type-safety property against silent regression across every module that reasons
about a cell's mode or status.
"""

from __future__ import annotations

import ast
from pathlib import Path

APPLICATION = Path(__file__).resolve().parents[1] / "src" / "northstack" / "application"
ENUM_USERS = ("orchestrator.py", "scheduling.py", "planning.py", "routing.py")


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("northstack.domain")
        for alias in node.names
    }


def test_cell_enums_are_imported_where_they_are_used() -> None:
    names = set().union(*(_imported_names(APPLICATION / name) for name in ENUM_USERS))
    assert "CellMode" in names, "the application layer must import CellMode"
    assert "CellStatus" in names, "the application layer must import CellStatus"


def _string_comparisons(source: str) -> list[tuple[int, str]]:
    """Lines that compare a cell.mode/status field against a bare string literal."""
    return [
        (node.lineno, cmp.value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Attribute)
        and node.left.attr in {"mode", "status"}
        for cmp in node.comparators
        if isinstance(cmp, ast.Constant) and isinstance(cmp.value, str)
    ]


def test_no_bare_string_mode_status_comparisons() -> None:
    hits = {
        name: _string_comparisons((APPLICATION / name).read_text(encoding="utf-8"))
        for name in ENUM_USERS
    }
    offenders = {name: found for name, found in hits.items() if found}
    assert not offenders, f"cell.mode/status compared to bare strings: {offenders}"
