"""Integrity gate for the shipped live-pilot task suite.

Pins, for every task in benchmarks/live-pilot.json:

  - the suite loads through the benchmark loader and templates exist;
  - the PRISTINE template never fully passes its own hidden checks (a task
    that starts solved measures nothing);
  - the intended solution reaches score 1.0 (the checks are achievable);
  - snapshot copies are isolated from the immutable templates.

If a template or check drifts and breaks one of these properties, this test
fails before any live benchmark burns budget on a broken task.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from northstack.application.benchmark import load_task_suite
from northstack.application.benchmark_live import ScoringGate, snapshot_workspace

_SUITE = Path(__file__).parent.parent / "benchmarks" / "live-pilot.json"
_BASE = _SUITE.parent

# Expected pristine scores: every task must be partially or fully unsolved.
_PRISTINE_EXPECTED = {
    "fix-failing-test": 0.5,
    "add-feature": 0.0,
    "safe-refactor": 0.5,
    "docs-config": 0.0,
    "fix-two-bugs": 0.5,
    "add-validation": 0.0,
    "extract-helper": 0.5,
    "write-config": 0.0,
}


def _solve(task_id: str, ws: Path) -> None:
    """Apply the intended minimal solution inside a snapshot copy."""
    if task_id == "fix-failing-test":
        path = ws / "calc.py"
        path.write_text(
            path.read_text().replace(
                "return a - b  # BUG: subtracts instead of adding", "return a + b"
            )
        )
    elif task_id == "add-feature":
        path = ws / "greeter.py"
        path.write_text(
            path.read_text().replace(
                '        return f"Hello, {name}!"',
                '        return f"Hello, {name}!"\n\n'
                "    def farewell(self, name: str) -> str:\n"
                '        """Return a friendly farewell."""\n'
                '        return f"Goodbye, {name}!"',
            )
        )
    elif task_id == "safe-refactor":
        path = ws / "mathops.py"
        path.write_text(path.read_text().replace("_combine", "_merge"))
    elif task_id == "docs-config":
        (ws / "CONTRIBUTING.md").write_text(
            "# Contributing\n\n## How to contribute\n\n1. Fork the repo\n2. Open a PR\n"
        )
    elif task_id == "fix-two-bugs":
        path = ws / "arith.py"
        text = path.read_text()
        text = text.replace("return a + b  # BUG: adds instead of multiplying", "return a * b")
        path.write_text(
            text.replace("return a * b  # BUG: multiplies instead of dividing", "return a / b")
        )
    elif task_id == "add-validation":
        path = ws / "points.py"
        path.write_text(
            path.read_text().replace(
                '    """Euclidean distance between two points."""\n    return',
                '    """Euclidean distance between two points."""\n'
                "    for v in (x1, y1, x2, y2):\n"
                "        if isinstance(v, bool) or not isinstance(v, (int, float)):\n"
                '            raise ValueError(f"not numeric: {v!r}")\n'
                "    return",
            )
        )
    elif task_id == "extract-helper":
        (ws / "shapes.py").write_text(
            '"""Shape area helpers with a single shared validation helper."""\n\n\n'
            "def _require_non_negative(*dimensions: float) -> None:\n"
            '    """Reject any negative dimension."""\n'
            "    if any(d < 0 for d in dimensions):\n"
            '        raise ValueError("dimensions must be non-negative")\n\n\n'
            "def rectangle_area(width: float, height: float) -> float:\n"
            '    """Area of a rectangle."""\n'
            "    _require_non_negative(width, height)\n"
            "    return width * height\n\n\n"
            "def triangle_area(base: float, height: float) -> float:\n"
            '    """Area of a triangle."""\n'
            "    _require_non_negative(base, height)\n"
            "    return base * height / 2\n"
        )
    elif task_id == "write-config":
        (ws / "settings.json").write_text(
            '{"host": "127.0.0.1", "port": 8080, "debug": false, "features": ["metrics"]}\n'
        )
    else:  # pragma: no cover - new tasks must extend this table
        raise AssertionError(f"no _solve mapping for task {task_id!r}")


@pytest.fixture(scope="module")
def suite():
    loaded = load_task_suite(_SUITE)
    assert loaded.tasks, "live-pilot suite must contain tasks"
    return loaded


def test_suite_tasks_have_templates(suite):
    for task in suite.tasks:
        template = _BASE / task.workspace
        assert template.is_dir(), f"missing template for {task.id}: {template}"
        assert task.checks, f"task {task.id} has no hidden checks"


async def test_pristine_templates_are_not_solved(suite):
    for task, expected in _PRISTINE_EXPECTED.items():
        found = next(t for t in suite.tasks if t.id == task)
        outcome = await ScoringGate(found.checks).score(_BASE / found.workspace)
        assert outcome.score == pytest.approx(expected), (
            f"{task}: pristine score drifted to {outcome.score} (failures={outcome.failures})"
        )


async def test_solved_snapshots_score_perfect(suite, tmp_path: Path):
    try:
        for task in suite.tasks:
            snap = snapshot_workspace(_BASE / task.workspace, tmp_path / task.id / "ws")
            _solve(task.id, snap)
            outcome = await ScoringGate(task.checks).score(snap)
            assert outcome.score == 1.0, (
                f"{task.id}: intended solution does not satisfy the checks "
                f"(failures={outcome.failures})"
            )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


async def test_solving_a_snapshot_never_touches_the_template(suite, tmp_path: Path):
    from northstack.application.benchmark_live import tree_digest

    task = suite.tasks[0]
    before = tree_digest(_BASE / task.workspace)
    snap = snapshot_workspace(_BASE / task.workspace, tmp_path / "iso" / "ws")
    _solve(task.id, snap)
    await ScoringGate(task.checks).score(snap)
    assert tree_digest(_BASE / task.workspace) == before
