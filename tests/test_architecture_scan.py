"""Architectural scan tests: single-ownership and forbidden-import guards.

These properties must hold and be *proven by a scan test* (not just asserted
in passing):

1. ``contract.budget.max_retries`` is read in exactly one module. The per-cell
   retry cap has a single owner (``application/cell_runner.py``); no other
   module reads the attribute.  ``RecoveryManager`` keeps docstrings that
   *mention* ``max_retries`` (to document that it does NOT own the cap) --
   those are prose, not reads, so the scan keys on attribute access
   (``.max_retries``) with a word boundary, which matches a real read and not
   the identifier inside a sentence.

2. No blocking ``time.sleep(`` remains anywhere under ``src/northstack/`` that
   belongs to the retry path. Scope is *retry only*: ``interfaces/`` (the TUI
   poll loop) is excluded by decision -- its ``time.sleep`` is a blocking poll,
   not a retry backoff, and is out of the plan's retry-centralisation concern.
   ``asyncio.sleep`` is fine (it yields the loop); only the blocking
   ``time.sleep`` is forbidden here.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "northstack"


def _py_files(root: Path, *, exclude_dirs: tuple[str, ...] = ()) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*.py")
        if not any(part in exclude_dirs for part in p.relative_to(root).parts)
    )


# Guard 1: contract.budget.max_retries read in exactly one module


def _attribute_reads(file: Path, attribute: str) -> list[str]:
    """Lines that read ``.<attribute>`` as a *runtime* attribute access.

    Uses the AST so docstring/prose mentions of ``Budget.max_retries`` (which
    ``RecoveryManager`` keeps to document that it does NOT own the cap) are
    not mistaken for a real read -- strings are never ``ast.Attribute`` nodes.
    """
    tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == attribute:
            hits.append(f"{file.name}:{node.lineno}: {attribute}")
    return hits


class TestMaxRetriesSingleOwner:
    """``contract.budget.max_retries`` is read in exactly one module."""

    def test_max_retries_read_in_exactly_one_module(self) -> None:
        files = _py_files(_SRC_ROOT)
        assert files, "scan root produced no python files -- check _SRC_ROOT"

        readers: list[str] = []
        for path in files:
            hits = _attribute_reads(path, "max_retries")
            if hits:
                readers.extend(hits)

        # The single owner is the per-cell attempt loop.  A single source
        # line may read the attribute more than once (e.g. a cap guard
        # `a.max_retries != 0 and n >= a.max_retries`), so the invariant is
        # about the *module*, not the hit count: every read must be in the
        # owner module, and the owner must read it at least once.
        owner = "cell_runner.py"
        non_owner = [h for h in readers if not h.startswith(owner)]

        assert readers, "no module reads contract.budget.max_retries at all"
        assert non_owner == [], (
            "contract.budget.max_retries must be read in exactly one module "
            f"({owner}); found reads outside it:\n" + "\n".join(non_owner)
        )
        assert any(h.startswith(owner) for h in readers), (
            f"{owner} must be the single reader of contract.budget.max_retries"
        )


# Guard 3: no blocking time.sleep in the retry path (interfaces excluded)


class TestNoBlockingSleepInRetryPath:
    """No ``time.sleep(`` anywhere under ``src/northstack/`` except the TUI
    poll loop (``interfaces/``), which is out of the retry-centralisation
    scope by decision. ``asyncio.sleep`` is permitted (it yields)."""

    EXCLUDE_DIRS = ("interfaces",)

    def test_no_blocking_time_sleep_outside_interfaces(self) -> None:
        files = _py_files(_SRC_ROOT, exclude_dirs=self.EXCLUDE_DIRS)
        assert files, "scan root produced no python files -- check _SRC_ROOT"

        offenders: list[str] = []
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    # time.sleep(...) -- an attribute call on a ``time`` name.
                    and node.func.attr == "sleep"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "time"
                ):
                    offenders.append(f"{path.relative_to(_SRC_ROOT)}:{node.lineno}: time.sleep(")

        assert not offenders, (
            "blocking time.sleep() found in the retry path (src/northstack "
            "minus interfaces/) -- retry backoff must use asyncio.sleep:\n" + "\n".join(offenders)
        )
