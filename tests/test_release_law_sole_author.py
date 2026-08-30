"""Guard: ``RunOutcome`` is constructed only inside ``release_law.py``.

``ReleaseLaw`` is the sole authority that decides a run's outcome.  If any
other module constructs a ``RunOutcome`` value, the decision has leaked out of
the law -- this test pins that it has not.

The test scans ``src/`` for the call form ``RunOutcome(`` (constructing the enum
from a value) and asserts every hit is inside ``application/release_law.py``.
Member access -- ``RunOutcome.VERIFIED`` etc. -- is allowed everywhere: routing,
status mapping and event payloads legitimately reference the enum members; only
*constructing* an outcome from a value is reserved to the law.  The class
definition line in ``domain/outcome.py`` is also allowed.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
_CALL_RE = re.compile(r"\bRunOutcome\s*\(")
# Files where ``RunOutcome(`` is permitted: the law itself, and the enum's own
# definition (the regex does not match a class statement, but listing it makes
# the intent explicit if the definition ever grows a callable alias).
_ALLOWED_FILES = {
    SRC / "northstack" / "application" / "release_law.py",
    SRC / "northstack" / "domain" / "outcome.py",
}


def _scan() -> list[tuple[Path, int, str]]:
    """Return (file, line_no, line) for every ``RunOutcome(`` call in src/."""
    hits: list[tuple[Path, int, str]] = []
    for path in SRC.rglob("*.py"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _CALL_RE.search(line):
                hits.append((path, n, line.strip()))
    return hits


def test_run_outcome_is_constructed_only_in_release_law():
    hits = _scan()
    offenders = [
        f"{hit[0].relative_to(SRC)}:{hit[1]}  {hit[2]}"
        for hit in hits
        if hit[0].resolve() not in {p.resolve() for p in _ALLOWED_FILES}
    ]
    assert not offenders, (
        "RunOutcome must be constructed only inside release_law.py, but these "
        "sites construct it directly:\n  " + "\n  ".join(offenders)
    )


def test_outcome_determiner_is_gone():
    """The dead OutcomeDeterminer class must not survive in the tree."""
    determiner = SRC / "northstack" / "application" / "verification" / "outcome.py"
    if determiner.exists():
        text = determiner.read_text(encoding="utf-8")
        assert "class OutcomeDeterminer" not in text, (
            "OutcomeDeterminer still exists; ReleaseLaw is the sole outcome authority"
        )


def test_no_inline_outcome_branch_in_orchestrator():
    """The orchestrator must not carry the old hard/soft/budget if/elif branch."""
    orch = SRC / "northstack" / "application" / "orchestrator.py"
    text = orch.read_text(encoding="utf-8")
    # The deleted branch assigned final_outcome from an if/elif on hard fail
    # vs material_disagreement.  The outcome now comes from ReleaseLaw.
    assert "final_outcome = RunOutcome.FAILED" not in text
    assert "final_outcome = RunOutcome.ABSTAINED" not in text
    assert "final_outcome = RunOutcome.VERIFIED" not in text
