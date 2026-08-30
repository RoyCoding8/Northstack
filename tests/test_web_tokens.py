"""Token-discipline gate for the hand-rolled MD3 frontend.

The three-layer token rule (from the ``design-system`` skill):
  primitives (raw hex, tokens.css ONLY) -> semantic (--md-sys-color-*) ->
  component (--btn-* etc).

No raw hex value may appear inside any component/panel/view CSS or HTML or JS
-- those layers must reference tokens via ``var(--…)``.  This test greps the
checked-in ``static/`` tree and fails if any ``#<hex>`` appears outside
``tokens.css``.  It is the Python-side mirror of the Bun+regex gate used
during the UI build (task #34).

Exemptions:
  - ``static/tokens.css``        -- the primitives layer; hex lives here.
  - any ``*.map`` / non-text assets are ignored.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = (
    Path(__file__).resolve().parents[1] / "src" / "northstack" / "interfaces" / "web" / "static"
)

# A CSS/JS/HTML hex color literal: #rgb, #rrggbb, #rrggbbaa.  Word-boundary on
# the left, non-hex on the right, so ``#ff7700`` matches but ``#define`` /
# ``#id`` (handled below) do not.  We anchor the right side on a non-hex char
# or end-of-string to avoid matching longer hex runs (e.g. git hashes).
HEX_RE = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})\b")

# Exempt token-layer file (the only place hex is allowed).
EXEMPT = {"tokens.css"}

# Extensions we scan for inline color literals.
SCAN_EXT = {".css", ".html", ".js", ".ts"}


def _scan() -> list[tuple[str, int, str]]:
    """Return (rel_path, line_no, matched_hex) for every offending hex."""
    offenders: list[tuple[str, int, str]] = []
    if not STATIC.is_dir():
        return offenders
    for path in sorted(STATIC.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in SCAN_EXT:
            continue
        rel = path.relative_to(STATIC).as_posix()
        if rel in EXEMPT:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for m in HEX_RE.finditer(line):
                offenders.append((rel, line_no, m.group(0)))
    return offenders


def test_no_raw_hex_outside_tokens_layer() -> None:
    """No ``#hex`` color literal outside ``static/tokens.css``."""
    offenders = _scan()
    if offenders:
        lines = "\n".join(f"  {rel}:{ln}  {hx}" for rel, ln, hx in offenders)
        pytest.fail(
            "Raw hex color literals must live only in tokens.css (the primitive "
            "layer). Found outside it:\n" + lines
        )


def test_tokens_layer_exists() -> None:
    """The primitive token layer file must exist (sanity for the static tree)."""
    assert (STATIC / "tokens.css").is_file(), f"expected tokens.css under {STATIC}"
