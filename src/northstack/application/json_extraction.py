"""Lenient JSON extraction from free-form model text.

Every model-backed role parses gateway output the same way: strip a
surrounding markdown code fence, then scan for the first complete JSON
object.  Callers keep their own fallback and failure semantics; this
module owns only extraction.
"""

from __future__ import annotations

import json
from typing import Any


def strip_code_fence(text: str) -> str:
    """Drop a surrounding markdown code fence, if one wraps the payload."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) > 1:
            stripped = "\n".join(lines[1:]).removesuffix("```").strip()
    return stripped


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    """First complete JSON object in text, or None."""
    stripped = strip_code_fence(text)
    decoder = json.JSONDecoder()
    start = stripped.find("{")
    while start >= 0:
        try:
            value, end = decoder.raw_decode(stripped, start)
        except json.JSONDecodeError:
            start = stripped.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = stripped.find("{", start + 1)
    return None
