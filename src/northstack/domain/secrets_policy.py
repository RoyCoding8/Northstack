"""Sensitive-name and secret-value classification -- the single source of truth.

Used by both CommandProfile (workspace.py) and CommandConfig (config.py) so
secret env names are rejected wherever an allowlist is configured, and by
every store that persists free text (the ledger, long-term memory) so a
credential that reached a payload does not reach the disk.
"""

from __future__ import annotations

import re

_SENSITIVE_KEYWORDS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "AUTH",
    "CREDENTIAL",
    "COOKIE",
    "SIGNATURE",
)

_SENSITIVE_EXACT_NAMES = frozenset({"sig"})
_SECRET_FIELDS = frozenset(
    {
        "api_key",
        "apikey",
        "secret",
        "password",
        "passwd",
        "pass",
        "token",
        "authorization",
        "auth",
        "private_key",
        "credentials",
        "credential",
        "cookie",
        "signature",
        "sig",
    }
)
_SECRET_SUFFIXES = (
    "_secret",
    "_password",
    "_passwd",
    "_passphrase",
    "_credential",
    "_credentials",
    "_cookie",
    "_signature",
    "_token",
    "_api_key",
    "_private_key",
    "_secret_key",
    "_access_key",
)

_KEYWORD_ALT = "|".join(_SENSITIVE_KEYWORDS)

_SENSITIVE_PATTERNS = re.compile(
    rf"(?:(?:^|_)(?:{_KEYWORD_ALT})S?(?:_|$))|(?:[A-Za-z]+(?:{_KEYWORD_ALT})S?$)",
    re.IGNORECASE,
)


_SECRET_VALUE_PATTERNS = (
    re.compile(r"Bearer\s+", re.IGNORECASE),
    re.compile(r"sk-[a-zA-Z0-9]"),  # OpenAI-style
    re.compile(r"ghp_[a-zA-Z0-9]"),  # GitHub PAT
    re.compile(r"gho_[a-zA-Z0-9]"),  # GitHub OAuth
)


def looks_like_secret_value(value: str) -> bool:
    """True if a *value* (not a name) carries a recognisable credential shape."""
    return any(p.search(value) for p in _SECRET_VALUE_PATTERNS)


def is_sensitive_name(name: str) -> bool:
    """True if a field, header, query, or env-var name looks sensitive."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return (
        normalized.casefold() in _SENSITIVE_EXACT_NAMES
        or _SENSITIVE_PATTERNS.search(normalized) is not None
    )


def is_secret_field_name(name: str) -> bool:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_").casefold()
    return normalized in _SECRET_FIELDS or normalized.endswith(_SECRET_SUFFIXES)


def is_sensitive_env_name(name: str) -> bool:
    """True if an env var name looks sensitive. Fail-safe: over-reject."""
    return is_sensitive_name(name)


def validate_env_allowlist(names: list[str]) -> list[str]:
    """Reject any sensitive env name in an allowlist; returns names on success."""
    keywords = "/".join(_SENSITIVE_KEYWORDS)
    for name in names:
        if is_sensitive_env_name(name):
            raise ValueError(
                f"Env var '{name}' matches a sensitive pattern "
                f"({keywords}) and is not allowed in env_allowlist"
            )
    return names
