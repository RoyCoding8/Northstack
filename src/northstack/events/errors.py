"""Failures raised when a stored event cannot be read back faithfully."""

from __future__ import annotations


class LedgerCorruption(Exception):
    """A stored event does not match the catalog.

    Raised instead of degrading: a renamed, missing, or mistyped field must
    stop the replay and name the row, not be repaired with an invented default.
    """

    def __init__(self, seq: int, kind: str, detail: str) -> None:
        super().__init__(f"seq {seq} ({kind}): {detail}")
        self.seq = seq
        self.kind = kind
        self.detail = detail


class UnknownSchemaVersion(LedgerCorruption):
    """A stored event declares a schema version this build cannot read."""

    def __init__(self, seq: int, kind: str, version: int, maximum: int) -> None:
        super().__init__(seq, kind, f"schema_version {version} is newer than {maximum}")
        self.version = version
        self.maximum = maximum
