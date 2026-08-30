"""Tiny calculator module with one deliberate bug."""


def add(a: int, b: int) -> int:
    """Return the sum of a and b."""
    return a - b  # BUG: subtracts instead of adding


def sub(a: int, b: int) -> int:
    """Return a minus b."""
    return a - b
