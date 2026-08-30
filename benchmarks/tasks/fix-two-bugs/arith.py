"""Arithmetic helpers with two deliberate bugs."""


def mul(a: int, b: int) -> int:
    """Return a times b."""
    return a + b  # BUG: adds instead of multiplying


def div(a: int, b: int) -> float:
    """Return a divided by b."""
    return a * b  # BUG: multiplies instead of dividing


def add(a: int, b: int) -> int:
    """Return a plus b (correct)."""
    return a + b
