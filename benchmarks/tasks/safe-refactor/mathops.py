"""Math helpers whose internal helper should be renamed."""


def _combine(values, start):
    """Fold values left-to-right starting from start."""
    total = start
    for v in values:
        total = total + v
    return total


def total(values):
    """Sum a list of numbers."""
    return _combine(values, 0)


def concat(parts):
    """Concatenate strings."""
    return _combine(parts, "")
