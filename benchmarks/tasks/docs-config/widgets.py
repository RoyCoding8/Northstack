"""Widget utilities."""


def widget_id(name: str) -> str:
    """Return the canonical widget identifier."""
    return f"widget-{name.lower()}"
