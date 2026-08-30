"""Shape area helpers with duplicated logic to extract."""


def rectangle_area(width: float, height: float) -> float:
    """Area of a rectangle; negative inputs are invalid."""
    if width < 0 or height < 0:
        raise ValueError("dimensions must be non-negative")
    return width * height


def triangle_area(base: float, height: float) -> float:
    """Area of a triangle; negative inputs are invalid."""
    if base < 0 or height < 0:
        raise ValueError("dimensions must be non-negative")
    return base * height / 2
