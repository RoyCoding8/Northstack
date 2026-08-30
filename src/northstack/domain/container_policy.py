"""Validation for container image references used as Docker CLI data."""

from __future__ import annotations


def validate_docker_image(image: str) -> str:
    if not image.strip():
        raise ValueError("docker isolation requires docker_image: non-empty image")
    if image != image.strip() or image.startswith("-") or any(ord(char) <= 32 for char in image):
        raise ValueError("docker image must be a single non-option reference without whitespace")
    return image
