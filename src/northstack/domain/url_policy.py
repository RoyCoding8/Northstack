"""Security policy for configured model-provider endpoints."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse


def is_loopback_host(hostname: str | None) -> bool:
    """Return whether a hostname is explicitly local loopback."""
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def validate_provider_url(
    url: str,
    *,
    credentialed: bool,
    allow_insecure_http: bool = False,
) -> None:
    """Validate syntax and transport security for a provider base URL."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("base_url must use http or https")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not contain credentials")
    if parsed.query:
        raise ValueError("base_url must not contain a query string")
    if parsed.fragment:
        raise ValueError("base_url must not contain a fragment")
    if not parsed.hostname:
        raise ValueError("base_url must include a hostname")
    if (
        credentialed
        and parsed.scheme == "http"
        and not is_loopback_host(parsed.hostname)
        and not allow_insecure_http
    ):
        raise ValueError(
            "credentialed non-loopback base_url must use https; "
            "set allow_insecure_http=true only for a trusted proxy"
        )
