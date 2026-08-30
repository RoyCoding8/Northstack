"""Public-web read-only fetch with SSRF protection.

Public seam:
  - WebReader(policy, transport?, resolver?) -> fetch(url, method?) -> ToolResult
  - FetchTransport / DNSResolver protocols for dependency injection
  - RealDNSResolver for production DNS resolution
  - WebFetchError / SSRFBlocked exceptions

Design:
  - GET/HEAD only.  No caller cookies/auth headers.
  - Bounded redirects, size, time, and content types.
  - DNS/IP validation at every hop: rejects localhost, private, link-local,
    loopback, multicast, reserved, and unspecified networks.
  - Untrusted-evidence marking on all successful fetches.
  - Transport and DNS resolver are injectable for deterministic tests.

KNOWN LIMITATION -- DNS rebinding:
  This module validates DNS at each hop before making the HTTP request, but
  does NOT pin the TCP connection to the validated IP address.  Between DNS
  resolution and TCP connect, a DNS rebinding attack could change the A/AAAA
  record to point to a private IP.  httpx does not expose connection-level
  IP pinning at this layer.  The evidence metadata includes dns_pinned=False
  to document this boundary.  For full rebinding protection, a lower-level
  transport wrapper or IP-restricted proxy would be required.
"""

from __future__ import annotations

import ipaddress
import time
from typing import Any, Protocol
from urllib.parse import urljoin, urlparse

import httpx

from northstack.adapters.workspace.restricted import ToolEvidence, ToolResult, WebFetchPolicy
from northstack.domain.url_policy import is_loopback_host


def _is_private_or_blocked_ip(ip_str: str) -> bool:
    """Return True if the IP is in a blocked range (localhost, private, etc.)."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # Unparseable = blocked
    return (
        addr.is_loopback
        or addr.is_private
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


class FetchTransport(Protocol):
    """Minimal httpx-like transport protocol for dependency injection."""

    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        follow_redirects: bool = False,
        timeout: float | None = None,
    ) -> httpx.Response: ...


class DNSResolver(Protocol):
    """Minimal DNS resolver protocol for injection."""

    def resolve(self, hostname: str) -> list[str]: ...


class RealDNSResolver:
    """Default resolver using system getaddrinfo."""

    def resolve(self, hostname: str) -> list[str]:
        import socket

        try:
            infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            return list({host for info in infos if isinstance((host := info[4][0]), str)})
        except (socket.gaierror, OSError):
            return []


class WebFetchError(Exception):
    """Raised when a web fetch operation fails at the policy level."""


class SSRFBlocked(WebFetchError):
    """Raised when a URL is blocked by SSRF protection."""


class WebReader:
    """Public-web read-only fetch with SSRF protection.

    Usage:
        reader = WebReader(policy=WebFetchPolicy(max_response_bytes=1_000_000))
        result = reader.fetch("http://example.com")

    KNOWN LIMITATION -- DNS rebinding:
      DNS is validated before each hop, but the TCP connection is NOT pinned
      to the validated IP.  Evidence metadata includes dns_pinned=False.
      See module docstring for details.
    """

    _ALLOWED_METHODS = frozenset({"GET", "HEAD"})
    _ALLOWED_SCHEMES = frozenset({"http", "https"})
    _REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})

    def __init__(
        self,
        policy: WebFetchPolicy | None = None,
        transport: FetchTransport | None = None,
        resolver: DNSResolver | None = None,
    ) -> None:
        self._policy = policy or WebFetchPolicy()
        self._transport = transport or self._make_default_transport()
        self._resolver = resolver or RealDNSResolver()

    def _make_default_transport(self) -> httpx.Client:
        return httpx.Client(follow_redirects=False)

    def _validate_url(self, url: str) -> tuple[str, str, str]:
        """Parse and validate URL scheme and host.

        Returns (scheme, host, normalized_url).
        Rejects: non-http(s) schemes, no hostname, credentials/userinfo,
        non-default ports unless policy allows them.
        """
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        if scheme not in self._ALLOWED_SCHEMES:
            raise SSRFBlocked(
                f"Scheme '{scheme}' not allowed; only {self._ALLOWED_SCHEMES} permitted"
            )
        host = parsed.hostname
        if not host:
            raise SSRFBlocked("No hostname in URL")
        if parsed.username or parsed.password:
            raise SSRFBlocked("URL contains credentials (userinfo); not allowed")
        if parsed.port is not None:
            default_ports = {"http": 80, "https": 443}
            if parsed.port != default_ports.get(scheme):
                raise SSRFBlocked(
                    f"Port {parsed.port} is not a default port for {scheme}; not allowed"
                )
        return scheme, host, url

    def _validate_ip(self, hostname: str) -> None:
        """Resolve hostname and check IP is not in a blocked range.

        Raises SSRFBlocked on DNS failure, empty answer, or blocked IP.
        """
        try:
            ips = self._resolver.resolve(hostname)
        except OSError as e:
            raise SSRFBlocked(f"DNS resolution failed for {hostname}: {e}") from e
        if not ips:
            raise SSRFBlocked(f"DNS resolution returned no addresses for {hostname}")
        for ip_str in ips:
            if _is_private_or_blocked_ip(ip_str):
                raise SSRFBlocked(
                    f"IP {ip_str} for {hostname} is in a blocked range "
                    f"(localhost/private/link-local/multicast/reserved/unspecified)"
                )

    def _check_content_type(self, headers: dict[str, str]) -> bool:
        """Check if response content type is in the allowlist."""
        allowed = self._policy.allowed_content_types
        if not allowed:
            return True
        ct = headers.get("content-type", "").split(";")[0].strip().lower()
        return ct in allowed

    def _resolve_redirect(self, location: str, current_url: str) -> str:
        """Resolve a redirect Location header against the current URL.

        Handles: absolute URLs, scheme-relative (//host), and path-relative.
        Uses urljoin for proper relative resolution.
        """
        if location.startswith("//"):
            parsed_current = urlparse(current_url)
            location = f"{parsed_current.scheme}:{location}"
        elif not urlparse(location).scheme:
            location = urljoin(current_url, location)
        return location

    def fetch(self, url: str, method: str = "GET") -> ToolResult:
        """Fetch a URL with SSRF protection.

        Returns a ToolResult with the response body in data and
        untrusted-evidence marking in evidence.

        Conservative policy:
          - GET and HEAD only
          - 2xx responses are success; all other codes are structured failure
          - HEAD responses return no body data
          - dns_pinned=False in evidence (see module docstring)
        """
        start = time.perf_counter()

        def result(ok: bool, error: str = "", **fields: Any) -> ToolResult:
            return ToolResult(
                ok=ok,
                operation="web_fetch",
                error=error,
                duration_ms=int((time.perf_counter() - start) * 1000),
                **fields,
            )

        if method not in self._ALLOWED_METHODS:
            return result(False, f"Method '{method}' not allowed; only GET/HEAD permitted")

        try:
            original_scheme, host, normalized_url = self._validate_url(url)
        except SSRFBlocked as e:
            return result(False, str(e))

        try:
            self._validate_ip(host)
        except SSRFBlocked as e:
            return result(False, str(e))

        max_redirects = self._policy.max_redirects
        current_url = normalized_url
        hops = 0

        while True:
            parsed = urlparse(current_url)
            hop_host = parsed.hostname or ""
            try:
                self._validate_ip(hop_host)
            except SSRFBlocked as e:
                return result(False, f"SSRF blocked at redirect hop {hops}: {e}")

            try:
                self._validate_url(current_url)
            except SSRFBlocked as e:
                return result(False, f"Invalid redirect URL at hop {hops}: {e}")

            try:
                response = self._transport.request(
                    method=method,
                    url=current_url,
                    headers={"User-Agent": "northstack-webreader/1.0"},
                    follow_redirects=False,
                    timeout=self._policy.timeout_seconds,
                )
            except (httpx.TimeoutException, httpx.RequestError, OSError) as e:
                return result(False, f"Request failed: {e}")

            if response.status_code in self._REDIRECT_STATUS:
                hops += 1
                if hops > max_redirects:
                    return result(False, f"Too many redirects (max {max_redirects})")
                location = response.headers.get("location", "")
                if not location:
                    return result(False, "Redirect with no Location header")
                current_url = self._resolve_redirect(location, current_url)
                hop_parsed = urlparse(current_url)
                hop_scheme = hop_parsed.scheme.lower()
                hop_host = hop_parsed.hostname or ""
                if (
                    original_scheme == "https"
                    and hop_scheme == "http"
                    and not is_loopback_host(hop_host)
                ):
                    return result(
                        False,
                        f"Refusing https->http protocol downgrade in redirect to {current_url}",
                    )
                continue

            if response.status_code < 200 or response.status_code >= 300:
                return result(
                    False,
                    f"HTTP {response.status_code} response (non-2xx treated as failure)",
                    evidence=ToolEvidence(
                        url=current_url,
                        status_code=response.status_code,
                        content_type=dict(response.headers).get("content-type", ""),
                        hops=hops,
                        untrusted=True,
                    ),
                )

            resp_headers = dict(response.headers)
            if not self._check_content_type(resp_headers):
                ct = resp_headers.get("content-type", "unknown")
                return result(False, f"Content-Type '{ct}' not in allowlist")

            if method == "HEAD":
                return result(
                    True,
                    data=b"",
                    evidence=ToolEvidence(
                        url=current_url,
                        status_code=response.status_code,
                        content_type=resp_headers.get("content-type", ""),
                        size_bytes=0,
                        hops=hops,
                        untrusted=True,
                    ),
                )

            body = b""
            total_bytes = 0
            truncated = False
            limit = self._policy.max_response_bytes
            try:
                for chunk in response.iter_bytes():
                    total_bytes += len(chunk)
                    if total_bytes <= limit:
                        body += chunk
                    else:
                        truncated = True
                        break
            except (httpx.StreamError, OSError):
                truncated = True

            return result(
                True,
                data=body,
                truncated=truncated,
                total_bytes=total_bytes,
                evidence=ToolEvidence(
                    url=current_url,
                    status_code=response.status_code,
                    content_type=resp_headers.get("content-type", ""),
                    size_bytes=total_bytes,
                    hops=hops,
                    untrusted=True,
                ),
            )
