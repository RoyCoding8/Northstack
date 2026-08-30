"""RestrictedWorkspace: mediated filesystem operations with security boundaries.

Public seam:
  - RestrictedWorkspace(root, config, ledger?) -> mediated fs ops
  - ToolResult / ToolEvidence -> typed return types
  - WorkspaceConfig / CommandProfile / WebFetchPolicy -> config schemas

The workspace is a deep module: callers learn a small interface, and all
path resolution, lease management, content limits, encoding policy, and
security checks are hidden inside.

NOTE: This is a restricted-execution module, not a security sandbox.
Path containment is enforced via os.path.commonpath (never prefix string
comparison), but a determined attacker with process-level access may
circumvent these checks.  The module reduces the attack surface for
AI-worker subprocesses; it does not provide a provably secure boundary.

Security invariants (checked at every existing ancestor including root):
  - Workspace-relative paths only.
  - No absolute paths, '..' escapes, symlinks, junctions,
    or Windows reparse points.
  - Containment verified via os.path.commonpath, never prefix string comparison.
"""

from __future__ import annotations

import builtins
import heapq
import hashlib
import json
import os
import secrets
import stat
import threading
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from northstack.adapters.atomic_io import atomic_write_bytes

from northstack.adapters.workspace.commands import (
    docker_available,
    run_command,
    wrap_docker_argv,
)
from northstack.config import CommandConfig
from northstack.domain.container_policy import validate_docker_image
from northstack.domain.secrets_policy import validate_env_allowlist

_SENSITIVE_FILENAMES = frozenset(
    {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", ".htpasswd", "credentials.json"}
)
_SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx")


def is_sensitive_path(path: Path | str) -> bool:
    """True when a path names a secret, key, or ledger -- by its parts, not alias."""
    parts = [part.lower() for part in Path(path).parts]
    name = parts[-1] if parts else ""
    if ".git" in parts or ".ssh" in parts or name == "ledger.db":
        return True
    if name in _SENSITIVE_FILENAMES:
        return True
    if name == ".env" or name.startswith(".env."):
        return True
    return name.endswith(_SENSITIVE_SUFFIXES)


def _denied_file_identity(path: Path) -> bool:
    try:
        return is_sensitive_path(path) or path.stat().st_nlink > 1
    except OSError:
        return True


class ToolEvidence(BaseModel):
    """Evidence metadata attached to a ToolResult for event logging."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    url: str = Field(default="", description="Source URL for web fetches")
    status_code: int = Field(default=0, description="HTTP status code for web fetches")
    content_type: str = Field(default="", description="Response content type")
    size_bytes: int = Field(default=0, description="Response body size")
    hops: int = Field(default=0, description="Number of redirect hops")
    untrusted: bool = Field(default=False, description="True if content is from public web")
    dns_pinned: bool = Field(
        default=False,
        description="True if TCP connection was pinned to DNS-resolved IP (currently always False)",
    )


class ToolResult(BaseModel):
    """Typed return from every workspace operation.

    Suitable for event logging: every field is serializable and
    self-describing. Callers use .to_event_payload() and .to_evidence()
    for ledger integration.
    """

    ok: bool = Field(description="Whether the operation succeeded")
    operation: str = Field(description="Operation name (read, write, list, etc.)")
    error: str = Field(default="", description="Error message if not ok")
    error_kind: str | None = Field(
        default=None, description="Structural failure category (e.g. sensitive_denied) if set"
    )
    data: bytes = Field(default=b"", description="Operation result data (bytes)")
    digest_before: str | None = Field(
        default=None, description="SHA-256 digest before mutation (None for reads/creates)"
    )
    digest_after: str | None = Field(
        default=None, description="SHA-256 digest after mutation (None for reads)"
    )
    truncated: bool = Field(default=False, description="True if output was truncated")
    truncation_reason: str | None = Field(
        default=None, description="Stable limit name that caused incomplete output"
    )
    total_entries: int = Field(default=0, description="Total entries before truncation")
    total_bytes: int = Field(default=0, description="Total bytes before truncation")
    duration_ms: int = Field(default=0, description="Wall-clock duration in milliseconds")
    exit_code: int | None = Field(
        default=None, description="Process exit code for command execution"
    )
    evidence: ToolEvidence | None = Field(
        default=None, description="Evidence metadata for event logging"
    )

    def to_event_payload(self) -> dict[str, Any]:
        """Return a plain-dict payload suitable for ledger event storage."""
        payload: dict[str, Any] = {
            "operation": self.operation,
            "ok": self.ok,
            "duration_ms": self.duration_ms,
        }
        if self.error:
            payload["error"] = self.error
        if self.error_kind:
            payload["error_kind"] = self.error_kind
        if self.digest_before:
            payload["digest_before"] = self.digest_before
        if self.digest_after:
            payload["digest_after"] = self.digest_after
        if self.truncated:
            payload["truncated"] = True
            if self.truncation_reason:
                payload["truncation_reason"] = self.truncation_reason
            payload["total_entries"] = self.total_entries
            payload["total_bytes"] = self.total_bytes
        if self.exit_code is not None:
            payload["exit_code"] = self.exit_code
        if self.evidence:
            payload["evidence"] = self.evidence.model_dump()
        return payload

    def to_evidence(self) -> dict[str, Any]:
        """Return evidence dict for event logging."""
        if self.evidence:
            return self.evidence.model_dump()
        return {
            "digest_after": self.digest_after,
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
        }


_BASELINE_ENV_WIN = frozenset(
    {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "PATHEXT",
        "COMSPEC",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
    }
)
_BASELINE_ENV_POSIX = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "SHELL",
    }
)
_BASELINE_ENV = _BASELINE_ENV_WIN if os.name == "nt" else _BASELINE_ENV_POSIX


class WorkspaceConfig(BaseModel):
    """Limits for workspace operations."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    max_list_entries: int = Field(default=1000, ge=1, description="Max entries returned by list()")
    max_read_bytes: int = Field(default=1_048_576, ge=0, description="Max bytes returned by read()")
    max_search_results: int = Field(default=100, ge=1, description="Max results from search()")
    max_search_file_bytes: int = Field(
        default=1_048_576, ge=1, description="Max bytes inspected per searched file"
    )
    max_search_files: int = Field(default=10_000, ge=1, description="Max files searched")
    max_search_directories: int = Field(
        default=1_000, ge=1, description="Max directories traversed by search()"
    )
    max_write_bytes: int = Field(default=1_048_576, ge=0, description="Max bytes for create/write")
    max_patch_old_bytes: int = Field(default=65536, ge=0, description="Max size of patch old text")
    lease_timeout_seconds: float = Field(
        default=30.0, ge=0.0, description="Lease TTL; 0 = no timeout"
    )


class CommandProfile(BaseModel):
    """Named command profile for subprocess execution.

    Subprocess rules (enforced, not configurable):
      - argv is executed verbatim with shell=False -- no arbitrary shell.
      - No API keys or provider secrets are inherited into the subprocess
        environment.  env_allowlist is validated to reject sensitive names.
      - Workspace cwd used by default.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    name: str = Field(
        min_length=1,
        max_length=60,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
        description="Profile name",
    )
    argv: list[str] = Field(min_length=1, description="Exact command argv (no shell)")
    timeout_seconds: float = Field(default=10.0, ge=0.0, description="Max wall-clock seconds")
    max_output_bytes: int = Field(default=65536, ge=0, description="Max combined stdout+stderr")
    env_allowlist: list[str] = Field(
        default_factory=lambda: ["PATH"],
        description="Env vars allowed in subprocess; sensitive names rejected at validation",
    )
    workspace_cwd: bool = Field(default=True, description="Whether to set cwd to workspace root")
    isolation: Literal["host", "docker"] = Field(default="host")
    docker_image: str = Field(default="", description="Required when isolation = docker")

    @classmethod
    def from_config(cls, config: CommandConfig) -> CommandProfile:
        """The single CommandConfig -> CommandProfile translation point."""
        return cls(
            name=config.name,
            argv=config.argv,
            timeout_seconds=config.timeout_seconds,
            max_output_bytes=config.max_output_bytes,
            env_allowlist=config.env_allowlist,
            isolation=config.isolation,
            docker_image=config.docker_image,
        )

    @field_validator("env_allowlist", mode="before")
    @classmethod
    def _validate_env_allowlist(cls, v: list[str]) -> list[str]:
        """Reject sensitive env var names in the allowlist."""
        return validate_env_allowlist(v)

    @model_validator(mode="after")
    def _validate_docker_isolation(self) -> CommandProfile:
        if self.isolation == "docker":
            try:
                validate_docker_image(self.docker_image)
            except ValueError as error:
                raise ValueError(f"command profile {self.name!r}: {error}") from error
        return self


class WebFetchPolicy(BaseModel):
    """Constraints for public-web read-only fetch."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    max_response_bytes: int = Field(default=1_048_576, ge=0, description="Max response body size")
    timeout_seconds: float = Field(default=10.0, ge=0.0, description="Max request time")
    max_redirects: int = Field(default=5, ge=0, description="Max redirect hops")
    allowed_content_types: list[str] = Field(
        default_factory=lambda: ["text/html", "text/plain", "application/json"],
        description="Allowed response content types",
    )


class _Lease:
    """Internal lease state."""

    __slots__ = ("acquired_at", "owner", "timeout", "token")

    def __init__(self, token: str, owner: str, timeout: float) -> None:
        self.token = token
        self.owner = owner
        self.acquired_at = time.perf_counter()
        self.timeout = timeout

    def is_expired(self) -> bool:
        if self.timeout <= 0:
            return False
        return (time.perf_counter() - self.acquired_at) > self.timeout


class RestrictedWorkspace:
    """Mediated filesystem operations with security boundaries.

    A deep module: callers learn a small interface.  All path resolution,
    lease management, content limits, encoding policy, and security checks
    are hidden inside.

    NOTE: This is a restricted-execution module, not a security sandbox.
    Path containment is enforced but may be circumvented by a determined
    attacker with process-level access.

    Constructor:
        RestrictedWorkspace(root, config, ledger=None)

    Operations:
        list(path) -> ToolResult
        search(pattern, path) -> ToolResult
        read(path) -> ToolResult
        create(path, content, lease?) -> ToolResult
        write(path, content, lease?) -> ToolResult
        replace(path, old, new, lease?) -> ToolResult
        patch(path, patches, lease?) -> ToolResult
        execute_command(profile) -> ToolResult
        acquire_lease(owner) -> str | None
        release_lease(token) -> None
    """

    _ENCODING = "utf-8"

    def __init__(
        self,
        root: Path | str,
        config: WorkspaceConfig | None = None,
        ledger: Any | None = None,
    ) -> None:
        root_path = Path(root)
        if self._is_reparse_point(root_path):
            raise ValueError(f"Workspace root is a symlink or reparse point: {root_path}")
        self._root = root_path.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._config = config or WorkspaceConfig()
        self._ledger = ledger
        self._lease: _Lease | None = None
        self._lease_lock = threading.Lock()

    @property
    def root(self) -> Path:
        """The resolved workspace root, for evidence tools that must hash trees."""
        return self._root

    @staticmethod
    def _is_reparse_point(path: Path) -> bool:
        """True when path itself is a symlink or (on Windows) any reparse point.

        lstat-based: a broken link is still seen (its target need not exist),
        an absent path is not a link (the caller may be about to create it),
        and a path that exists but cannot be inspected is treated as one --
        fail closed. Junctions are not symlinks on Windows, so the reparse
        attribute is what catches them; on POSIX lstat has no
        st_file_attributes and the getattr default makes that check a no-op.
        """
        try:
            st = path.lstat()
        except FileNotFoundError:
            return False
        except OSError:
            return True  # exists but cannot inspect -- assume unsafe
        if stat.S_ISLNK(st.st_mode):
            return True
        return bool(getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)

    def _resolve_safe(self, rel_path: str) -> Path | None:
        """Resolve a workspace-relative path safely.

        Returns the resolved absolute path if valid, None if invalid. Every
        component of the UNRESOLVED path is checked for symlinks, junctions,
        and Windows reparse points before resolve() is allowed to run --
        resolve() follows links, so checking after it can only ever see a
        link-free path.
        """
        if not rel_path:
            return None

        if rel_path == ".":
            return self._root

        p = Path(rel_path)
        if p.is_absolute():
            return None

        if len(rel_path) >= 2 and rel_path[1] == ":":
            return None

        parts = Path(rel_path).parts
        if ".." in parts or "." in parts:
            return None

        if "\\" in rel_path and os.name != "nt":
            return None

        candidate = self._root / rel_path
        current = self._root
        for part in Path(rel_path).parts:
            current = current / part
            if self._is_reparse_point(current):
                return None

        resolved = candidate.resolve()

        try:
            common = os.path.commonpath([str(self._root), str(resolved)])
            if common != str(self._root):
                return None
        except ValueError:
            return None

        return resolved

    def _atomic_write(
        self,
        target: Path,
        content: bytes,
        *,
        require_exists: bool | None = None,
        operation: str = "write",
    ) -> tuple[bool, str, str | None, str | None]:
        """Atomically write content to target.

        Uses the shared same-directory atomic writer after revalidating the
        parent and target immediately before replacement.

        require_exists:
          None  -- file may or may not exist (write/create-if-missing)
          True  -- file must exist (replace/patch)
          False -- file must NOT exist (create)

        Returns (ok, error, digest_before, digest_after).
        """
        parent = target.parent
        if not parent.is_dir():
            if require_exists is True:
                return False, f"Parent directory does not exist: {parent}", None, None
            parent.mkdir(parents=True, exist_ok=True)

        digest_before: str | None = None
        if target.exists():
            if require_exists is False:
                return False, f"File already exists: {target.name}", None, None
            try:
                old_content = target.read_bytes()
                digest_before = self._digest(old_content)
            except OSError:
                pass
        else:
            if require_exists is True:
                return False, f"File not found: {target.name}", None, None

        try:
            if not parent.is_dir():
                raise OSError(f"Parent directory disappeared: {parent}")
            atomic_write_bytes(target, content, create_parents=False)
            return True, "", digest_before, self._digest(content)
        except OSError as e:
            return False, f"{operation} failed: {e}", digest_before, None

    def list(self, rel_path: str) -> ToolResult:
        """List directory entries with bounded count."""
        start = time.perf_counter()
        resolved = self._resolve_safe(rel_path)
        if resolved is None:
            return ToolResult(
                ok=False,
                operation="list",
                error="Invalid or unsafe path",
                duration_ms=self._elapsed(start),
            )
        if not resolved.is_dir():
            return ToolResult(
                ok=False,
                operation="list",
                error=f"Not a directory: {rel_path}",
                duration_ms=self._elapsed(start),
            )
        try:
            total = 0

            def names():
                nonlocal total
                for entry in resolved.iterdir():
                    total += 1
                    yield entry.name

            entries = heapq.nsmallest(self._config.max_list_entries, names())
        except OSError as e:
            return ToolResult(
                ok=False,
                operation="list",
                error=f"List failed: {e}",
                duration_ms=self._elapsed(start),
            )
        truncated = total > self._config.max_list_entries
        return ToolResult(
            ok=True,
            operation="list",
            data=json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode(),
            truncated=truncated,
            truncation_reason="entry_limit" if truncated else None,
            total_entries=total,
            duration_ms=self._elapsed(start),
        )

    def search(self, pattern: str, path: str = ".") -> ToolResult:
        """Search for pattern in workspace files (simple substring match)."""
        start = time.perf_counter()
        resolved = self._resolve_safe(path)
        if resolved is None:
            return ToolResult(
                ok=False,
                operation="search",
                error="Invalid or unsafe path",
                duration_ms=self._elapsed(start),
            )
        if not resolved.exists():
            return ToolResult(
                ok=False,
                operation="search",
                error=f"Path not found: {path}",
                duration_ms=self._elapsed(start),
            )

        matches: list[dict[str, str]] = []
        total = files = directories = 0
        reason: str | None = None
        stack = [resolved]
        while stack:
            entry = stack.pop()
            try:
                rel = str(entry.relative_to(self._root))
            except ValueError:
                continue
            safe = self._resolve_safe(rel or ".")
            if safe is None:
                continue
            try:
                if safe.is_dir():
                    if directories >= self._config.max_search_directories:
                        reason = reason or "directory_limit"
                        break
                    directories += 1
                    stack.extend(sorted(safe.iterdir(), key=lambda item: item.name, reverse=True))
                    continue
                if not safe.is_file() or _denied_file_identity(safe):
                    continue
                if files >= self._config.max_search_files:
                    reason = reason or "file_limit"
                    break
                files += 1
                with safe.open("rb") as stream:
                    content = stream.read(self._config.max_search_file_bytes + 1)
            except OSError:
                continue
            if len(content) > self._config.max_search_file_bytes:
                content = content[: self._config.max_search_file_bytes]
                reason = reason or "file_byte_limit"
            for line_number, line in enumerate(
                content.decode(self._ENCODING, errors="replace").splitlines(), 1
            ):
                if pattern not in line:
                    continue
                total += 1
                if total > self._config.max_search_results:
                    reason = reason or "result_limit"
                    stack.clear()
                    break
                matches.append(
                    {
                        "path": rel,
                        "line": str(line_number),
                        "text": line.strip()[:200],
                    }
                )
        return ToolResult(
            ok=True,
            operation="search",
            data=json.dumps(matches, ensure_ascii=False, separators=(",", ":")).encode(),
            truncated=reason is not None,
            truncation_reason=reason,
            total_entries=total,
            duration_ms=self._elapsed(start),
        )

    def read(self, rel_path: str) -> ToolResult:
        """Read a file with byte limit."""
        start = time.perf_counter()
        resolved = self._resolve_safe(rel_path)
        if resolved is None:
            return ToolResult(
                ok=False,
                operation="read",
                error="Invalid or unsafe path",
                error_kind="unsafe_path",
                duration_ms=self._elapsed(start),
            )
        if not resolved.exists():
            return ToolResult(
                ok=False,
                operation="read",
                error=f"File not found: {rel_path}",
                error_kind="not_found",
                duration_ms=self._elapsed(start),
            )
        if not resolved.is_file():
            return ToolResult(
                ok=False,
                operation="read",
                error=f"Not a file: {rel_path}",
                error_kind="not_file",
                duration_ms=self._elapsed(start),
            )
        if _denied_file_identity(resolved):
            return ToolResult(
                ok=False,
                operation="read",
                error="sensitive file reads are denied",
                error_kind="sensitive_denied",
                duration_ms=self._elapsed(start),
            )
        try:
            with resolved.open("rb") as stream:
                content = stream.read(self._config.max_read_bytes + 1)
                total_bytes = os.fstat(stream.fileno()).st_size
        except OSError as e:
            return ToolResult(
                ok=False,
                operation="read",
                error=f"Read failed: {e}",
                error_kind="io_error",
                duration_ms=self._elapsed(start),
            )
        truncated = max(total_bytes, len(content)) > self._config.max_read_bytes
        content = content[: self._config.max_read_bytes]
        return ToolResult(
            ok=True,
            operation="read",
            data=content,
            truncated=truncated,
            truncation_reason="byte_limit" if truncated else None,
            total_bytes=total_bytes,
            duration_ms=self._elapsed(start),
        )

    def _check_lease(self, lease_token: str | None) -> str | None:
        """Validate lease token. Returns error message or None if valid."""
        if lease_token is None:
            return "Mutation requires a valid lease token"
        with self._lease_lock:
            if self._lease is None:
                return "No lease held"
            if self._lease.is_expired():
                self._lease = None
                return "Lease has expired"
            return None if self._lease.token == lease_token else "Invalid lease token"

    def _digest(self, content: bytes) -> str:
        return "sha256:" + hashlib.sha256(content).hexdigest()

    def _mutate_file(
        self,
        rel_path: str,
        content: bytes,
        operation: str,
        require_exists: bool | None,
        lease: str | None,
    ) -> ToolResult:
        """Shared create/write path: lease, size-limit, path-validate, atomic write."""
        start = time.perf_counter()
        err = self._check_lease(lease)
        if err:
            return ToolResult(
                ok=False, operation=operation, error=err, duration_ms=self._elapsed(start)
            )
        if len(content) > self._config.max_write_bytes:
            return ToolResult(
                ok=False,
                operation=operation,
                error=f"Content size {len(content)} exceeds limit {self._config.max_write_bytes}",
                duration_ms=self._elapsed(start),
            )
        resolved = self._resolve_safe(rel_path)
        if resolved is None:
            return ToolResult(
                ok=False,
                operation=operation,
                error="Invalid or unsafe path",
                duration_ms=self._elapsed(start),
            )
        ok, error, d_before, d_after = self._atomic_write(
            resolved, content, require_exists=require_exists, operation=operation
        )
        return ToolResult(
            ok=ok,
            operation=operation,
            error=error,
            digest_before=d_before,
            digest_after=d_after,
            duration_ms=self._elapsed(start),
        )

    def create(self, rel_path: str, content: bytes, lease: str | None = None) -> ToolResult:
        """Create a new file atomically. Fails if file already exists."""
        return self._mutate_file(rel_path, content, "create", require_exists=False, lease=lease)

    def write(self, rel_path: str, content: bytes, lease: str | None = None) -> ToolResult:
        """Write (overwrite) a file atomically."""
        return self._mutate_file(rel_path, content, "write", require_exists=None, lease=lease)

    def replace(self, rel_path: str, old: str, new: str, lease: str | None = None) -> ToolResult:
        """Replace exact string in file. Fails if old is not found or ambiguous."""
        start = time.perf_counter()
        err = self._check_lease(lease)
        if err:
            return ToolResult(
                ok=False, operation="replace", error=err, duration_ms=self._elapsed(start)
            )

        resolved = self._resolve_safe(rel_path)
        if resolved is None:
            return ToolResult(
                ok=False,
                operation="replace",
                error="Invalid or unsafe path",
                duration_ms=self._elapsed(start),
            )
        if not resolved.exists():
            return ToolResult(
                ok=False,
                operation="replace",
                error=f"File not found: {rel_path}",
                duration_ms=self._elapsed(start),
            )

        try:
            content = resolved.read_text(encoding=self._ENCODING)
        except OSError as e:
            return ToolResult(
                ok=False,
                operation="replace",
                error=f"Read failed: {e}",
                duration_ms=self._elapsed(start),
            )

        count = content.count(old)
        if count == 0:
            return ToolResult(
                ok=False,
                operation="replace",
                error=f"Target string not found in {rel_path}",
                duration_ms=self._elapsed(start),
            )
        if count > 1:
            return ToolResult(
                ok=False,
                operation="replace",
                error=f"Ambiguous replacement: '{old}' appears {count} times in {rel_path}",
                duration_ms=self._elapsed(start),
            )

        new_content = content.replace(old, new, 1)
        new_bytes = new_content.encode(self._ENCODING)

        if len(new_bytes) > self._config.max_write_bytes:
            return ToolResult(
                ok=False,
                operation="replace",
                error="Replacement would exceed write size limit",
                duration_ms=self._elapsed(start),
            )

        ok, error, d_before, d_after = self._atomic_write(
            resolved,
            new_bytes,
            require_exists=True,
            operation="replace",
        )
        return ToolResult(
            ok=ok,
            operation="replace",
            error=error,
            digest_before=d_before,
            digest_after=d_after,
            duration_ms=self._elapsed(start),
        )

    def patch(
        self,
        rel_path: str,
        patches: builtins.list[dict[str, str]],
        lease: str | None = None,
    ) -> ToolResult:
        """Apply exact-replacement patches. Each patch has 'old' and 'new' keys.

        All patches are validated before any writes. If any patch fails,
        the file is unchanged.
        """
        start = time.perf_counter()
        err = self._check_lease(lease)
        if err:
            return ToolResult(
                ok=False, operation="patch", error=err, duration_ms=self._elapsed(start)
            )

        if not patches:
            return ToolResult(
                ok=False,
                operation="patch",
                error="No patches provided",
                duration_ms=self._elapsed(start),
            )

        resolved = self._resolve_safe(rel_path)
        if resolved is None:
            return ToolResult(
                ok=False,
                operation="patch",
                error="Invalid or unsafe path",
                duration_ms=self._elapsed(start),
            )
        if not resolved.exists():
            return ToolResult(
                ok=False,
                operation="patch",
                error=f"File not found: {rel_path}",
                duration_ms=self._elapsed(start),
            )

        try:
            content = resolved.read_text(encoding=self._ENCODING)
        except OSError as e:
            return ToolResult(
                ok=False,
                operation="patch",
                error=f"Read failed: {e}",
                duration_ms=self._elapsed(start),
            )

        for i, p in enumerate(patches):
            old_text = p.get("old", "")
            if old_text not in content:
                return ToolResult(
                    ok=False,
                    operation="patch",
                    error=f"Patch {i}: old text not found in {rel_path}",
                    duration_ms=self._elapsed(start),
                )

        new_content = content
        for p in patches:
            new_content = new_content.replace(p["old"], p["new"], 1)

        new_bytes = new_content.encode(self._ENCODING)
        if len(new_bytes) > self._config.max_write_bytes:
            return ToolResult(
                ok=False,
                operation="patch",
                error="Patched content would exceed write size limit",
                duration_ms=self._elapsed(start),
            )

        ok, error, d_before, d_after = self._atomic_write(
            resolved,
            new_bytes,
            require_exists=True,
            operation="patch",
        )
        return ToolResult(
            ok=ok,
            operation="patch",
            error=error,
            digest_before=d_before,
            digest_after=d_after,
            duration_ms=self._elapsed(start),
        )

    def execute_command(self, profile: CommandProfile) -> ToolResult:
        """Execute a named command profile as a subprocess.

        - shell=False always
        - workspace cwd
        - Bounded capture via tempfiles (no unbounded memory growth)
        - timeout with process-tree termination on Windows
        - env allowlist (no API key inheritance)
        - isolation="docker" wraps argv in a throwaway no-network container
          (workspace bind-mounted at /workspace) and FAILS CLOSED when Docker
          is unavailable -- it never silently runs on the host instead.
        """
        start = time.perf_counter()
        container_name = None
        if profile.isolation == "docker":
            available, detail = docker_available()
            if not available:
                return ToolResult(
                    ok=False,
                    operation="execute_command",
                    error=(
                        f"command {profile.name!r} requires Docker isolation but "
                        f"Docker is unavailable ({detail}); refusing to run on host"
                    ),
                    duration_ms=self._elapsed(start),
                )
            container_name = f"northstack-{secrets.token_hex(8)}"
            argv = wrap_docker_argv(profile.argv, self._root, profile.docker_image, container_name)
            cwd: Path | None = self._root
        else:
            argv = profile.argv
            cwd = self._root if profile.workspace_cwd else None
        result = run_command(
            argv,
            cwd=cwd,
            env_allowlist=profile.env_allowlist,
            timeout_seconds=profile.timeout_seconds,
            max_output_bytes=profile.max_output_bytes,
        )
        if container_name and "timed out" in result.error.lower():
            cleanup = run_command(
                ["docker", "rm", "-f", container_name],
                cwd=self._root,
                env_allowlist=profile.env_allowlist,
                timeout_seconds=5,
                max_output_bytes=4096,
            )
            inspect = run_command(
                ["docker", "container", "inspect", container_name],
                cwd=self._root,
                env_allowlist=profile.env_allowlist,
                timeout_seconds=5,
                max_output_bytes=4096,
            )
            detail = (inspect.stdout + inspect.stderr).decode(errors="replace").lower()
            if inspect.ok or (not cleanup.ok and "no such" not in detail):
                reason = "container still exists" if inspect.ok else cleanup.error or detail.strip()
                result = result.model_copy(
                    update={"error": f"{result.error}; container cleanup failed: {reason}"}
                )
        return ToolResult(
            ok=result.ok,
            operation="execute_command",
            data=result.stdout + result.stderr,
            truncated=result.truncated,
            total_bytes=result.total_bytes,
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
            error=result.error,
        )

    def acquire_lease(self, owner: str) -> str | None:
        """Try to acquire a mutation lease. Returns token or None.

        Re-entrant for the current owner: the worker re-acquires before each
        mutation, so a same-owner call must return the live token rather than
        report the workspace locked against itself.

        Locked because concurrent cells share one workspace: an unguarded
        check-then-set lets two owners both observe a free lease and both
        claim it, which is the exclusion this method exists to provide.
        """
        with self._lease_lock:
            if self._lease is not None and not self._lease.is_expired():
                return self._lease.token if self._lease.owner == owner else None
            token = secrets.token_hex(32)
            self._lease = _Lease(token, owner, self._config.lease_timeout_seconds)
            return token

    def release_lease(self, token: str) -> None:
        """Release a mutation lease."""
        with self._lease_lock:
            if self._lease and self._lease.token == token:
                self._lease = None

    def _elapsed(self, start: float) -> int:
        return int((time.perf_counter() - start) * 1000)
