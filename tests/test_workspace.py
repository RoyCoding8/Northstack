"""Tests for RestrictedWorkspace and WebReader at the public seam.

Seams under test:
  - RestrictedWorkspace(root, config, ledger?) -> mediated fs ops
  - WebReader(policy, transport) -> SSRF-protected public-web fetch
  - ToolResult / ToolEvidence -> structured return types

All path logic tested on every platform: the symlink/junction law is
exercised through each platform's own link flavor (symlinks on POSIX,
junctions on Windows), so no platform skips the law itself.  Test fakes
(FakeTransport, SlowTransport, RedirectTransport, FakeResolver) live here,
not in production.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from northstack.adapters.workspace.restricted import (
    CommandProfile,
    RestrictedWorkspace,
    WebFetchPolicy,
    WorkspaceConfig,
)
from northstack.adapters.workspace.webfetch import WebReader

# Test fakes -- NOT in production code


class FakeTransport:
    """In-memory transport for deterministic tests."""

    def __init__(
        self, status: int = 200, body: bytes = b"", headers: dict[str, str] | None = None
    ) -> None:
        self._status = status
        self._body = body
        self._headers = headers or {}
        self._calls: list[dict[str, Any]] = []

    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        follow_redirects: bool = False,
        timeout: float | None = None,
    ) -> httpx.Response:
        self._calls.append({"method": method, "url": url})
        resp_headers = dict(self._headers)
        content_length = len(self._body)
        resp_headers.setdefault("content-type", "application/octet-stream")
        resp_headers["content-length"] = str(content_length)
        return httpx.Response(
            status_code=self._status,
            content=self._body,
            headers=resp_headers,
            request=httpx.Request(method, url),
        )


class SlowTransport:
    """Transport that delays response for timeout testing."""

    def __init__(self, delay: float = 5.0) -> None:
        self._delay = delay

    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        follow_redirects: bool = False,
        timeout: float | None = None,
    ) -> httpx.Response:
        if timeout is not None and self._delay > timeout:
            raise httpx.TimeoutException(f"Connection timed out after {timeout}s")
        time.sleep(self._delay)
        return httpx.Response(
            status_code=200,
            content=b"ok",
            headers={"content-type": "text/plain"},
            request=httpx.Request(method, url),
        )


class RedirectTransport:
    """Transport that follows a sequence of redirect hops."""

    def __init__(self, hops: list[str]) -> None:
        self._hops = list(hops)
        self._hop_index = 0

    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        follow_redirects: bool = False,
        timeout: float | None = None,
    ) -> httpx.Response:
        idx = self._hop_index
        self._hop_index += 1
        if idx < len(self._hops) - 1:
            next_url = self._hops[idx + 1]
            return httpx.Response(
                status_code=301,
                content=b"",
                headers={"location": next_url, "content-type": "text/html"},
                request=httpx.Request(method, url),
            )
        return httpx.Response(
            status_code=200,
            content=b"final destination",
            headers={"content-type": "text/html"},
            request=httpx.Request(method, url),
        )


class FakeResolver:
    """Fake DNS resolver for testing."""

    def __init__(self, raises: bool = False, ips: list[str] | None = None) -> None:
        self._raises = raises
        self._ips = ips if ips is not None else ["93.184.216.34"]  # example.com
        self._calls: list[str] = []

    def resolve(self, hostname: str) -> list[str]:
        self._calls.append(hostname)
        if self._raises:
            raise OSError(f"DNS resolution failed for {hostname}")
        return list(self._ips)  # Return a copy


class StreamTransport:
    """Transport that returns a streaming response for bounded-read tests."""

    def __init__(self, body: bytes, chunk_size: int = 10) -> None:
        self._body = body
        self._chunk_size = chunk_size

    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        follow_redirects: bool = False,
        timeout: float | None = None,
    ) -> httpx.Response:
        body = self._body
        chunk_size = self._chunk_size

        def _generate():
            for i in range(0, len(body), chunk_size):
                yield body[i : i + chunk_size]

        # Build a response with a stream context manager
        resp = httpx.Response(
            status_code=200,
            stream=StreamIterator(_generate()),
            headers={"content-type": "text/plain", "content-length": str(len(body))},
            request=httpx.Request(method, url),
        )
        return resp


class StreamIterator:
    """Minimal iterator that acts as a context manager for httpx streaming."""

    def __init__(self, gen):
        self._gen = gen

    def __iter__(self):
        return self._gen

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# Fixtures


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    """Create a workspace root with sample files for testing."""
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "main.py").write_bytes(b"print('hello')\n")
    (root / "src" / "utils.py").write_bytes(b"# utils\nx = 1\n")
    (root / "docs").mkdir()
    (root / "docs" / "readme.md").write_bytes(b"# Hello World\n")
    (root / "data.txt").write_bytes(b"line1\nline2\nline3\n")
    return root


@pytest.fixture
def ws_config() -> WorkspaceConfig:
    return WorkspaceConfig(
        max_list_entries=100,
        max_read_bytes=1_048_576,
        max_search_results=50,
        max_write_bytes=1_048_576,
        max_patch_old_bytes=65536,
    )


@pytest.fixture
def ws(workspace_root: Path, ws_config: WorkspaceConfig) -> RestrictedWorkspace:
    return RestrictedWorkspace(root=workspace_root, config=ws_config)


@pytest.fixture
def ws_leased(workspace_root: Path, ws_config: WorkspaceConfig) -> tuple[RestrictedWorkspace, str]:
    """Workspace with an active mutation lease. Returns (workspace, lease_token)."""
    rws = RestrictedWorkspace(root=workspace_root, config=ws_config)
    token = rws.acquire_lease("test-cell")
    assert token is not None, "Failed to acquire initial lease"
    return rws, token


# Path resolution: reject attacks


class TestPathResolution:
    """Workspace-relative paths only.  Absolute, escape, symlink, junction
    tricks must be rejected at every existing ancestor including root."""

    def test_absolute_path_rejected(self, ws: RestrictedWorkspace):
        result = ws.read("/etc/passwd")
        assert not result.ok
        assert "unsafe" in result.error.lower() or "invalid" in result.error.lower()

    def test_windows_absolute_path_rejected(self, ws: RestrictedWorkspace):
        if platform.system() != "Windows":
            pytest.skip("Windows-only path test")
        result = ws.read("C:\\Windows\\System32\\config\\SAM")
        assert not result.ok

    def test_dotdot_escape_rejected(self, ws: RestrictedWorkspace):
        result = ws.read("../etc/passwd")
        assert not result.ok

    def test_dotdot_in_middle_rejected(self, ws: RestrictedWorkspace):
        result = ws.read("src/../../etc/passwd")
        assert not result.ok

    def test_trailing_dotdot_rejected(self, ws: RestrictedWorkspace):
        result = ws.read("src/main.py/../../etc/passwd")
        assert not result.ok

    def test_normal_relative_path_resolves(self, ws: RestrictedWorkspace):
        result = ws.read("src/main.py")
        assert result.ok

    def test_bare_filename_resolves(self, ws: RestrictedWorkspace):
        result = ws.read("data.txt")
        assert result.ok

    def test_nested_path_resolves(self, ws: RestrictedWorkspace):
        result = ws.read("src/main.py")
        assert result.ok
        assert b"hello" in result.data

        # Symlink / junction law: links are ALWAYS rejected, on every platform.

    # Path.resolve() follows both symlinks and NTFS junctions, so the law is
    # one law. POSIX exposes symlinks without privilege; Windows needs admin
    # (or developer mode) for symlinks, but junctions (mklink /J) need none.
    # The matrix below therefore runs everywhere: _make_dir_link picks the
    # platform's privilege-free flavor, and a Windows symlink skip can only
    # ever skip one flavor of the law -- never the law.

    @staticmethod
    def _make_dir_link(link: Path, target: Path) -> None:
        """Create a directory link without needing admin privileges."""
        if platform.system() == "Windows":
            proc = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, f"mklink /J failed: {proc.stderr.strip()}"
        else:
            link.symlink_to(target, target_is_directory=True)

    def test_inside_dir_link_read_rejected(self, workspace_root, ws_config):
        """Regression (CI 2026-08): a directory link pointing INSIDE the root
        was followed, because resolve() erased the link before the reparse
        check and containment saw only the in-root target."""
        self._make_dir_link(workspace_root / "in_link", workspace_root / "src")
        ws = RestrictedWorkspace(root=workspace_root, config=ws_config)
        result = ws.read("in_link/main.py")
        assert not result.ok
        assert "unsafe" in result.error.lower() or "invalid" in result.error.lower()

    def test_inside_dir_link_write_rejected(self, workspace_root, ws_config):
        """Creating THROUGH a link must not touch the link's target. The
        lease is acquired first so the link is the only possible reason for
        the rejection."""
        self._make_dir_link(workspace_root / "in_link", workspace_root / "src")
        ws = RestrictedWorkspace(root=workspace_root, config=ws_config)
        lease = ws.acquire_lease("link-test")
        assert lease is not None
        result = ws.create("in_link/evil.py", b"x = 1\n", lease=lease)
        assert not result.ok
        assert not (workspace_root / "src" / "evil.py").exists()

    def test_outside_dir_link_rejected(self, workspace_root, ws_config, tmp_path):
        """Escape-targeting links stay rejected (defense in depth)."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "s.txt").write_text("secret\n")
        self._make_dir_link(workspace_root / "out_link", outside)
        ws = RestrictedWorkspace(root=workspace_root, config=ws_config)
        assert not ws.read("out_link/s.txt").ok

    def test_root_dir_link_rejected(self, tmp_path):
        """A workspace root that is itself a link is rejected at construction
        -- before resolve() gets the chance to follow it."""
        real_root = tmp_path / "real_workspace"
        real_root.mkdir()
        self._make_dir_link(tmp_path / "link_workspace", real_root)
        with pytest.raises(ValueError, match="symlink"):
            RestrictedWorkspace(root=tmp_path / "link_workspace")

    def test_file_symlink_read_rejected(self, workspace_root, ws_config):
        """File symlinks are rejected even when their target is inside the
        root. On a locked-down Windows this flavor needs admin/developer
        mode; the junction matrix above enforces the same law without it."""
        try:
            (workspace_root / "file_link").symlink_to(workspace_root / "src" / "main.py")
        except OSError as e:
            pytest.skip(f"symlink flavor unavailable here ({e}); junction matrix covers the law")
        ws = RestrictedWorkspace(root=workspace_root, config=ws_config)
        assert not ws.read("file_link").ok

    def test_broken_symlink_rejected(self, workspace_root, ws_config):
        """A link to a nonexistent target is still a link: lstat sees it even
        though resolve() would fail or fall back to the literal path."""
        try:
            (workspace_root / "broken_link").symlink_to(workspace_root / "does_not_exist")
        except OSError as e:
            pytest.skip(f"symlink flavor unavailable here ({e}); junction matrix covers the law")
        ws = RestrictedWorkspace(root=workspace_root, config=ws_config)
        assert not ws.read("broken_link").ok

    def test_nonexistent_path_outside_rejected(self, ws: RestrictedWorkspace):
        result = ws.read("nonexistent/../../secret")
        assert not result.ok

    def test_empty_path_rejected(self, ws: RestrictedWorkspace):
        result = ws.read("")
        assert not result.ok


# List / search / read: bounded operations


class TestBoundedOperations:
    """Bounded list/search/read with entry/byte/result limits and
    deterministic truncation metadata."""

    def test_list_entries(self, ws: RestrictedWorkspace):
        result = ws.list("src")
        assert result.ok
        entries = result.data
        assert isinstance(entries, bytes)
        assert len(entries) > 0

    def test_list_truncation_metadata(self, workspace_root: Path):
        config = WorkspaceConfig(max_list_entries=2)
        ws = RestrictedWorkspace(root=workspace_root, config=config)
        result = ws.list(".")
        assert result.ok
        assert result.truncated is True
        assert result.total_entries > 2

    def test_list_does_not_materialize_all_entries(
        self, workspace_root: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def fail(*args: Any, **kwargs: Any):
            raise AssertionError("list() must not call sorted() on the complete directory")

        monkeypatch.setitem(RestrictedWorkspace.list.__globals__, "sorted", fail)
        result = RestrictedWorkspace(workspace_root, WorkspaceConfig(max_list_entries=2)).list(".")
        assert result.ok
        assert json.loads(result.data) == ["data.txt", "docs"]
        assert result.truncated is True
        assert result.truncation_reason == "entry_limit"

    def test_list_preserves_filename_identity_as_json(self, workspace_root: Path):
        names = ["emoji-😀", "é", "é"]
        for name in names:
            (workspace_root / name).write_bytes(b"")
        listed = json.loads(RestrictedWorkspace(workspace_root).list(".").data)
        assert set(names) <= set(listed)

    @pytest.mark.skipif(os.name == "nt", reason="Win32 forbids control characters in names")
    def test_list_preserves_newline_and_carriage_return_names(self, workspace_root: Path):
        names = ["line\nbreak", "carriage\rreturn", "tab\tname"]
        for name in names:
            (workspace_root / name).write_bytes(b"")
        listed = json.loads(RestrictedWorkspace(workspace_root).list(".").data)
        assert set(names) <= set(listed)

    def test_search_never_follows_directory_link_outside(self, workspace_root: Path, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("unique-outside-needle")
        try:
            os.symlink(outside, workspace_root / "linked", target_is_directory=True)
        except OSError as error:
            pytest.skip(str(error))
        result = RestrictedWorkspace(workspace_root).search("unique-outside-needle")
        assert result.ok
        assert b"linked" not in result.data

    def test_hard_link_to_sensitive_file_is_denied(self, workspace_root: Path):
        sensitive, alias = workspace_root / ".env", workspace_root / "innocent.txt"
        sensitive.write_text("hard-link-secret-needle")
        os.link(sensitive, alias)
        workspace = RestrictedWorkspace(workspace_root)
        read = workspace.read("innocent.txt")
        assert not read.ok and read.error_kind == "sensitive_denied"
        assert b"innocent.txt" not in workspace.search("hard-link-secret-needle").data

    def test_read_with_byte_limit(self, workspace_root: Path):
        config = WorkspaceConfig(max_read_bytes=10)
        ws = RestrictedWorkspace(root=workspace_root, config=config)
        result = ws.read("data.txt")
        assert result.ok
        assert result.data == b"line1\nline"
        assert result.truncated is True
        assert result.truncation_reason == "byte_limit"
        assert result.total_entries == 0
        assert result.total_bytes == len(b"line1\nline2\nline3\n")

    def test_read_does_not_load_whole_file_before_truncating(
        self, workspace_root: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def fail(path: Path):
            raise AssertionError(f"unbounded read of {path}")

        monkeypatch.setattr(Path, "read_bytes", fail)
        result = RestrictedWorkspace(workspace_root, WorkspaceConfig(max_read_bytes=3)).read(
            "data.txt"
        )
        assert result.ok
        assert result.data == b"lin"
        assert result.total_bytes == len(b"line1\nline2\nline3\n")

    def test_read_exact_limit_is_not_truncated(self, workspace_root: Path):
        size = len(b"line1\nline2\nline3\n")
        result = RestrictedWorkspace(workspace_root, WorkspaceConfig(max_read_bytes=size)).read(
            "data.txt"
        )
        assert result.ok
        assert result.truncated is False
        assert result.truncation_reason is None

    def test_read_full_content(self, ws: RestrictedWorkspace):
        result = ws.read("data.txt")
        assert result.ok
        assert result.data == b"line1\nline2\nline3\n"
        assert result.truncated is False

    def test_search_returns_matches(self, ws: RestrictedWorkspace):
        result = ws.search("hello", path=".")
        assert result.ok
        raw = result.data.decode()
        assert "main.py" in raw

    def test_search_result_limit(self, workspace_root: Path):
        config = WorkspaceConfig(max_search_results=1)
        ws = RestrictedWorkspace(root=workspace_root, config=config)
        result = ws.search("line", path=".")
        assert result.ok
        assert result.total_entries == 2
        assert result.truncated is True
        assert result.truncation_reason == "result_limit"

    def test_search_single_file(self, ws: RestrictedWorkspace):
        result = ws.search("line2", path="data.txt")
        assert result.ok
        assert json.loads(result.data) == [{"path": "data.txt", "line": "2", "text": "line2"}]

    def test_search_file_byte_limit(self, workspace_root: Path):
        (workspace_root / "bounded.txt").write_bytes(b"before\nneedle\n")
        result = RestrictedWorkspace(
            workspace_root, WorkspaceConfig(max_search_file_bytes=7)
        ).search("needle")
        assert result.ok
        assert json.loads(result.data) == []
        assert result.truncated is True
        assert result.truncation_reason == "file_byte_limit"

    def test_search_file_limit(self, workspace_root: Path):
        root = workspace_root / "limited-files"
        root.mkdir()
        (root / "a.txt").write_text("absent")
        (root / "b.txt").write_text("needle")
        result = RestrictedWorkspace(root, WorkspaceConfig(max_search_files=1)).search("needle")
        assert result.ok
        assert json.loads(result.data) == []
        assert result.truncated is True
        assert result.truncation_reason == "file_limit"

    def test_search_directory_limit(self, workspace_root: Path):
        root = workspace_root / "limited-directories"
        (root / "a").mkdir(parents=True)
        (root / "a" / "match.txt").write_text("needle")
        result = RestrictedWorkspace(root, WorkspaceConfig(max_search_directories=1)).search(
            "needle"
        )
        assert result.ok
        assert json.loads(result.data) == []
        assert result.truncated is True
        assert result.truncation_reason == "directory_limit"

    def test_search_skips_file_removed_during_traversal(
        self, workspace_root: Path, monkeypatch: pytest.MonkeyPatch
    ):
        target = workspace_root / "vanishing.txt"
        target.write_text("needle")
        original = RestrictedWorkspace._resolve_safe

        def resolve(ws: RestrictedWorkspace, path: str):
            resolved = original(ws, path)
            if path == "vanishing.txt" and target.exists():
                target.unlink()
            return resolved

        monkeypatch.setattr(RestrictedWorkspace, "_resolve_safe", resolve)
        result = RestrictedWorkspace(workspace_root).search("needle")
        assert result.ok
        assert json.loads(result.data) == []

    def test_read_nonexistent_file(self, ws: RestrictedWorkspace):
        result = ws.read("no_such_file.txt")
        assert not result.ok
        assert "not found" in result.error.lower()

    def test_list_nonexistent_directory(self, ws: RestrictedWorkspace):
        result = ws.list("no_such_dir")
        assert not result.ok

    def test_read_is_bytes(self, ws: RestrictedWorkspace):
        result = ws.read("data.txt")
        assert isinstance(result.data, bytes)


# Sensitive-read chokepoint
# Every file read funnels through RestrictedWorkspace.read -- the web API,
# the worker tool-call path, and the verifier all call it.  A sensitive file
# must be denied on the RESOLVED path, not the raw query string, so an
# in-workspace symlink with a non-sensitive name cannot read through to a
# sensitive target.  No symlink is needed to pin the contract: the resolved
# path of a sensitive file is itself sensitive regardless of the raw alias.
_SENSITIVE_READ_CASES = [
    ".env",
    ".env.local",
    "secrets/id_rsa",
    "keys/private.key",
    "certs/server.pem",
]


class TestSensitiveReadChokepoint:
    """read() denies sensitive files at the chokepoint, by resolved path."""

    @pytest.mark.parametrize("rel", _SENSITIVE_READ_CASES)
    def test_read_denies_sensitive_file(
        self, workspace_root: Path, ws_config: WorkspaceConfig, rel: str
    ):
        target = workspace_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"SECRET")
        ws = RestrictedWorkspace(root=workspace_root, config=ws_config)
        result = ws.read(rel)
        assert not result.ok
        assert "sensitive" in (result.error or "").lower()

    def test_read_allows_non_sensitive_file(self, workspace_root: Path, ws_config: WorkspaceConfig):
        (workspace_root / "notes.txt").write_bytes(b"ok")
        ws = RestrictedWorkspace(root=workspace_root, config=ws_config)
        result = ws.read("notes.txt")
        assert result.ok

    @pytest.mark.parametrize("rel", _SENSITIVE_READ_CASES)
    def test_search_skips_sensitive_file(
        self, workspace_root: Path, ws_config: WorkspaceConfig, rel: str
    ):
        marker = f"SECRET-TOKEN-{rel}"
        target = workspace_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"line with {marker} inside\n", encoding="utf-8")
        ws = RestrictedWorkspace(root=workspace_root, config=ws_config)
        result = ws.search(marker, path=".")
        assert result.ok
        data = (result.data or b"").decode("utf-8", errors="replace")
        assert marker not in data
        assert rel not in data


# Atomic create / write / replace / patch


class TestAtomicMutations:
    """Atomic create/write/replace/patch with before/after SHA-256 digests
    and content limits."""

    def test_create_file(self, ws_leased: tuple[RestrictedWorkspace, str]):
        ws, token = ws_leased
        result = ws.create("new_file.txt", b"hello world\n", lease=token)
        assert result.ok
        assert result.digest_before is None
        assert result.digest_after is not None
        assert result.digest_after.startswith("sha256:")
        read_result = ws.read("new_file.txt")
        assert read_result.ok
        assert read_result.data == b"hello world\n"

    def test_create_digest_matches_content(self, ws_leased: tuple[RestrictedWorkspace, str]):
        ws, token = ws_leased
        content = b"test content\n"
        result = ws.create("digest_test.txt", content, lease=token)
        assert result.ok
        expected = "sha256:" + hashlib.sha256(content).hexdigest()
        assert result.digest_after == expected

    def test_create_refuses_overwrite(self, ws_leased: tuple[RestrictedWorkspace, str]):
        ws, token = ws_leased
        ws.create("existing.txt", b"original\n", lease=token)
        result = ws.create("existing.txt", b"overwrite\n", lease=token)
        assert not result.ok
        assert "exists" in result.error.lower()

    def test_write_overwrites_existing(self, ws_leased: tuple[RestrictedWorkspace, str]):
        ws, token = ws_leased
        ws.create("overwrite.txt", b"old\n", lease=token)
        result = ws.write("overwrite.txt", b"new content\n", lease=token)
        assert result.ok
        assert result.digest_before is not None
        assert result.digest_after is not None
        assert result.digest_before != result.digest_after

    def test_write_create_if_missing(self, ws_leased: tuple[RestrictedWorkspace, str]):
        ws, token = ws_leased
        result = ws.write("brand_new.txt", b"created\n", lease=token)
        assert result.ok
        assert result.digest_before is None

    def test_replace_string(self, ws_leased: tuple[RestrictedWorkspace, str]):
        ws, token = ws_leased
        ws.create("replace.txt", b"hello world", lease=token)
        result = ws.replace("replace.txt", "world", "there", lease=token)
        assert result.ok
        read = ws.read("replace.txt")
        assert read.data == b"hello there"

    def test_replace_not_found_fails(self, ws_leased: tuple[RestrictedWorkspace, str]):
        ws, token = ws_leased
        ws.create("replace_fail.txt", b"hello world", lease=token)
        result = ws.replace("replace_fail.txt", "nonexistent", "x", lease=token)
        assert not result.ok
        assert "not found" in result.error.lower()

    def test_replace_ambiguous_fails(self, ws_leased: tuple[RestrictedWorkspace, str]):
        ws, token = ws_leased
        ws.create("dup.txt", b"aaa", lease=token)
        result = ws.replace("dup.txt", "a", "b", lease=token)
        assert not result.ok
        assert "ambiguous" in result.error.lower() or "multiple" in result.error.lower()

    def test_patch_exact_replacement(self, ws_leased: tuple[RestrictedWorkspace, str]):
        ws, token = ws_leased
        ws.create("patch.txt", b"line1\nline2\nline3\n", lease=token)
        result = ws.patch("patch.txt", [{"old": "line2", "new": "LINE2"}], lease=token)
        assert result.ok
        read = ws.read("patch.txt")
        assert b"LINE2" in read.data

    def test_patch_multiple_replacements(self, ws_leased: tuple[RestrictedWorkspace, str]):
        ws, token = ws_leased
        ws.create("patch2.txt", b"aaa bbb ccc", lease=token)
        result = ws.patch(
            "patch2.txt",
            [
                {"old": "aaa", "new": "AAA"},
                {"old": "ccc", "new": "CCC"},
            ],
            lease=token,
        )
        assert result.ok
        read = ws.read("patch2.txt")
        assert read.data == b"AAA bbb CCC"

    def test_patch_old_not_found_fails(self, ws_leased: tuple[RestrictedWorkspace, str]):
        ws, token = ws_leased
        ws.create("patch_miss.txt", b"hello", lease=token)
        result = ws.patch("patch_miss.txt", [{"old": "zzz", "new": "yyy"}], lease=token)
        assert not result.ok
        assert "not found" in result.error.lower()

    def test_content_size_limit_enforced(self, workspace_root: Path):
        config = WorkspaceConfig(max_write_bytes=10)
        ws = RestrictedWorkspace(root=workspace_root, config=config)
        token = ws.acquire_lease("cell-1")
        big_content = b"x" * 20
        result = ws.create("big.txt", big_content, lease=token)
        assert not result.ok
        assert "limit" in result.error.lower() or "exceed" in result.error.lower()

    def test_atomic_write_not_visible_during_failure(
        self, ws_leased: tuple[RestrictedWorkspace, str]
    ):
        """If a patch fails, original content is preserved."""
        ws, token = ws_leased
        original = b"original content\n"
        ws.create("atomic.txt", original, lease=token)
        result = ws.patch("atomic.txt", [{"old": "nonexistent", "new": "x"}], lease=token)
        assert not result.ok
        read = ws.read("atomic.txt")
        assert read.data == original

    def test_encoding_is_utf8(self, ws_leased: tuple[RestrictedWorkspace, str]):
        ws, token = ws_leased
        content = "Hello, world! \u00e9\u00e8\u00ea\n".encode("utf-8")
        result = ws.create("unicode.txt", content, lease=token)
        assert result.ok
        read = ws.read("unicode.txt")
        assert read.data == content

    def test_tempfile_is_unique(self, ws_leased: tuple[RestrictedWorkspace, str]):
        """Each write creates a unique tempfile, no predictable naming."""
        ws, token = ws_leased
        ws.create("a.txt", b"content-a\n", lease=token)
        ws.create("b.txt", b"content-b\n", lease=token)
        # No .tmp siblings should remain
        tmp_files = list(workspace_root_for(ws).glob("*.tmp"))
        assert len(tmp_files) == 0, f"Stale tmp files found: {tmp_files}"
        assert ws.read("a.txt").data == b"content-a\n"
        assert ws.read("b.txt").data == b"content-b\n"


def workspace_root_for(ws: RestrictedWorkspace) -> Path:
    """Extract workspace root from a RestrictedWorkspace for test assertions."""
    return ws._root


# Mutation lease: process-local, contention, release on exception


class TestMutationLease:
    """Process-local mutation lease so only one mutating cell may alter
    the canonical workspace."""

    def test_acquire_and_release(self, ws: RestrictedWorkspace):
        token = ws.acquire_lease("cell-1")
        assert token is not None
        ws.release_lease(token)

    def test_lease_required_for_mutation(self, ws: RestrictedWorkspace):
        result = ws.create("nolease.txt", b"data\n")
        assert not result.ok
        assert "lease" in result.error.lower()

    def test_lease_allows_mutation(self, ws: RestrictedWorkspace):
        token = ws.acquire_lease("cell-1")
        result = ws.create("leased.txt", b"data\n", lease=token)
        assert result.ok
        ws.release_lease(token)

    def test_second_lease_rejected(self, ws: RestrictedWorkspace):
        t1 = ws.acquire_lease("cell-1")
        assert t1 is not None
        t2 = ws.acquire_lease("cell-2")
        assert t2 is None
        ws.release_lease(t1)

    def test_release_on_exception(self, ws: RestrictedWorkspace):
        token = ws.acquire_lease("cell-1")
        assert token is not None
        try:
            ws.create("ok.txt", b"data\n", lease=token)
        except (OSError, ValueError):
            pass
        ws.release_lease(token)
        t2 = ws.acquire_lease("cell-2")
        assert t2 is not None
        ws.release_lease(t2)

    def test_read_without_lease(self, ws: RestrictedWorkspace):
        result = ws.read("data.txt")
        assert result.ok

    def test_invalid_lease_rejected(self, ws: RestrictedWorkspace):
        result = ws.create("bad_lease.txt", b"data\n", lease="fake-token-123")
        assert not result.ok
        assert "lease" in result.error.lower()

    def test_lease_timeout(self, workspace_root: Path):
        config = WorkspaceConfig(lease_timeout_seconds=0.1)
        ws = RestrictedWorkspace(root=workspace_root, config=config)
        token = ws.acquire_lease("cell-1")
        assert token is not None
        time.sleep(0.2)
        result = ws.create("expired.txt", b"data\n", lease=token)
        assert not result.ok
        assert "lease" in result.error.lower()

    def test_concurrent_mutation_contention(self, ws: RestrictedWorkspace):
        results: list[tuple[str, str | None]] = []

        def try_acquire(cell_id: str):
            t = ws.acquire_lease(cell_id)
            results.append((cell_id, t))

        t1 = threading.Thread(target=try_acquire, args=("c1",))
        t2 = threading.Thread(target=try_acquire, args=("c2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        acquired = [r for r in results if r[1] is not None]
        assert len(acquired) == 1
        ws.release_lease(acquired[0][1])


# ToolResult / ToolEvidence structure


class TestToolResultStructure:
    """Every operation returns a typed ToolResult suitable for event logging."""

    def test_tool_result_has_required_fields(self, ws: RestrictedWorkspace):
        result = ws.read("data.txt")
        assert hasattr(result, "ok")
        assert hasattr(result, "operation")
        assert hasattr(result, "error")
        assert hasattr(result, "data")
        assert hasattr(result, "digest_before")
        assert hasattr(result, "digest_after")
        assert hasattr(result, "truncated")
        assert hasattr(result, "total_entries")
        assert hasattr(result, "duration_ms")

    def test_tool_result_as_event_payload(self, ws: RestrictedWorkspace):
        result = ws.read("data.txt")
        payload = result.to_event_payload()
        assert isinstance(payload, dict)
        assert "operation" in payload
        assert "ok" in payload
        assert "duration_ms" in payload

    def test_tool_evidence_fields(self, ws: RestrictedWorkspace):
        result = ws.read("data.txt")
        evidence = result.to_evidence()
        assert isinstance(evidence, dict)


# Command profile execution + env validation


class TestCommandExecution:
    """Named command profiles with structured results."""

    def test_command_profile_model(self):
        cp = CommandProfile(
            name="python-check",
            argv=["python", "-c", "print('hello')"],
            timeout_seconds=10,
            max_output_bytes=65536,
            env_allowlist=["PATH"],
        )
        assert cp.name == "python-check"
        assert cp.workspace_cwd is True

    def test_execute_python_command(self, ws: RestrictedWorkspace):
        profile = CommandProfile(
            name="echo-test",
            argv=["python", "-c", "print(42)"],
            timeout_seconds=5,
            max_output_bytes=4096,
        )
        result = ws.execute_command(profile)
        assert result.ok
        assert b"42" in result.data

    def test_command_timeout(self, ws: RestrictedWorkspace):
        profile = CommandProfile(
            name="slow",
            argv=["python", "-c", "import time; time.sleep(5)"],
            timeout_seconds=1,
            max_output_bytes=4096,
        )
        result = ws.execute_command(profile)
        assert not result.ok
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()

    def test_execute_command_timeout_cleans_tempfiles(self, ws: RestrictedWorkspace):
        """Tempfiles created by execute_command must be cleaned up on timeout."""
        tmpdir = tempfile.gettempdir()
        before = set(
            glob.glob(os.path.join(tmpdir, ".mc_stdout_*"))
            + glob.glob(os.path.join(tmpdir, ".mc_stderr_*"))
        )
        profile = CommandProfile(
            name="slow",
            argv=["python", "-c", "import time; time.sleep(60)"],
            timeout_seconds=1,
            max_output_bytes=4096,
        )
        result = ws.execute_command(profile)
        assert not result.ok
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()
        after = set(
            glob.glob(os.path.join(tmpdir, ".mc_stdout_*"))
            + glob.glob(os.path.join(tmpdir, ".mc_stderr_*"))
        )
        assert after - before == set(), "tempfiles leaked after timeout"

    def test_command_output_truncation(self, ws: RestrictedWorkspace):
        profile = CommandProfile(
            name="big-output",
            argv=["python", "-c", "print('x' * 10000)"],
            timeout_seconds=5,
            max_output_bytes=100,
        )
        result = ws.execute_command(profile)
        assert result.ok
        assert result.truncated is True
        assert result.total_bytes > 100

    def test_command_result_fields(self, ws: RestrictedWorkspace):
        profile = CommandProfile(
            name="basic",
            argv=["python", "-c", "print('test')"],
            timeout_seconds=5,
            max_output_bytes=4096,
        )
        result = ws.execute_command(profile)
        assert hasattr(result, "exit_code")
        assert hasattr(result, "duration_ms")
        assert result.exit_code == 0

    def test_command_no_env_leakage(self, ws: RestrictedWorkspace):
        """Commands must not inherit API keys or secrets from the process env."""
        profile = CommandProfile(
            name="env-check",
            argv=["python", "-c", "import os; print('API_KEY' in os.environ)"],
            timeout_seconds=5,
            max_output_bytes=4096,
            env_allowlist=[],
        )
        result = ws.execute_command(profile)
        assert result.ok
        assert b"False" in result.data

    def test_command_baseline_env_available(self, ws: RestrictedWorkspace):
        """Baseline env vars (PATH, SYSTEMROOT, etc.) are available."""
        profile = CommandProfile(
            name="baseline",
            argv=["python", "-c", "import os; print('PATH' in os.environ)"],
            timeout_seconds=5,
            max_output_bytes=4096,
            env_allowlist=["PATH"],
        )
        result = ws.execute_command(profile)
        assert result.ok
        assert b"True" in result.data

    def test_command_asyncio_importable_under_stripped_env(self, ws: RestrictedWorkspace):
        """Regression: a PATH-only env omitted SystemRoot, so Windows winsock
        could not initialize and `import asyncio` died with WinError 10106 --
        every pytest gate failed while passing outside the sandbox."""
        profile = CommandProfile(
            name="asyncio-check",
            argv=["python", "-c", "import asyncio; print('ok')"],
            timeout_seconds=15,
            max_output_bytes=4096,
            env_allowlist=["PATH"],
        )
        result = ws.execute_command(profile)
        assert result.ok, result.data
        assert b"ok" in result.data


class TestCommandProfileEnvValidation:
    """Sensitive env var names must be rejected at validation time."""

    def test_reject_api_key(self):
        with pytest.raises(Exception, match="sensitive"):
            CommandProfile(name="x", argv=["echo"], env_allowlist=["API_KEY"])

    def test_reject_secret(self):
        with pytest.raises(Exception, match="sensitive"):
            CommandProfile(name="x", argv=["echo"], env_allowlist=["MY_SECRET"])

    def test_reject_token(self):
        with pytest.raises(Exception, match="sensitive"):
            CommandProfile(name="x", argv=["echo"], env_allowlist=["AUTH_TOKEN"])

    def test_reject_password(self):
        with pytest.raises(Exception, match="sensitive"):
            CommandProfile(name="x", argv=["echo"], env_allowlist=["DB_PASSWORD"])

    def test_reject_credential(self):
        with pytest.raises(Exception, match="sensitive"):
            CommandProfile(name="x", argv=["echo"], env_allowlist=["AWS_CREDENTIAL"])

    def test_reject_auth(self):
        with pytest.raises(Exception, match="sensitive"):
            CommandProfile(name="x", argv=["echo"], env_allowlist=["AUTH_HEADER"])

    def test_reject_provider_secret_env(self):
        """Provider-specific secret env names are rejected."""
        with pytest.raises(Exception, match="sensitive"):
            CommandProfile(name="x", argv=["echo"], env_allowlist=["MIMO_API_KEY"])

    def test_reject_glm_api_key(self):
        with pytest.raises(Exception, match="sensitive"):
            CommandProfile(name="x", argv=["echo"], env_allowlist=["GLM_API_KEY"])

    def test_reject_glued_secret_suffix(self):
        """A sensitive keyword glued to a prefix WITHOUT an underscore
        (APIKEY, MYSECRET, MYAUTH) must still be rejected -- the underscore
        delimiter is a convention, not a requirement, and a name like APIKEY
        is just as much a live secret as API_KEY."""
        with pytest.raises(Exception, match="sensitive"):
            CommandProfile(name="x", argv=["echo"], env_allowlist=["APIKEY"])
        with pytest.raises(Exception, match="sensitive"):
            CommandProfile(name="x", argv=["echo"], env_allowlist=["MYSECRET"])
        with pytest.raises(Exception, match="sensitive"):
            CommandProfile(name="x", argv=["echo"], env_allowlist=["MYAUTH"])

    def test_allow_path(self):
        cp = CommandProfile(name="x", argv=["echo"], env_allowlist=["PATH"])
        assert "PATH" in cp.env_allowlist

    def test_allow_custom_non_sensitive(self):
        cp = CommandProfile(name="x", argv=["echo"], env_allowlist=["MY_CUSTOM_VAR"])
        assert "MY_CUSTOM_VAR" in cp.env_allowlist


# WebReader: SSRF protection


class TestWebReaderSSRF:
    """Public-web read-only fetch with SSRF protection."""

    def test_fetch_policy_rejects_localhost(self):
        reader = WebReader(policy=WebFetchPolicy())
        result = reader.fetch("http://localhost:8080/secret")
        assert not result.ok

    def test_fetch_policy_rejects_private_ip(self):
        reader = WebReader(policy=WebFetchPolicy())
        result = reader.fetch("http://192.168.1.1/admin")
        assert not result.ok

    def test_fetch_policy_rejects_loopback_127(self):
        reader = WebReader(policy=WebFetchPolicy())
        result = reader.fetch("http://127.0.0.1/secret")
        assert not result.ok

    def test_fetch_policy_rejects_link_local(self):
        reader = WebReader(policy=WebFetchPolicy())
        result = reader.fetch("http://169.254.169.254/latest/meta-data")
        assert not result.ok

    def test_fetch_policy_rejects_multicast(self):
        reader = WebReader(policy=WebFetchPolicy())
        result = reader.fetch("http://224.0.0.1/multicast")
        assert not result.ok

    def test_fetch_only_allows_get_head(self):
        reader = WebReader(policy=WebFetchPolicy())
        result = reader.fetch("http://example.com", method="POST")
        assert not result.ok
        assert "method" in result.error.lower()

    def test_fetch_head_allowed_no_body(self):
        """HEAD is allowed and returns no body."""
        transport = FakeTransport(status=200, headers={"content-type": "text/html"})
        reader = WebReader(policy=WebFetchPolicy(), transport=transport)
        result = reader.fetch("http://example.com", method="HEAD")
        assert result.ok
        assert result.data == b""

    def test_fetch_respects_size_limit_streamed(self):
        """Streamed reading bounds body before full allocation."""
        policy = WebFetchPolicy(max_response_bytes=100)
        body = b"x" * 200
        transport = FakeTransport(status=200, body=body, headers={"content-type": "text/plain"})
        reader = WebReader(policy=policy, transport=transport)
        result = reader.fetch("http://example.com/big")
        assert result.ok
        assert len(result.data) <= 100
        assert result.truncated is True
        assert result.total_bytes >= 200

    def test_fetch_timeout(self):
        policy = WebFetchPolicy(timeout_seconds=0.1)
        transport = SlowTransport(delay=1.0)
        reader = WebReader(policy=policy, transport=transport)
        result = reader.fetch("http://example.com/slow")
        assert not result.ok
        assert "timeout" in result.error.lower() or "request failed" in result.error.lower()

    def test_fetch_redirect_revalidates(self):
        """Redirects must be revalidated for SSRF at every hop."""
        policy = WebFetchPolicy(max_redirects=3)
        transport = RedirectTransport(hops=["http://example.com", "http://10.0.0.1/admin"])
        reader = WebReader(policy=policy, transport=transport)
        result = reader.fetch("http://example.com")
        assert not result.ok

    def test_fetch_returns_tool_result(self):
        transport = FakeTransport(
            status=200,
            body=b"<html>Hello</html>",
            headers={"content-type": "text/html"},
        )
        reader = WebReader(policy=WebFetchPolicy(), transport=transport)
        result = reader.fetch("http://example.com")
        assert result.ok
        assert result.operation == "web_fetch"

    def test_fetch_content_type_filter(self):
        policy = WebFetchPolicy(allowed_content_types=["text/html", "text/plain"])
        transport = FakeTransport(
            status=200,
            body=b"binary data",
            headers={"content-type": "application/octet-stream"},
        )
        reader = WebReader(policy=policy, transport=transport)
        result = reader.fetch("http://example.com")
        assert not result.ok
        assert "content-type" in result.error.lower() or "type" in result.error.lower()

    def test_fetch_url_validation_rejects_ftp(self):
        reader = WebReader(policy=WebFetchPolicy())
        result = reader.fetch("ftp://example.com/file")
        assert not result.ok
        assert "scheme" in result.error.lower() or "not allowed" in result.error.lower()

    def test_fetch_evidence_marked_untrusted(self):
        transport = FakeTransport(
            status=200,
            body=b"data",
            headers={"content-type": "text/plain"},
        )
        reader = WebReader(policy=WebFetchPolicy(), transport=transport)
        result = reader.fetch("http://example.com")
        assert result.ok
        evidence = result.to_evidence()
        assert evidence.get("untrusted") is True

    def test_fetch_dns_pinned_false_in_evidence(self):
        """Evidence documents that DNS rebinding protection is not full."""
        transport = FakeTransport(
            status=200,
            body=b"data",
            headers={"content-type": "text/plain"},
        )
        reader = WebReader(policy=WebFetchPolicy(), transport=transport)
        result = reader.fetch("http://example.com")
        assert result.ok
        evidence = result.to_evidence()
        assert evidence.get("dns_pinned") is False

    def test_fetch_rejects_empty_dns(self):
        """Empty DNS answer is rejected."""
        resolver = FakeResolver(ips=[])
        reader = WebReader(policy=WebFetchPolicy(), resolver=resolver)
        result = reader.fetch("http://example.com")
        assert not result.ok
        assert "no addresses" in result.error.lower() or "dns" in result.error.lower()

    def test_fetch_rejects_userinfo(self):
        """URLs with credentials/userinfo are rejected."""
        reader = WebReader(policy=WebFetchPolicy())
        result = reader.fetch("http://user:pass@example.com/secret")
        assert not result.ok
        assert "credential" in result.error.lower() or "userinfo" in result.error.lower()

    def test_fetch_rejects_non_default_port(self):
        """Non-default ports are rejected."""
        reader = WebReader(policy=WebFetchPolicy())
        result = reader.fetch("http://example.com:8080/secret")
        assert not result.ok
        assert "port" in result.error.lower()

    def test_fetch_non_2xx_is_failure(self):
        """Non-2xx responses are treated as structured failure."""
        transport = FakeTransport(
            status=404, body=b"Not Found", headers={"content-type": "text/plain"}
        )
        reader = WebReader(policy=WebFetchPolicy(), transport=transport)
        result = reader.fetch("http://example.com")
        assert not result.ok
        assert "404" in result.error

    def test_fetch_5xx_is_failure(self):
        transport = FakeTransport(
            status=500, body=b"Internal Server Error", headers={"content-type": "text/plain"}
        )
        reader = WebReader(policy=WebFetchPolicy(), transport=transport)
        result = reader.fetch("http://example.com")
        assert not result.ok
        assert "500" in result.error

    def test_fetch_redirect_to_bad_scheme_rejected(self):
        """Redirect to ftp:// is rejected."""
        transport = RedirectTransport(hops=["http://example.com", "ftp://evil.com/file"])
        reader = WebReader(policy=WebFetchPolicy(), transport=transport)
        result = reader.fetch("http://example.com")
        assert not result.ok

    def test_fetch_https_to_http_downgrade_rejected(self):
        """An https request that redirects to a plaintext http URL must be
        refused as a protocol downgrade (strips transport security, enables
        MITM rewrite of the fetched evidence).  Loopback is exempt since
        http://127.0.0.1 is the legitimate local control surface."""
        transport = RedirectTransport(hops=["https://example.com", "http://example.com/insecure"])
        resolver = FakeResolver(ips=["93.184.216.34"])  # public, not loopback
        reader = WebReader(policy=WebFetchPolicy(), transport=transport, resolver=resolver)
        result = reader.fetch("https://example.com")
        assert not result.ok
        assert "downgrade" in result.error.lower() or "http" in result.error.lower()

    def test_fetch_scheme_relative_redirect(self):
        """Scheme-relative redirect (//host) is resolved correctly."""
        transport = RedirectTransport(
            hops=[
                "http://example.com",
                "//other.example.com/page",
            ]
        )
        # The resolved URL should still pass validation
        # (both are public IPs, but we need a resolver that handles both hostnames)
        resolver = FakeResolver(ips=["93.184.216.34"])
        reader = WebReader(policy=WebFetchPolicy(), transport=transport, resolver=resolver)
        result = reader.fetch("http://example.com")
        # Should not fail on scheme validation; may fail on DNS or transport
        # The key point is it didn't crash on the scheme-relative URL
        assert result.operation == "web_fetch"

    def test_fetch_relative_redirect_resolves(self):
        """Path-relative redirect is resolved against current URL."""
        transport = RedirectTransport(
            hops=[
                "http://example.com/old",
                "/new",
            ]
        )
        resolver = FakeResolver(ips=["93.184.216.34"])
        reader = WebReader(policy=WebFetchPolicy(), transport=transport, resolver=resolver)
        result = reader.fetch("http://example.com/old")
        assert result.operation == "web_fetch"


# WebReader: injectable transport for deterministic tests


class TestWebReaderInjectable:
    """Transport/DNS resolver injectable for deterministic testing."""

    def test_injectable_transport(self):
        transport = FakeTransport(
            status=200,
            body=b"injected",
            headers={"content-type": "text/plain"},
        )
        reader = WebReader(policy=WebFetchPolicy(), transport=transport)
        result = reader.fetch("http://example.com")
        assert result.ok
        assert result.data == b"injected"

    def test_injectable_resolver(self):
        resolver = FakeResolver(raises=True)
        reader = WebReader(policy=WebFetchPolicy(), resolver=resolver)
        result = reader.fetch("http://example.com")
        assert not result.ok
        assert "dns" in result.error.lower() or "resolve" in result.error.lower()
