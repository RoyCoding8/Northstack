"""Deterministic, bounded workspace scan for intake (repo constraints analysis).

The scan is a *control-plane* fact-gathering step, not a model judgment: it
walks the workspace through :class:`RestrictedWorkspace` (the single path
safety chokepoint -- symlink/junction/traversal rejection comes free), skips
sensitive paths via the same policy the workspace tools apply, and produces a
frozen :class:`RepoScan` with a content digest. Two scans of the same tree
always agree, so the ledger's recorded ``scan_digest`` is reproducible
evidence of what intake saw.

Bounds: entry counts and read sizes are capped; a tree larger than the cap is
marked ``truncated`` rather than silently partial.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from northstack.adapters.workspace.restricted import RestrictedWorkspace

_KEY_FILES: tuple[str, ...] = (
    "README.md",
    "README.rst",
    "README.txt",
    "pyproject.toml",
    "setup.py",
    "requirements.txt",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "Makefile",
    "tox.ini",
)

_SKIPPED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".northstack",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
)

_EXTENSION_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "c++",
    ".hpp": "c++",
    ".cs": "c#",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".sh": "shell",
    ".ps1": "powershell",
    ".toml": "toml",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
}


class RepoScan(BaseModel):
    """Frozen, bounded snapshot of a workspace's shape.

    ``file_extensions`` counts only files the bounded walk saw; ``key_files``
    maps a present manifest to a full or explicitly prefixed digest;
    ``truncated`` is True when any bound clipped the walk.
    """

    model_config = ConfigDict(frozen=True)

    root: str
    top_level: list[str] = Field(default_factory=list)
    directories: list[str] = Field(default_factory=list)
    file_extensions: dict[str, int] = Field(default_factory=dict)
    key_files: dict[str, str] = Field(default_factory=dict)
    entries_seen: int = Field(default=0, ge=0)
    files_seen: int = Field(default=0, ge=0)
    truncated: bool = False

    @property
    def primary_language(self) -> str:
        """Dominant code language by file count ("" when none detected)."""
        best, best_count = "", 0
        for ext, count in self.file_extensions.items():
            language = _EXTENSION_LANGUAGES.get(ext.lower(), "")
            if language and count > best_count:
                best, best_count = language, count
        return best

    def summary(self) -> str:
        """One deterministic line describing the tree."""
        parts = [f"{self.files_seen} files"]
        if self.primary_language:
            parts.append(f"primary language {self.primary_language}")
        if self.key_files:
            parts.append("manifests: " + ", ".join(sorted(self.key_files)))
        if self.top_level:
            shown = ", ".join(self.top_level[:8])
            more = f" (+{len(self.top_level) - 8} more)" if len(self.top_level) > 8 else ""
            parts.append(f"top-level: {shown}{more}")
        if self.truncated:
            parts.append("scan truncated at bounds")
        return "; ".join(parts)

    def digest(self) -> str:
        """Stable sha256 of the canonical scan JSON."""
        payload = self.model_dump(mode="json")
        payload.pop("root")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def scan_workspace(
    root: Path | str,
    *,
    max_entries: int = 400,
    max_dirs: int = 60,
    max_extensions: int = 24,
) -> RepoScan:
    """Walk the workspace deterministically through the restricted chokepoint.

    Depth is bounded to two levels (top level + one directory down); counts
    and lists are capped. Returns a frozen :class:`RepoScan`.
    """
    if min(max_entries, max_dirs, max_extensions) < 0:
        raise ValueError("scan bounds must be nonnegative")
    workspace = RestrictedWorkspace(Path(root))
    top_result = workspace.list(".")
    top_candidates = (
        sorted(
            name
            for name in (json.loads(top_result.data) if top_result.ok and top_result.data else [])
            if name not in _SKIPPED_DIRS and not name.startswith(".")
        )
        if top_result.ok
        else []
    )
    top: list[str] = []
    directories: list[str] = []
    extensions: dict[str, int] = {}
    entries_seen = files_seen = 0
    truncated = top_result.truncated

    def _account_file(name: str) -> None:
        nonlocal files_seen
        files_seen += 1
        suffix = Path(name).suffix.lower()
        if suffix:
            extensions[suffix] = extensions.get(suffix, 0) + 1

    def _account_directory(name: str) -> bool:
        nonlocal truncated
        if len(directories) >= max_dirs:
            truncated = True
            return False
        directories.append(name)
        return True

    for name in top_candidates:
        if entries_seen >= max_entries:
            truncated = True
            break
        entries_seen += 1
        top.append(name)
        nested = workspace.list(name)
        if not nested.ok:
            _account_file(name)
            continue
        if not _account_directory(name):
            continue
        truncated |= nested.truncated
        children = [
            child
            for child in (json.loads(nested.data) if nested.data else [])
            if child not in _SKIPPED_DIRS and not child.startswith(".")
        ]
        for child in children:
            if entries_seen >= max_entries:
                truncated = True
                break
            entries_seen += 1
            child_path = f"{name}/{child}"
            child_list = workspace.list(child_path)
            if child_list.ok:
                _account_directory(child_path)
            else:
                _account_file(child_path)

    if len(extensions) > max_extensions:
        top_exts = dict(sorted(extensions.items(), key=lambda kv: (-kv[1], kv[0]))[:max_extensions])
        extensions = top_exts
        truncated = True

    key_files: dict[str, str] = {}
    all_top = set(top)
    for key in _KEY_FILES:
        if key in all_top and len(key_files) < 12:
            read_result = workspace.read(key)
            if read_result.ok:
                digest = hashlib.sha256(read_result.data).hexdigest()
                key_files[key] = (
                    f"sha256-prefix:{digest}:{len(read_result.data)}/{read_result.total_bytes}"
                    if read_result.truncated
                    else f"sha256:{digest}"
                )
                truncated |= read_result.truncated

    return RepoScan(
        root=str(Path(root)),
        top_level=top,
        directories=sorted(directories),
        file_extensions=dict(sorted(extensions.items())),
        key_files=key_files,
        entries_seen=entries_seen,
        files_seen=files_seen,
        truncated=truncated,
    )


def conventions_from_scan(scan: RepoScan) -> tuple[list[str], list[str]]:
    """Derive deterministic conventions/patterns from a scan.

    Returns ``(conventions, existing_patterns)`` for the repo analysis. These
    are conservative statements of fact (what manifests exist), not guesses.
    """
    conventions: list[str] = []
    patterns: list[str] = []
    language = scan.primary_language
    if language:
        conventions.append(f"workspace primarily contains {language} code")

    if "pyproject.toml" in scan.key_files:
        conventions.append("python project defined by pyproject.toml")
        patterns.append("use pyproject.toml as the project definition")
    if "setup.py" in scan.key_files:
        patterns.append("legacy setup.py present; prefer pyproject.toml for new config")
    if "requirements.txt" in scan.key_files:
        conventions.append("dependencies pinned in requirements.txt")
    if "package.json" in scan.key_files:
        conventions.append("node project defined by package.json")
        patterns.append("use the package.json scripts as the command surface")
    if "Cargo.toml" in scan.key_files:
        conventions.append("rust crate defined by Cargo.toml")
    if "go.mod" in scan.key_files:
        conventions.append("go module defined by go.mod")
    if any(k.startswith("README") for k in scan.key_files):
        patterns.append("README present; keep documentation conventions it establishes")
    if "tests" in scan.top_level or "test" in scan.top_level:
        conventions.append("a dedicated tests directory exists")
    if scan.truncated:
        conventions.append("workspace scan was truncated at bounds; verify scope manually")
    return conventions, patterns
