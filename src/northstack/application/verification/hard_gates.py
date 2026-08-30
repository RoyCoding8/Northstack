"""Hard acceptance gates: real command execution and real evidence.

Hard gate failure cannot be waived; the soft rubric only runs when every hard
gate passes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import assert_never

from pydantic import BaseModel, ConfigDict, Field

from northstack.adapters.artifacts import ArtifactStore
from northstack.adapters.workspace.restricted import CommandProfile, RestrictedWorkspace, ToolResult
from northstack.domain.contract import (
    AcceptanceCriterion,
    CommandCriterion,
    CriterionKind,
    FileDiffCriterion,
    PolicyCriterion,
    SchemaCriterion,
    SoftRubricCriterion,
    TreeDigestCriterion,
    WorkContract,
)
from northstack.domain.outcome import ArtifactRef

_TREE_EXCLUDE_DIRS = frozenset({"__pycache__", ".pytest_cache", ".northstack"})
_TREE_MAX_ENTRIES = 10_000
_TREE_MAX_BYTES = 64 * 1024 * 1024
_TREE_MAX_DEPTH = 64


def compute_tree_digest(
    root: Path,
    rel_dir: str,
    *,
    max_entries: int = _TREE_MAX_ENTRIES,
    max_bytes: int = _TREE_MAX_BYTES,
    max_depth: int = _TREE_MAX_DEPTH,
) -> str:
    """Stable digest of every file under ``root/rel_dir``.

    Feeds sorted relative paths and per-file content hashes into one SHA-256,
    so edits, deletions, and newly added files all change the digest. Cache
    directories are excluded. A missing directory is an error; it is never
    conflated with an empty directory. The
    compile side hashes the same way, so that state is representable.
    """
    root, rel = root.resolve(), Path(rel_dir)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe tree path: {rel_dir}")
    cursor = root
    for part in rel.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"tree path contains link: {rel_dir}")
    base = (root / rel).resolve(strict=True)
    if not base.is_relative_to(root):
        raise ValueError(f"tree path escapes workspace: {rel_dir}")
    if not base.is_dir():
        raise NotADirectoryError(rel_dir)
    files: list[tuple[str, Path]] = []
    stack, entries = [(base, 0)], 0
    while stack:
        directory, depth = stack.pop()
        with os.scandir(directory) as scanned:
            children = sorted(scanned, key=lambda entry: entry.name)
        for entry in children:
            is_dir = entry.is_dir(follow_symlinks=False)
            if is_dir and entry.name in _TREE_EXCLUDE_DIRS:
                continue
            entries += 1
            if entries > max_entries:
                raise ValueError(f"tree entry limit exceeded: {max_entries}")
            if entry.is_symlink():
                raise ValueError(f"tree contains link: {entry.path}")
            path = Path(entry.path)
            if is_dir:
                if depth >= max_depth:
                    raise ValueError(f"tree depth limit exceeded: {max_depth}")
                stack.append((path, depth + 1))
            elif entry.is_file(follow_symlinks=False):
                files.append((path.relative_to(base).as_posix(), path))
            else:
                raise ValueError(f"tree contains unsupported entry: {entry.path}")
    h, total = hashlib.sha256(b"northstack-tree-v2\0"), 0
    for relative, path in sorted(files):
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or total + before.st_size > max_bytes:
            raise ValueError(f"tree byte limit exceeded: {max_bytes}")
        file_hash, size = hashlib.sha256(), 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                if total + size > max_bytes:
                    raise ValueError(f"tree byte limit exceeded: {max_bytes}")
                file_hash.update(chunk)
            opened = os.fstat(stream.fileno())
        after = path.stat(follow_symlinks=False)
        identities = {
            (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
            for value in (before, opened, after)
        }
        if len(identities) != 1:
            raise OSError(f"tree file changed while hashing: {relative}")
        total += size
        h.update(relative.encode("utf-8"))
        h.update(b"\0")
        h.update(file_hash.digest())
    return h.hexdigest()


class CommandEvidence(BaseModel):
    """Serialised record of a command gate's runtime evidence.

    Stored as the evidence artifact for a command criterion. Built from the
    real ``ToolResult`` (stdout/stderr bytes decoded explicitly, the boolean
    ``ok`` flag, and the error string) and serialised via ``model_dump_json``,
    so the stored artifact is always valid JSON -- never an f-string quoted
    repr of a ``bytes`` value that ``json.loads`` would reject.
    """

    model_config = ConfigDict(frozen=True)

    stdout: str = Field(default="")
    ok: bool
    error: str = Field(default="")


class HardCheckResult(BaseModel):
    """Result of a single hard-gate check."""

    model_config = ConfigDict(frozen=True)

    criterion_index: int = Field(ge=0)
    kind: CriterionKind
    passed: bool
    detail: str = Field(default="")
    evidence_ref: ArtifactRef | None = None


class HardGateVerifier:
    """Executes hard acceptance criteria through the workspace.

    Command checks run actual CommandProfile via workspace.execute_command,
    store stdout/stderr as content-addressed artifact, and use real exit code.
    File checks verify actual file existence/digest. Schema validates JSON
    with jsonschema. Policy checks actual tool/capability audit.
    """

    def __init__(
        self,
        workspace: RestrictedWorkspace,
        artifact_store: ArtifactStore,
        command_profiles: dict[str, CommandProfile] | None = None,
    ) -> None:
        self._workspace = workspace
        self._artifact_store = artifact_store
        self._command_profiles = command_profiles or {}

    async def verify(
        self,
        contract: WorkContract,
        evidence_digests: dict[int, str] | None = None,
        tools_used: list[str] | None = None,
    ) -> list[HardCheckResult]:
        results: list[HardCheckResult] = []
        evidence_digests = evidence_digests or {}
        tools_used = tools_used or []

        for i, criterion in enumerate(contract.acceptance_criteria):
            checked = (
                await asyncio.gather(
                    self._check_criterion(criterion, i, evidence_digests.get(i, ""), tools_used),
                    return_exceptions=True,
                )
            )[0]
            if isinstance(checked, Exception):
                result = HardCheckResult(
                    criterion_index=i,
                    kind=criterion.kind,
                    passed=False,
                    detail=f"checker crashed: {type(checked).__name__}: {str(checked)[:200]}",
                )
            elif isinstance(checked, BaseException):
                raise checked
            else:
                result = checked
            results.append(result)

        return results

    async def _check_criterion(
        self,
        criterion: AcceptanceCriterion,
        index: int,
        evidence_digest: str,
        tools_used: list[str],
    ) -> HardCheckResult:
        match criterion:
            case CommandCriterion():
                return await asyncio.to_thread(self._check_command, criterion, index)
            case FileDiffCriterion():
                return await asyncio.to_thread(self._check_file_diff, criterion, index)
            case TreeDigestCriterion():
                return await asyncio.to_thread(self._check_tree_digest, criterion, index)
            case SchemaCriterion():
                return await asyncio.to_thread(
                    self._check_schema, criterion, index, evidence_digest
                )
            case PolicyCriterion():
                return self._check_policy(criterion, index, tools_used)
            case SoftRubricCriterion():
                return HardCheckResult(
                    criterion_index=index,
                    kind=criterion.kind,
                    passed=True,
                    detail="not a hard gate; evaluated by the soft rubric",
                )
            case _ as unreachable:
                assert_never(unreachable)

    def _check_command(self, criterion: CommandCriterion, index: int) -> HardCheckResult:
        """Execute a command criterion via workspace and store real evidence."""
        cmd_name = criterion.command_name
        expected_exit_code = criterion.exit_code

        if not cmd_name or cmd_name not in self._command_profiles:
            return HardCheckResult(
                criterion_index=index,
                kind=CriterionKind.COMMAND,
                passed=False,
                detail=f"no command profile for '{cmd_name}'",
            )

        profile = self._command_profiles[cmd_name]
        tool_result: ToolResult = self._workspace.execute_command(profile)

        evidence = CommandEvidence(
            stdout=tool_result.data.decode("utf-8", errors="replace"),
            ok=tool_result.ok,
            error=tool_result.error,
        )
        ref = self._artifact_store.write(
            evidence.model_dump_json().encode("utf-8"),
            media_type="application/json",
        )

        passed = tool_result.exit_code == expected_exit_code
        actual = "not started" if tool_result.exit_code is None else str(tool_result.exit_code)

        return HardCheckResult(
            criterion_index=index,
            kind=CriterionKind.COMMAND,
            passed=passed,
            detail=(
                f"command '{cmd_name}' exited {actual}; expected {expected_exit_code}"
                if not passed
                else f"command '{cmd_name}' exited {actual} as expected"
            ),
            evidence_ref=ref,
        )

    def _check_tree_digest(self, criterion: TreeDigestCriterion, index: int) -> HardCheckResult:
        """Compare the directory's live digest against the compiled one."""
        try:
            actual = compute_tree_digest(self._workspace.root, criterion.path)
        except (OSError, ValueError) as exc:
            return HardCheckResult(
                criterion_index=index,
                kind=CriterionKind.TREE_DIGEST,
                passed=False,
                detail=f"tree digest error for '{criterion.path}': {exc}",
            )
        passed = actual == criterion.tree_hash
        return HardCheckResult(
            criterion_index=index,
            kind=CriterionKind.TREE_DIGEST,
            passed=passed,
            detail=(
                f"tree '{criterion.path}' matches compiled digest"
                if passed
                else (
                    f"tree '{criterion.path}' changed since compile "
                    f"(expected {criterion.tree_hash[:16]}..., got {actual[:16]}...)"
                )
            ),
        )

    def _check_file_diff(self, criterion: FileDiffCriterion, index: int) -> HardCheckResult:
        """Check file existence/digest/exact content via workspace."""
        path = criterion.path
        tool_result: ToolResult = self._workspace.read(path)

        def result(passed: bool, detail: str, ref: ArtifactRef | None = None) -> HardCheckResult:
            return HardCheckResult(
                criterion_index=index,
                kind=CriterionKind.FILE_DIFF,
                passed=passed,
                detail=detail,
                evidence_ref=ref,
            )

        if not tool_result.ok:
            category = tool_result.error_kind or "unknown_error"
            if not criterion.must_exist and category == "not_found":
                return result(True, f"file '{path}' correctly absent")
            return result(False, f"file '{path}' read failed ({category}): {tool_result.error}")
        if not criterion.must_exist:
            return result(False, f"file '{path}' exists but must be absent")
        if tool_result.truncated:
            return result(
                False,
                f"file '{path}' read truncated: observed {len(tool_result.data)} of "
                f"{tool_result.total_bytes} bytes ({tool_result.truncation_reason})",
            )

        if criterion.content_hash:
            actual_hash = hashlib.sha256(tool_result.data).hexdigest()
            if actual_hash != criterion.content_hash:
                return result(
                    False,
                    f"file '{path}' hash mismatch: expected {criterion.content_hash[:16]}..., "
                    f"got {actual_hash[:16]}...",
                )

        try:
            text = tool_result.data.decode("utf-8")
        except UnicodeDecodeError:
            return result(False, f"file '{path}' is not valid UTF-8")
        if criterion.content_equals is not None and text != criterion.content_equals:
            return result(
                False,
                f"file '{path}' content mismatch: expected exact {criterion.content_equals!r}",
            )
        if criterion.content_contains is not None and criterion.content_contains not in text:
            return result(
                False,
                f"file '{path}' missing required content: {criterion.content_contains!r}",
            )

        ref = self._artifact_store.write(tool_result.data, media_type="application/octet-stream")
        return result(True, f"file '{path}' check passed", ref)

    def _check_schema(
        self,
        criterion: SchemaCriterion,
        index: int,
        runtime_artifact_digest: str = "",
    ) -> HardCheckResult:
        """Validate the authoritative runtime artifact against a JSON schema."""
        artifact_digest = runtime_artifact_digest or criterion.artifact_digest
        json_schema = criterion.json_schema

        def result(passed: bool, detail: str) -> HardCheckResult:
            return HardCheckResult(
                criterion_index=index, kind=CriterionKind.SCHEMA, passed=passed, detail=detail
            )

        if not artifact_digest or not json_schema:
            return result(False, "missing artifact_digest or json_schema")

        try:
            ref = ArtifactRef(digest=artifact_digest, media_type="application/json", size_bytes=0)
            data = json.loads(self._artifact_store.read(ref))
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            return result(False, f"artifact read/parse failed: {e}")

        import jsonschema

        try:
            jsonschema.Draft202012Validator.check_schema(json_schema)
        except jsonschema.SchemaError as exc:
            return result(False, f"invalid JSON schema: {exc.message[:200]}")
        try:
            jsonschema.Draft202012Validator(json_schema).validate(data)
        except jsonschema.ValidationError as exc:
            return result(False, f"candidate failed schema: {exc.message[:200]}")
        return result(True, "schema validation passed")

    def _check_policy(
        self,
        criterion: PolicyCriterion,
        index: int,
        tools_used: list[str],
    ) -> HardCheckResult:
        """Check policy compliance against the authoritative tool audit."""
        check = criterion.check

        if check == "forbidden_tools":
            forbidden = criterion.tools
            violations = [t for t in tools_used if t in forbidden]
            passed = len(violations) == 0
            return HardCheckResult(
                criterion_index=index,
                kind=CriterionKind.POLICY,
                passed=passed,
                detail=f"forbidden tools: {violations}" if violations else "policy ok",
            )

        return HardCheckResult(
            criterion_index=index,
            kind=CriterionKind.POLICY,
            passed=False,
            detail=f"unsupported policy check '{check}'",
        )
