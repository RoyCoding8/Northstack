from __future__ import annotations

import os
from pathlib import Path

import pytest

from northstack.adapters.artifacts import ArtifactStore
from northstack.adapters.workspace.restricted import RestrictedWorkspace, WorkspaceConfig
from northstack.application.verification.hard_gates import HardGateVerifier, compute_tree_digest
from northstack.domain import (
    Budget,
    FileDiffCriterion,
    SchemaCriterion,
    TreeDigestCriterion,
    WorkContract,
)


def _contract(criterion: object) -> WorkContract:
    return WorkContract(
        id="wc-hard-safety",
        objective="verify",
        deliverables=["result"],
        budget=Budget(token_limit=100, cost_limit_usd=0.1),
        acceptance_criteria=[criterion],
    )


def _verifier(root: Path, *, max_read_bytes: int = 1_048_576) -> HardGateVerifier:
    return HardGateVerifier(
        RestrictedWorkspace(root, WorkspaceConfig(max_read_bytes=max_read_bytes)),
        ArtifactStore(root / ".artifacts"),
    )


async def test_file_diff_rejects_truncated_content_even_when_prefix_matches(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("required trailing-data", encoding="utf-8")
    result = (
        await _verifier(tmp_path, max_read_bytes=8).verify(
            _contract(
                FileDiffCriterion(
                    description="full content",
                    path="large.txt",
                    content_contains="required",
                )
            )
        )
    )[0]
    assert not result.passed
    assert "truncated" in result.detail
    assert "8" in result.detail and str(len("required trailing-data")) in result.detail


async def test_must_not_exist_fails_when_file_exists(tmp_path: Path) -> None:
    (tmp_path / "forbidden.txt").write_text("present", encoding="utf-8")
    result = (
        await _verifier(tmp_path).verify(
            _contract(
                FileDiffCriterion(
                    description="must remain absent",
                    path="forbidden.txt",
                    must_exist=False,
                )
            )
        )
    )[0]
    assert not result.passed


async def test_must_not_exist_does_not_treat_sensitive_denial_as_absence(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("SECRET=x", encoding="utf-8")
    result = (
        await _verifier(tmp_path).verify(
            _contract(FileDiffCriterion(description="absent", path=".env", must_exist=False))
        )
    )[0]
    assert not result.passed
    assert "sensitive_denied" in result.detail


async def test_tree_digest_rejects_traversal_and_missing_directory(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-tree"
    outside.mkdir(exist_ok=True)
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    verifier = _verifier(tmp_path)
    for path in ("../outside-tree", "missing"):
        result = (
            await verifier.verify(
                _contract(TreeDigestCriterion(description="tree", path=path, tree_hash="0" * 64))
            )
        )[0]
        assert not result.passed
        assert "error" in result.detail


def test_tree_digest_rejects_links(tmp_path: Path) -> None:
    tree, outside = tmp_path / "tree", tmp_path / "outside.txt"
    tree.mkdir()
    outside.write_text("secret", encoding="utf-8")
    try:
        os.symlink(outside, tree / "linked.txt")
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    with pytest.raises(ValueError, match="link"):
        compute_tree_digest(tmp_path, "tree")


async def test_checker_crash_becomes_one_failed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _verifier(tmp_path)

    def crash(*args: object) -> None:
        raise RuntimeError("injected checker crash")

    monkeypatch.setattr(verifier, "_check_file_diff", crash)
    results = await verifier.verify(_contract(FileDiffCriterion(description="file", path="x.txt")))
    assert len(results) == 1
    assert not results[0].passed
    assert "checker crashed" in results[0].detail


async def test_invalid_schema_is_distinct_from_candidate_failure(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)
    ref = verifier._artifact_store.write(b"{}", media_type="application/json")
    invalid = SchemaCriterion(
        description="invalid schema",
        artifact_digest=ref.digest,
        json_schema={"type": 42},
    )
    mismatch = SchemaCriterion(
        description="candidate mismatch",
        artifact_digest=ref.digest,
        json_schema={"type": "array"},
    )
    invalid_result = (await verifier.verify(_contract(invalid)))[0]
    mismatch_result = (await verifier.verify(_contract(mismatch)))[0]
    assert "invalid JSON schema" in invalid_result.detail
    assert "candidate failed schema" in mismatch_result.detail
