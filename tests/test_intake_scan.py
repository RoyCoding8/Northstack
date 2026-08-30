"""Hermetic tests for the deterministic intake workspace scan."""

from __future__ import annotations

from pathlib import Path

import pytest

from northstack.application.intake_scan import conventions_from_scan, scan_workspace


@pytest.fixture
def python_repo(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "README.md").write_text("# demo\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_one.py").write_text("def test_one():\n    assert True\n")
    (tmp_path / "src_demo.py").write_text("x = 1\n")
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / ".env").write_text("SECRET=1\n")
    return tmp_path


def test_scan_detects_python_shape(python_repo: Path):
    scan = scan_workspace(python_repo)
    assert scan.primary_language == "python"
    assert "pyproject.toml" in scan.key_files
    assert "README.md" in scan.key_files
    assert ".env" not in scan.key_files
    assert "tests" in scan.top_level
    # .git internals and dotfiles are skipped entirely
    assert all(".git" not in entry for entry in scan.top_level)
    assert scan.files_seen >= 3
    assert not scan.truncated


def test_scan_is_deterministic_and_digest_stable(python_repo: Path):
    first = scan_workspace(python_repo)
    second = scan_workspace(python_repo)
    assert first == second
    assert first.digest() == second.digest()
    assert first.digest().startswith("sha256:")


def test_scan_digest_changes_when_tree_changes(python_repo: Path):
    before = scan_workspace(python_repo).digest()
    (python_repo / "another.py").write_text("y = 2\n")
    after = scan_workspace(python_repo).digest()
    assert before != after


def test_scan_marks_truncation_at_bounds(python_repo: Path):
    scan = scan_workspace(python_repo, max_entries=1)
    assert scan.truncated is True


@pytest.mark.parametrize("limit,seen", [(0, 0), (1, 1), (3, 3)])
def test_scan_obeys_exact_global_entry_limit(python_repo: Path, limit: int, seen: int):
    scan = scan_workspace(python_repo, max_entries=limit)
    assert scan.entries_seen == seen
    assert len(scan.top_level) <= seen
    assert scan.truncated is True


def test_scan_marks_top_level_list_truncation(tmp_path: Path):
    for index in range(1001):
        (tmp_path / f"file-{index:04}.txt").write_bytes(b"")
    scan = scan_workspace(tmp_path, max_entries=2000)
    assert scan.entries_seen == 1000
    assert scan.truncated is True


def test_scan_marks_directory_limit(python_repo: Path):
    scan = scan_workspace(python_repo, max_dirs=0)
    assert scan.directories == []
    assert scan.truncated is True


def test_scan_labels_partial_manifest_digest(tmp_path: Path):
    prefix = b"a" * 1_048_576
    manifest = tmp_path / "requirements.txt"
    manifest.write_bytes(prefix + b"x")
    first = scan_workspace(tmp_path)
    manifest.write_bytes(prefix + b"y")
    second = scan_workspace(tmp_path)
    assert first.key_files["requirements.txt"].startswith("sha256-prefix:")
    assert first.key_files == second.key_files
    assert first.truncated and second.truncated


def test_scan_digest_ignores_absolute_root(tmp_path: Path):
    roots = [tmp_path / "one", tmp_path / "two"]
    for root in roots:
        root.mkdir()
        (root / "same.py").write_text("x = 1\n")
    assert scan_workspace(roots[0]).digest() == scan_workspace(roots[1]).digest()


def test_scan_on_missing_root_returns_empty_but_valid(tmp_path: Path):
    scan = scan_workspace(tmp_path / "does-not-exist")
    assert scan.top_level == []
    assert scan.files_seen == 0
    assert scan.primary_language == ""


def test_conventions_from_scan_states_facts(python_repo: Path):
    scan = scan_workspace(python_repo)
    conventions, patterns = conventions_from_scan(scan)
    joined = " | ".join(conventions + patterns)
    assert "python" in joined
    assert "pyproject.toml" in joined
    assert "README" in joined
    assert "tests" in joined


def test_conventions_flag_truncated_scan(python_repo: Path):
    scan = scan_workspace(python_repo, max_entries=1)
    conventions, _ = conventions_from_scan(scan)
    assert any("truncated" in c for c in conventions)


def test_summary_is_single_line_and_bounded(python_repo: Path):
    summary = scan_workspace(python_repo).summary()
    assert "\n" not in summary
    assert "python" in summary
    assert "manifests" in summary
