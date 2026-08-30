"""Regression tests for contracting internals."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from northstack.application.contracting import (
    AcceptanceAnalysis,
    ContractCompiler,
    DeterministicAnalysisRunner,
    ModelBackedAnalysisRunner,
    _drop_ungrounded_content_checks,
)
from northstack.adapters.providers.wire import FinishReason
from northstack.application.json_extraction import extract_first_json_object
from northstack.config import CommandConfig, NorthStackConfig
from northstack.domain import ProjectRequest


class TestStripToJson:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                ('```json\n{"content_contains": "return {}", "risks": ["} breaks"]}\n```'),
                {"content_contains": "return {}", "risks": ["} breaks"]},
            ),
            (
                'Analysis follows: {"outer": {"inner": true}, "quote": "\\"}"} done.',
                {"outer": {"inner": True}, "quote": '"}'},
            ),
            ('prefix {"first": 1} suffix {"second": 2}', {"first": 1}),
        ],
    )
    def test_extracts_first_complete_json_object(self, text, expected):
        assert extract_first_json_object(text) == expected

    @pytest.mark.parametrize("text", ["plain prose", "prefix {not json}", '{"open": true'])
    def test_returns_none_without_complete_json_object(self, text):
        assert extract_first_json_object(text) is None


class _Gateway:
    def __init__(
        self,
        *,
        text: str = "",
        error: Exception | None = None,
        replies: list[tuple[str, FinishReason]] | None = None,
    ) -> None:
        self._text = text
        self._error = error
        self._replies = replies
        self.budgets: list[int] = []

    async def complete(self, request):
        if self._error is not None:
            raise self._error
        self.budgets.append(request.max_output_tokens)
        if self._replies:
            text, finish = self._replies[min(len(self.budgets), len(self._replies)) - 1]
            return SimpleNamespace(text=text, finish_reason=finish)
        return SimpleNamespace(text=self._text, finish_reason=FinishReason.END_TURN)


class TestCompleteJsonDiagnostics:
    @pytest.mark.asyncio
    async def test_warns_when_gateway_completion_fails(self, caplog):
        runner = ModelBackedAnalysisRunner(
            _Gateway(error=RuntimeError("secret provider detail")),
            "test",
        )

        with caplog.at_level(logging.WARNING, logger="northstack.application.contracting"):
            result = await runner._complete_json("prompt", {})

        assert result is None
        assert "stage=gateway_failure" in caplog.text
        assert "gateway_failure:RuntimeError" in caplog.text
        assert "secret provider detail" not in caplog.text

    @pytest.mark.asyncio
    async def test_warns_for_malformed_response_without_logging_content(self, caplog):
        runner = ModelBackedAnalysisRunner(
            _Gateway(text="prefix {not valid} API_TOKEN=secret"), "test"
        )

        with caplog.at_level(logging.WARNING, logger="northstack.application.contracting"):
            result = await runner._complete_json("prompt", {})

        assert result is None
        assert "stage=candidate_decode_failure" in caplog.text
        assert "stage=raw_text_parse_failure" in caplog.text
        assert "API_TOKEN=secret" not in caplog.text


class TestTruncatedAnalysisRetries:
    """A hidden reasoning trace can swallow a small analysis budget whole. That
    is a truncation, not a refusal, and giving up on it drops the run onto an
    uncalibrated soft rubric that can never reach ``verified``.
    """

    @pytest.mark.asyncio
    async def test_an_empty_truncated_reply_gets_one_larger_try(self):
        gw = _Gateway(
            replies=[("", FinishReason.MAX_TOKENS), ('{"scope": "x"}', FinishReason.END_TURN)]
        )
        runner = ModelBackedAnalysisRunner(gw, "test", max_output_tokens=1024)
        assert await runner._complete_json("prompt", {}) == {"scope": "x"}
        assert gw.budgets == [1024, 4096]

    @pytest.mark.asyncio
    async def test_a_finished_reply_that_said_something_is_not_retried(self):
        gw = _Gateway(replies=[("no json here", FinishReason.END_TURN)])
        runner = ModelBackedAnalysisRunner(gw, "test", max_output_tokens=1024)
        assert await runner._complete_json("prompt", {}) is None
        assert gw.budgets == [1024]

    @pytest.mark.asyncio
    async def test_an_empty_end_turn_is_retried_too(self):
        """The same models that truncate also return a bare ``end_turn`` with no
        content blocks. An empty reply carries no refusal either way.
        """
        gw = _Gateway(
            replies=[("", FinishReason.END_TURN), ('{"scope": "x"}', FinishReason.END_TURN)]
        )
        runner = ModelBackedAnalysisRunner(gw, "test", max_output_tokens=1024)
        assert await runner._complete_json("prompt", {}) == {"scope": "x"}
        assert gw.budgets == [1024, 4096]

    @pytest.mark.asyncio
    async def test_two_truncations_stop_rather_than_loop(self):
        gw = _Gateway(replies=[("", FinishReason.MAX_TOKENS)])
        runner = ModelBackedAnalysisRunner(gw, "test", max_output_tokens=512)
        assert await runner._complete_json("prompt", {}) is None
        assert gw.budgets == [512, 2048]


class TestUngroundedContentChecks:
    GOAL = (
        "Add a low_stock(threshold) method to the Store class in inventory/store.py. "
        "It must return the sorted list of SKUs whose held quantity is strictly less "
        "than the given threshold, and it must raise ValueError if threshold is not "
        "positive. Also add tests for the new method in a new file "
        "tests/test_low_stock.py covering the normal case, the boundary where quantity "
        "equals the threshold, and the ValueError."
    )

    @staticmethod
    def _fd(path: str, needle: str | None) -> dict:
        return {
            "kind": "file_diff",
            "description": "",
            "parameters": {"path": path, "must_exist": True, "content_contains": needle},
        }

    def test_an_invented_test_function_name_is_not_made_a_hard_gate(self, tmp_path: Path):
        """Run 18: the analyzer turned 'covering the normal case' into a literal
        content_contains on 'test_normal_case' and hard-failed a worker whose
        equivalent test was named test_low_stock_normal_case."""
        criteria = [
            self._fd("tests/test_low_stock.py", "test_low_stock"),
            self._fd("tests/test_low_stock.py", "test_normal_case"),
            self._fd("tests/test_low_stock.py", "test_quantity_equals_threshold"),
        ]

        kept = _drop_ungrounded_content_checks(criteria, self.GOAL, tmp_path)

        assert [c["parameters"].get("content_contains") for c in kept] == ["test_low_stock"]

    def test_needles_grounded_in_the_target_file_survive(self, tmp_path: Path):
        (tmp_path / "inventory").mkdir()
        (tmp_path / "inventory" / "store.py").write_text("class Store:\n    def add(self): ...\n")
        criteria = [
            self._fd("inventory/store.py", "def low_stock(self, threshold)"),
            self._fd("inventory/store.py", "raise ValueError"),
            self._fd("inventory/store.py", "sorted("),
        ]

        kept = _drop_ungrounded_content_checks(criteria, self.GOAL, tmp_path)

        assert len(kept) == 3
        assert all(c["parameters"]["content_contains"] for c in kept)

    def test_a_lone_ungrounded_criterion_keeps_its_existence_check(self, tmp_path: Path):
        criteria = [self._fd("tests/test_low_stock.py", "test_normal_case")]

        kept = _drop_ungrounded_content_checks(criteria, self.GOAL, tmp_path)

        assert len(kept) == 1
        assert "content_contains" not in kept[0]["parameters"]
        assert kept[0]["parameters"]["must_exist"] is True

    def test_non_file_diff_criteria_pass_through_untouched(self, tmp_path: Path):
        criteria = [{"kind": "command", "description": "", "parameters": {"command_name": "test"}}]

        assert _drop_ungrounded_content_checks(criteria, self.GOAL, tmp_path) == criteria


class TestTestGoalCriterionUpgrade:
    @staticmethod
    def _soft_rubric_analysis() -> AcceptanceAnalysis:
        return AcceptanceAnalysis(
            criteria=[{"kind": "soft_rubric", "description": "default quality rubric"}]
        )

    @staticmethod
    def _config() -> NorthStackConfig:
        return NorthStackConfig(
            name="t",
            commands=[CommandConfig(name="test", argv=["python", "-m", "pytest", "tests/", "-q"])],
        )

    def test_appends_hard_criterion_for_test_goal(self, tmp_path: Path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")
        request = ProjectRequest(
            goal="Fix add() so the full pytest suite in tests/ passes.",
            workspace_root=str(tmp_path),
        )

        upgraded = ContractCompiler._upgrade_test_goal_criterion(
            request, self._config(), self._soft_rubric_analysis()
        )

        by_kind = [c["kind"] for c in upgraded.criteria]
        assert by_kind[0] == "soft_rubric"
        assert by_kind.count("command") == 1
        command = next(c for c in upgraded.criteria if c["kind"] == "command")
        assert command["parameters"] == {"command_name": "test", "exit_code": 0}

        trees = [c for c in upgraded.criteria if c["kind"] == "tree_digest"]
        assert len(trees) == 1
        assert trees[0]["parameters"]["path"] == "tests"
        assert len(trees[0]["parameters"]["tree_hash"]) == 64

        guards = [
            c
            for c in upgraded.criteria
            if c["kind"] == "file_diff" and c["parameters"].get("must_exist") is False
        ]
        guard_paths = {g["parameters"]["path"] for g in guards}
        assert "conftest.py" in guard_paths

    @pytest.mark.parametrize(
        "goal",
        [
            "Fix the defect in ledger/accounts.py so the suite passes.",
            "The suite fails; diagnose it and make the tests pass.",
            "'python -m pytest tests/ -q' exits non-zero. Make the tests pass.",
        ],
    )
    def test_a_filename_in_the_goal_does_not_drop_the_tamper_pin(self, tmp_path: Path, goal: str):
        """A dot inside a path must not end the phrase-match window."""
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")
        upgraded = ContractCompiler._upgrade_test_goal_criterion(
            ProjectRequest(goal=goal, workspace_root=str(tmp_path)),
            self._config(),
            self._soft_rubric_analysis(),
        )
        assert [c["kind"] for c in upgraded.criteria].count("tree_digest") == 1

    @pytest.mark.parametrize(
        "goal",
        [
            "Add unit tests for the parser.",
            "Rewrite the tests. The build must pass.",
            "Write a password reset flow.",
        ],
    )
    def test_a_goal_that_is_not_about_passing_tests_is_left_alone(self, tmp_path: Path, goal: str):
        (tmp_path / "tests").mkdir()
        analysis = self._soft_rubric_analysis()
        upgraded = ContractCompiler._upgrade_test_goal_criterion(
            ProjectRequest(goal=goal, workspace_root=str(tmp_path)), self._config(), analysis
        )
        assert upgraded.criteria == analysis.criteria

    def test_a_goal_that_asks_for_a_new_test_file_is_not_made_unsatisfiable(self, tmp_path: Path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")
        request = ProjectRequest(
            goal=(
                "Add low_stock() to store.py and add tests for it in a new file "
                "tests/test_low_stock.py. The full existing test suite must still pass."
            ),
            workspace_root=str(tmp_path),
        )
        analysis = AcceptanceAnalysis(
            criteria=[
                {
                    "kind": "file_diff",
                    "description": "the new test file exists",
                    "parameters": {"path": "tests/test_low_stock.py", "must_exist": True},
                }
            ]
        )

        upgraded = ContractCompiler._upgrade_test_goal_criterion(request, self._config(), analysis)

        # A tree pin would forbid the very file another criterion demands.
        assert not any(c["kind"] == "tree_digest" for c in upgraded.criteria)
        pins = {
            c["parameters"]["path"]: c["parameters"]
            for c in upgraded.criteria
            if c["kind"] == "file_diff"
        }
        assert pins["tests/test_x.py"]["must_exist"] is True
        assert "content_hash" in pins["tests/test_x.py"]
        assert pins["tests/conftest.py"]["must_exist"] is False
        assert pins["tests/test_low_stock.py"]["must_exist"] is True

    def test_a_goal_that_only_reads_tests_keeps_the_whole_tree_pin(self, tmp_path: Path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")
        request = ProjectRequest(
            goal="Fix add() so the full pytest suite in tests/ passes.",
            workspace_root=str(tmp_path),
        )

        upgraded = ContractCompiler._upgrade_test_goal_criterion(
            request, self._config(), self._soft_rubric_analysis()
        )

        assert sum(c["kind"] == "tree_digest" for c in upgraded.criteria) == 1

    def test_is_idempotent(self, tmp_path: Path):
        (tmp_path / "tests").mkdir()
        request = ProjectRequest(
            goal="Make the tests pass.",
            workspace_root=str(tmp_path),
        )
        once = ContractCompiler._upgrade_test_goal_criterion(
            request, self._config(), self._soft_rubric_analysis()
        )
        twice = ContractCompiler._upgrade_test_goal_criterion(request, self._config(), once)

        assert twice == once

    def test_tree_digest_changes_when_file_added(self, tmp_path: Path):
        from northstack.application.verification.hard_gates import compute_tree_digest

        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_x.py").write_text("x = 1\n")
        before = compute_tree_digest(tmp_path, "tests")

        (tests / "aaa_patch.py").write_text("import app\napp.add = lambda a, b: a + b\n")
        after = compute_tree_digest(tmp_path, "tests")

        assert before != after

    @staticmethod
    def _compiler() -> ContractCompiler:
        return ContractCompiler(analysis_runner=DeterministicAnalysisRunner())

    async def test_normalizes_tool_name_to_command_profile(self, tmp_path: Path):
        (tmp_path / "tests").mkdir()
        analysis = AcceptanceAnalysis(
            criteria=[
                {
                    "kind": "command",
                    "description": "suite passes",
                    "parameters": {"command_name": "pytest", "exit_code": 0},
                }
            ]
        )

        normalized = ContractCompiler._normalize_command_criteria(self._config(), analysis)

        assert normalized.criteria[0]["parameters"]["command_name"] == "test"

    async def test_drops_command_criterion_without_any_pytest_profile(self):
        config = NorthStackConfig(
            name="t",
            commands=[CommandConfig(name="lint", argv=["python", "-m", "ruff", "check", "."])],
        )
        analysis = AcceptanceAnalysis(
            criteria=[
                {
                    "kind": "command",
                    "description": "suite passes",
                    "parameters": {"command_name": "pytest", "exit_code": 0},
                }
            ]
        )

        normalized = ContractCompiler._normalize_command_criteria(config, analysis)

        assert normalized.criteria == []

    def test_does_not_duplicate_command_criterion(self, tmp_path: Path):
        (tmp_path / "tests").mkdir()
        request = ProjectRequest(
            goal="Make the tests pass.",
            workspace_root=str(tmp_path),
        )
        analysis = AcceptanceAnalysis(
            criteria=[
                {"kind": "soft_rubric", "description": "default quality rubric"},
                {
                    "kind": "command",
                    "description": "already present",
                    "parameters": {"command_name": "test", "exit_code": 0},
                },
            ]
        )

        upgraded = ContractCompiler._upgrade_test_goal_criterion(request, self._config(), analysis)

        assert sum(c["kind"] == "command" for c in upgraded.criteria) == 1
        assert (
            sum(
                c["kind"] == "file_diff" and c["parameters"]["path"] == "conftest.py"
                for c in upgraded.criteria
            )
            == 1
        )

    def test_keeps_model_executable_criteria(self, tmp_path: Path):
        (tmp_path / "tests").mkdir()
        request = ProjectRequest(
            goal="Make the tests pass.",
            workspace_root=str(tmp_path),
        )
        analysis = AcceptanceAnalysis(
            criteria=[
                {
                    "kind": "command",
                    "description": "model-proposed",
                    "parameters": {"command_name": "test", "exit_code": 0},
                }
            ]
        )

        upgraded = ContractCompiler._upgrade_test_goal_criterion(request, self._config(), analysis)

        assert sum(c["kind"] == "command" for c in upgraded.criteria) == 1
        assert upgraded.criteria[0] is not None
        assert any(
            c["kind"] == "file_diff" and c["parameters"]["path"] == "conftest.py"
            for c in upgraded.criteria
        )

    def test_noop_when_goal_is_not_a_test_goal(self, tmp_path: Path):
        (tmp_path / "tests").mkdir()
        request = ProjectRequest(
            goal="Write a limerick about databases.",
            workspace_root=str(tmp_path),
        )
        analysis = self._soft_rubric_analysis()

        assert (
            ContractCompiler._upgrade_test_goal_criterion(request, self._config(), analysis)
            is analysis
        )

    def test_noop_without_pytest_command_profile(self, tmp_path: Path):
        (tmp_path / "tests").mkdir()
        request = ProjectRequest(
            goal="Make the tests pass.",
            workspace_root=str(tmp_path),
        )
        config = NorthStackConfig(
            name="t",
            commands=[CommandConfig(name="lint", argv=["python", "-m", "ruff", "check", "."])],
        )
        analysis = self._soft_rubric_analysis()

        assert ContractCompiler._upgrade_test_goal_criterion(request, config, analysis) is analysis


class TestCriterionSalvage:
    """A model-authored criterion set must not be able to kill a run."""

    @staticmethod
    def _synthesize(criteria):
        from northstack.application.contracting import (
            DeterministicSynthesizer,
            RepoAnalysis,
            RequirementsAnalysis,
        )
        from northstack.domain import Budget

        return DeterministicSynthesizer().synthesize(
            ProjectRequest(goal="g", workspace_root="/w"),
            RequirementsAnalysis(deliverables=["d"]),
            RepoAnalysis(),
            AcceptanceAnalysis(criteria=criteria),
            Budget(token_limit=100, cost_limit_usd=0.0),
        )

    def test_an_invented_field_does_not_discard_the_criterion(self):
        contract = self._synthesize(
            [
                {
                    "kind": "soft_rubric",
                    "description": "quality",
                    "parameters": {"minimum_score": 0.8},
                }
            ]
        )
        assert [c.description for c in contract.acceptance_criteria] == ["quality"]

    def test_one_unparsable_criterion_does_not_lose_the_others(self, caplog):
        with caplog.at_level(logging.WARNING):
            contract = self._synthesize(
                [
                    {
                        "kind": "command",
                        "description": "tests",
                        "parameters": {"command_name": "test"},
                    },
                    {"kind": "not_a_kind", "description": "junk"},
                ]
            )
        assert [c.description for c in contract.acceptance_criteria] == ["tests"]
        assert "dropping unparsable acceptance criterion" in caplog.text

    def test_losing_every_criterion_falls_back_to_the_default_rubric(self):
        contract = self._synthesize([{"kind": "not_a_kind", "description": "junk"}])
        assert [c.description for c in contract.acceptance_criteria] == ["default quality rubric"]


class TestAcceptancePromptNamesConfiguredCommands:
    """The model cannot pick a valid command_name it was never shown."""

    @staticmethod
    def _prompt(command_names, tmp_path):
        captured: dict[str, str] = {}

        class _Capture:
            async def complete(self, request):
                captured["prompt"] = request.messages[0].content
                raise RuntimeError("stop after capture")

        runner = ModelBackedAnalysisRunner(_Capture(), "analysis", command_names=command_names)
        asyncio.run(
            runner.run_acceptance(
                ProjectRequest(goal="build a thing and test it", workspace_root=str(tmp_path)),
                None,
            )
        )
        return captured["prompt"]

    def test_configured_names_reach_the_prompt(self, tmp_path):
        prompt = self._prompt(["lint", "test", "format"], tmp_path)
        assert "one of 'lint', 'test', 'format'" in prompt
        assert "any other exit code is never useful" in prompt

    def test_no_configured_commands_keeps_the_generic_wording(self, tmp_path):
        assert "<a known command profile>" in self._prompt([], tmp_path)
