"""Tests for the model-backed analysis runner (option #3).

The runner's job: turn a goal into concrete, hard-checkable acceptance criteria
(command / file_diff) so a run can reach ``verified`` purely from hard gates.
When it cannot produce executable criteria, it must fall back to a single
``soft_rubric`` criterion -- preserving the abstention law rather than faking
confidence.
"""

from __future__ import annotations

import json

import pytest

from northstack.adapters.providers.wire import FinishReason, ModelRequest, ModelResponse, Usage
from northstack.application.contracting import ContractCompiler, ModelBackedAnalysisRunner
from northstack.config import ModelProfile, Protocol
from northstack.domain import Budget, CriterionKind, ProjectRequest

# Fakes


class FakeGateway:
    """Captures the last request and returns a canned ModelResponse."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.last_request: ModelRequest | None = None

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.last_request = request
        return ModelResponse(
            text=self._text,
            finish_reason=FinishReason.END_TURN,
            usage=Usage(input_tokens=10, output_tokens=5),
            provider="openai",
            model="test-model",
        )


class FailingGateway:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise RuntimeError("provider down")


class FailingButAssertingGateway:
    """Like FailingGateway but raises AssertionError if the model is ever
    called -- used to prove the deterministic shortcut bypassed the gateway."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("model must not be called for a heuristic goal")


def _profile() -> ModelProfile:
    return ModelProfile(
        name="cheap-worker",
        protocol=Protocol.OPENAI_CHAT,
        base_url="http://localhost/v1",
        model="test-model",
        max_concurrency=1,
    )


def _request(goal: str) -> ProjectRequest:
    return ProjectRequest(
        goal=goal,
        workspace_root="/tmp/ws",
        budget=Budget(token_limit=100_000, cost_limit_usd=1.0),
    )


# Acceptance analysis: emits hard criteria


class TestRunAcceptance:
    @pytest.mark.asyncio
    async def test_emits_file_diff_criteria_from_model_json(self):
        payload = {
            "criteria": [
                {
                    "kind": "file_diff",
                    "description": "hello.txt exists with required text",
                    "parameters": {
                        "path": "hello.txt",
                        "must_exist": True,
                        "content_contains": "NorthStack ran OK",
                    },
                }
            ],
            "risks": [],
            "ambiguities": [],
            "recommended_abstention_threshold": 0.5,
        }
        gw = FakeGateway(json.dumps(payload))
        runner = ModelBackedAnalysisRunner(gw, "cheap-worker")

        # The goal names the required text, so the content check is grounded and
        # survives the ungrounded-needle filter.
        goal = "Create hello.txt that reports NorthStack ran OK"

        acc = await runner.run_acceptance(_request(goal), _profile())

        assert len(acc.criteria) == 1
        c = acc.criteria[0]
        assert c["kind"] == CriterionKind.FILE_DIFF.value
        assert c["parameters"]["content_contains"] == "NorthStack ran OK"
        # No soft_rubric criterion present -> hard gates alone can verify.
        assert not any(c["kind"] == CriterionKind.SOFT_RUBRIC.value for c in acc.criteria)

    @pytest.mark.asyncio
    async def test_strips_json_fences_and_prose(self):
        # Model wraps JSON in a code fence and prepends prose.
        text = (
            'Here you go:\n```json\n{"criteria": [{"kind": "file_diff", '
            '"description": "x", "parameters": {"path": "a.txt", '
            '"must_exist": true}}], "risks": [], "ambiguities": []}\n```'
        )
        gw = FakeGateway(text)
        runner = ModelBackedAnalysisRunner(gw, "cheap-worker")

        acc = await runner.run_acceptance(_request("make a.txt"), _profile())

        assert len(acc.criteria) == 1
        assert acc.criteria[0]["kind"] == CriterionKind.FILE_DIFF.value

    @pytest.mark.asyncio
    async def test_drops_unknown_criterion_kinds(self):
        payload = {
            "criteria": [
                {"kind": "file_diff", "description": "ok", "parameters": {"path": "a"}},
                {"kind": "bogus_kind", "description": "no", "parameters": {}},
                {"kind": "command", "description": "no params dict", "parameters": "x"},
            ],
            "risks": [],
            "ambiguities": [],
        }
        gw = FakeGateway(json.dumps(payload))
        runner = ModelBackedAnalysisRunner(gw, "cheap-worker")

        acc = await runner.run_acceptance(_request("x"), _profile())

        # Only the valid file_diff survives.
        assert len(acc.criteria) == 1
        assert acc.criteria[0]["kind"] == CriterionKind.FILE_DIFF.value

    @pytest.mark.asyncio
    async def test_falls_back_to_soft_rubric_on_gateway_failure(self):
        runner = ModelBackedAnalysisRunner(FailingGateway(), "cheap-worker")

        acc = await runner.run_acceptance(_request("do something vague"), _profile())

        # Must NOT silently pass; must produce a soft_rubric so the run abstains.
        assert len(acc.criteria) == 1
        assert acc.criteria[0]["kind"] == CriterionKind.SOFT_RUBRIC.value

    @pytest.mark.asyncio
    async def test_falls_back_to_soft_rubric_on_unparseable_text(self):
        gw = FakeGateway("this is not json at all { incomplete")
        runner = ModelBackedAnalysisRunner(gw, "cheap-worker")

        acc = await runner.run_acceptance(_request("vague"), _profile())

        assert len(acc.criteria) == 1
        assert acc.criteria[0]["kind"] == CriterionKind.SOFT_RUBRIC.value

    @pytest.mark.asyncio
    async def test_falls_back_to_soft_rubric_on_empty_criteria(self):
        gw = FakeGateway(json.dumps({"criteria": [], "risks": [], "ambiguities": []}))
        runner = ModelBackedAnalysisRunner(gw, "cheap-worker")

        acc = await runner.run_acceptance(_request("vague"), _profile())

        assert len(acc.criteria) == 1
        assert acc.criteria[0]["kind"] == CriterionKind.SOFT_RUBRIC.value

    @pytest.mark.asyncio
    async def test_requests_structured_output_schema(self):
        payload = {"criteria": [], "risks": [], "ambiguities": []}
        gw = FakeGateway(json.dumps(payload))
        runner = ModelBackedAnalysisRunner(gw, "cheap-worker")

        await runner.run_acceptance(_request("x"), _profile())

        assert gw.last_request is not None
        assert gw.last_request.output_json_schema is not None
        assert gw.last_request.temperature == 0.0
        assert gw.last_request.profile_name == "cheap-worker"


# End-to-end: runner -> compiler -> contract with only hard criteria


class TestRunnerToContract:
    @pytest.mark.asyncio
    async def test_contract_has_only_hard_criteria_no_soft_rubric(self):
        """A run whose analyzer emits a file_diff criterion must compile to a
        contract with NO soft_rubric criterion -- so verification can reach
        ``verified`` from hard gates alone (no abstention)."""
        payload = {
            "criteria": [
                {
                    "kind": "file_diff",
                    "description": "hello.txt contains the required text",
                    "parameters": {
                        "path": "hello.txt",
                        "must_exist": True,
                        "content_contains": "NorthStack ran OK",
                    },
                }
            ],
            "risks": [],
            "ambiguities": [],
            "recommended_abstention_threshold": 0.5,
        }
        payload_req = {
            "scope": "create a file",
            "deliverables": ["hello.txt"],
            "constraints": [],
            "assumptions": [],
            "ambiguities": [],
        }

        class DualGateway:
            def __init__(self):
                self.n = 0

            async def complete(self, request: ModelRequest) -> ModelResponse:
                self.n += 1
                # First call is requirements, later calls are acceptance.
                text = json.dumps(payload_req) if self.n == 1 else json.dumps(payload)
                return ModelResponse(
                    text=text,
                    finish_reason=FinishReason.END_TURN,
                    usage=Usage(input_tokens=10, output_tokens=5),
                    provider="openai",
                    model="test-model",
                )

        gw = DualGateway()
        runner = ModelBackedAnalysisRunner(gw, "cheap-worker")
        compiler = ContractCompiler(analysis_runner=runner)

        request = _request("Create a file called hello.txt containing 'NorthStack ran OK'.")
        contract = await compiler.compile(request)

        kinds = {c.kind for c in contract.acceptance_criteria}
        assert CriterionKind.FILE_DIFF.value in kinds
        assert CriterionKind.SOFT_RUBRIC.value not in kinds
        # The file_diff criterion carries the content_contains field.
        fd = next(c for c in contract.acceptance_criteria if c.kind == "file_diff")
        assert fd.content_contains == "NorthStack ran OK"


# Deterministic file-content heuristic


class TestDeterministicFileContentHeuristic:
    @pytest.mark.asyncio
    async def test_no_model_call_for_explicit_file_content_goal(self):
        """An explicit 'create file X with text Y' goal must produce a
        file_diff criterion WITHOUT calling the model -- the gateway is never
        touched. This is the routing law: no model judgment required."""

        class ExplodingGateway:
            async def complete(self, request: ModelRequest) -> ModelResponse:
                raise AssertionError("model must not be called for this goal")

        runner = ModelBackedAnalysisRunner(ExplodingGateway(), "cheap-worker")

        acc = await runner.run_acceptance(
            _request(
                "Create a file called hello.txt containing the text 'NorthStack ran OK'. Then stop."
            ),
            _profile(),
        )

        assert len(acc.criteria) == 1
        c = acc.criteria[0]
        assert c["kind"] == CriterionKind.FILE_DIFF.value
        assert c["parameters"]["path"] == "hello.txt"
        assert c["parameters"]["must_exist"] is True
        assert c["parameters"]["content_contains"] == "NorthStack ran OK"

    @pytest.mark.asyncio
    async def test_requirements_also_uses_heuristic(self):
        class ExplodingGateway:
            async def complete(self, request: ModelRequest) -> ModelResponse:
                raise AssertionError("model must not be called for requirements")

        runner = ModelBackedAnalysisRunner(ExplodingGateway(), "cheap-worker")

        req = await runner.run_requirements(
            _request("Create a file called notes.txt containing the text 'hello'."),
            _profile(),
        )

        assert req.deliverables == ["notes.txt"]

    @pytest.mark.asyncio
    async def test_non_matching_goal_still_uses_model(self):
        """A goal the heuristic cannot parse must fall through to the model."""

        class CountingGateway:
            def __init__(self):
                self.n = 0

            async def complete(self, request: ModelRequest) -> ModelResponse:
                self.n += 1
                return ModelResponse(
                    text=json.dumps(
                        {
                            "criteria": [
                                {
                                    "kind": "soft_rubric",
                                    "description": "vague",
                                    "parameters": {},
                                }
                            ],
                            "risks": [],
                            "ambiguities": [],
                        }
                    ),
                    finish_reason=FinishReason.END_TURN,
                    usage=Usage(input_tokens=1, output_tokens=1),
                    provider="openai",
                    model="test-model",
                )

        gw = CountingGateway()
        runner = ModelBackedAnalysisRunner(gw, "cheap-worker")

        acc = await runner.run_acceptance(
            _request("Refactor the authentication module to use OAuth."),
            _profile(),
        )

        assert gw.n == 1  # the model was consulted
        assert acc.criteria[0]["kind"] == CriterionKind.SOFT_RUBRIC.value

    @pytest.mark.parametrize(
        "goal",
        [
            "Create a file called hello.txt containing the text 'NorthStack ran OK'. Then stop.",
            'write a file named out.txt with the content "done"',
            "Make a file called data.json containing 'val'",
        ],
    )
    @pytest.mark.asyncio
    async def test_heuristic_matches_variants(self, goal: str):
        runner = ModelBackedAnalysisRunner(FailingButAssertingGateway(), "cheap-worker")
        acc = await runner.run_acceptance(_request(goal), _profile())
        assert len(acc.criteria) == 1
        assert acc.criteria[0]["kind"] == CriterionKind.FILE_DIFF.value
        assert "content_contains" in acc.criteria[0]["parameters"]
