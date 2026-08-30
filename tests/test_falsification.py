"""Hermetic tests for the model-backed contract falsifier.

Pinned behavior: a concrete counter-interpretation REJECTS the contract at
compile time; "none"/empty means sound; gateway or parse failures fail OPEN
(an outage is not an objection); the wire request is schema-constrained and
carries the request, contract, and criteria.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from northstack.adapters.providers.wire import ModelRequest
from northstack.application.contracting import ContractCompiler, DeterministicAnalysisRunner
from northstack.application.falsification import ModelBackedFalsifier
from northstack.domain.budget import Budget
from northstack.domain.contract import FileDiffCriterion, WorkContract
from northstack.domain.request import ProjectRequest


class _Resp(BaseModel):
    text: str = ""


class _FakeGateway:
    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self._text = text
        self._error = error
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> _Resp:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return _Resp(text=self._text)


def _contract() -> WorkContract:
    return WorkContract(
        id="wc-f",
        objective="add a farewell method",
        deliverables=["greeter.py"],
        budget=Budget(token_limit=1000, cost_limit_usd=0.1),
        acceptance_criteria=[FileDiffCriterion(description="farewell exists", path="greeter.py")],
    )


def _request() -> ProjectRequest:
    return ProjectRequest(goal="add farewell", workspace_root=str(Path("/tmp/w")))


async def test_concrete_counter_interpretation_is_returned():
    falsifier = ModelBackedFalsifier(
        _FakeGateway('{"counter_interpretation": "an empty file passes the file_exists check"}'),
        "expert",
    )
    counter = await falsifier.check(_contract(), _request())
    assert counter is not None
    assert "empty file" in counter


async def test_none_verdict_means_sound():
    for text in ('{"counter_interpretation": "none"}', "none", ""):
        falsifier = ModelBackedFalsifier(_FakeGateway(text), "expert")
        assert await falsifier.check(_contract(), _request()) is None


async def test_plain_text_counter_is_accepted_leniently():
    falsifier = ModelBackedFalsifier(
        _FakeGateway("The criteria never check the method actually returns anything."),
        "expert",
    )
    counter = await falsifier.check(_contract(), _request())
    assert counter is not None
    assert "never check" in counter


async def test_fenced_json_counter_is_parsed():
    falsifier = ModelBackedFalsifier(
        _FakeGateway('```json\n{"counter_interpretation": "wrong file"}\n```'),
        "expert",
    )
    assert "wrong file" in (await falsifier.check(_contract(), _request()) or "")


async def test_gateway_failure_fails_open():
    falsifier = ModelBackedFalsifier(_FakeGateway(error=RuntimeError("down")), "expert")
    assert await falsifier.check(_contract(), _request()) is None


async def test_counter_interpretation_truncated_to_bound():
    falsifier = ModelBackedFalsifier(
        _FakeGateway('{"counter_interpretation": "%s"}' % ("x" * 5000)), "expert"
    )
    counter = await falsifier.check(_contract(), _request())
    assert counter is not None and len(counter) <= 500


async def test_wire_request_shape():
    gw = _FakeGateway('{"counter_interpretation": "none"}')
    falsifier = ModelBackedFalsifier(gw, "expert")
    await falsifier.check(_contract(), _request())
    request = gw.requests[0]
    assert request.profile_name == "expert"
    assert request.system and "passing-but-wrong" in request.system
    assert request.output_json_schema is not None
    body = request.messages[0].content
    assert "add farewell" in body
    assert "farewell exists" in body


async def test_compiler_rejects_contract_when_falsifier_finds_one(tmp_path: Path):
    """The falsifier finding is actionable: compilation raises, the run fails
    loudly rather than proceeding on a misread contract."""
    from northstack.adapters.sqlite_ledger import Ledger

    compiler = ContractCompiler(
        analysis_runner=DeterministicAnalysisRunner(),
        falsifier=ModelBackedFalsifier(
            _FakeGateway('{"counter_interpretation": "empty file passes"}'), "expert"
        ),
    )
    ledger = Ledger(path=tmp_path / "l.db")
    try:
        with pytest.raises(ValueError, match="Falsifier found counter-interpretation"):
            await compiler.compile(_request(), ledger, None, "run-f")
    finally:
        ledger.close()
