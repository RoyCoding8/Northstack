"""Hermetic tests for reviewer calibration measurement and loading.

No network: the reviewer panel's gateway is faked with verdicts scripted by
evidence content. Pinned behavior:

  - agreement / accuracy / false-acceptance / false-rejection math on labeled
    samples, including interleaved criterion indices;
  - the suggested threshold heuristic (max(0.5, 1 - FA rate));
  - the emitted records validate as CalibrationRecords and round-trip
    through load_calibration_records;
  - a malformed calibration file raises (never a silent empty panel);
  - build_company wiring: calibration_path feeds the panel; planner_mode and
    falsifier_mode route to model-backed implementations only when the
    matching role chain exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch

from northstack.adapters.providers.wire import FinishReason, ModelResponse, Usage
from northstack.application.build import build_company
from northstack.application.calibration import (
    CalibrationRunner,
    CalibrationSuite,
    load_calibration_records,
    load_calibration_suite,
)
from northstack.application.falsification import ModelBackedFalsifier
from northstack.application.planning_model import ModelBackedPlanner
from northstack.config import ModelProfile, NorthStackConfig, Protocol, Role, RouteMapping
from northstack.interfaces.cli import app

# Fixtures


def _sample(index: int, evidence: str, label: bool, objective: str = "ship it") -> dict:
    return {
        "criterion_index": index,
        "objective": objective,
        "description": "the deliverable is correct",
        "evidence_content": evidence,
        "label": label,
    }


def _verdict_gateway() -> Any:
    """A gateway whose two reviewer profiles verdict by evidence content.

    reviewer-a passes 'good' evidence and fails 'bad' evidence.
    reviewer-b passes 'good' evidence and ALSO passes 'bad' evidence (the
    weak reviewer the calibration is meant to expose).
    """

    class _Gateway:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: Any) -> ModelResponse:
            self.calls += 1
            profile = request.profile_name
            evidence = request.messages[0].content
            good = "GOOD-EVIDENCE" in evidence
            if profile == "reviewer-a":
                passed = good
            else:
                passed = True  # rubber stamp
            return ModelResponse(
                text=json.dumps({"passed": passed, "confidence": 0.9, "rationale": "r"}),
                finish_reason=FinishReason.END_TURN,
                usage=Usage(input_tokens=5, output_tokens=5),
                provider="openai",
                model="test",
            )

    return _Gateway()


# Measurement math


async def test_measurement_math_on_labeled_samples():
    # 4 samples on criterion 0 (2 good, 2 bad), interleaved with 2 samples on
    # criterion 1 -- interleaving must not corrupt the grouping.
    samples = [
        _sample(0, "GOOD-EVIDENCE-1", True),
        _sample(1, "GOOD-EVIDENCE-2", True),
        _sample(0, "BAD-EVIDENCE-1", False),
        _sample(0, "GOOD-EVIDENCE-3", True),
        _sample(1, "GOOD-EVIDENCE-4", True),
        _sample(0, "BAD-EVIDENCE-2", False),
    ]
    runner = CalibrationRunner(_verdict_gateway(), ["reviewer-a", "reviewer-b"])
    report = await runner.measure(CalibrationSuite(name="t", samples=samples))

    by_index = {m.criterion_index: m for m in report.per_criterion}
    m0 = by_index[0]
    # reviewer-a agrees with the label on all 4; reviewer-b disagrees on both
    # bad samples -> inter-reviewer agreement 2/4.
    assert m0.sample_count == 4
    assert m0.agreement_rate == pytest.approx(0.5)
    # Majority over a split 2-panel: a tie FAILS the criterion, so the
    # rubber-stamper's bad passes are rejected -> the majority matches the
    # label everywhere, and no false acceptance slips through.
    assert m0.majority_accuracy == pytest.approx(1.0)
    assert m0.false_acceptance_rate == pytest.approx(0.0)
    assert m0.false_rejection_rate == pytest.approx(0.0)
    # Threshold heuristic: max(0.5, 1 - 0.0) = 1.0.
    assert m0.suggested_threshold == pytest.approx(1.0)

    m1 = by_index[1]
    assert m1.sample_count == 2
    assert m1.agreement_rate == pytest.approx(1.0)
    assert m1.false_acceptance_rate == 0.0

    # Per-reviewer accuracy (pooled across BOTH criterion groups, 6 samples):
    # a matches every label; the rubber stamp matches only the 4 true ones.
    accuracies = {r.profile: r.accuracy for r in report.per_reviewer}
    assert accuracies["reviewer-a"] == pytest.approx(1.0)
    assert accuracies["reviewer-b"] == pytest.approx(4 / 6)


async def test_records_round_trip_through_load(tmp_path: Path):
    samples = [
        _sample(0, "GOOD-EVIDENCE-1", True),
        _sample(0, "BAD-EVIDENCE-1", False),
    ]
    runner = CalibrationRunner(_verdict_gateway(), ["reviewer-a", "reviewer-b"])
    report = await runner.measure(CalibrationSuite(name="rt", samples=samples))

    out = tmp_path / "calibration.json"
    out.write_text(json.dumps(report.to_json()), encoding="utf-8")
    loaded = load_calibration_records(out)
    assert len(loaded) == 1
    record = loaded[0]
    assert record.criterion_index == 0
    assert record.sample_count == 2
    assert record.min_reviewers == 2
    assert 0.0 <= record.agreement_threshold <= 1.0

    # The Markdown report renders the same numbers.
    md = report.to_markdown()
    assert "Reviewer calibration report" in md
    assert "reviewer-a" in md


def test_malformed_calibration_file_raises(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"records": [{"criterion_index": "zero"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        load_calibration_records(bad)

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"records": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no records"):
        load_calibration_records(empty)


def test_load_suite_rejects_unlabeled_samples(tmp_path: Path):
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps({"samples": [_sample(0, "e", True), {"no_label": 1}]}), encoding="utf-8"
    )
    with pytest.raises(Exception, match="label"):
        load_calibration_suite(p)


def test_runner_requires_reviewers():
    with pytest.raises(ValueError, match="at least one reviewer"):
        CalibrationRunner(_verdict_gateway(), [])


# build_company wiring


def _profiles() -> list[ModelProfile]:
    def profile(name: str, roles: list[Role]) -> ModelProfile:
        return ModelProfile(
            name=name,
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://localhost/v1",
            model=name,
            roles=roles,
            max_concurrency=2,
        )

    return [
        profile("cheap", [Role.WORKER]),
        profile("planner-mid", [Role.PLANNER]),
        profile("expert", [Role.SPECIALIST]),
    ]


def _routing() -> list[RouteMapping]:
    return [
        RouteMapping(role=Role.WORKER, profiles=["cheap"]),
        RouteMapping(role=Role.PLANNER, profiles=["planner-mid"]),
        RouteMapping(role=Role.SPECIALIST, profiles=["expert"]),
    ]


def _wired_config(**run_kw: Any) -> NorthStackConfig:
    from northstack.config import RunConfig

    return NorthStackConfig(
        name="wired",
        profiles=_profiles(),
        routing=_routing(),
        run=RunConfig(**run_kw),
    )


def test_build_wires_model_planner_when_configured(tmp_path: Path):
    with patch("northstack.application.build.ModelGateway", MagicMock()):
        components = build_company(_wired_config(planner_mode="model"), tmp_path)
    try:
        assert isinstance(components.company._planner, ModelBackedPlanner)
    finally:
        components.ledger.close()


def test_build_keeps_default_planner_without_model_mode(tmp_path: Path):
    with patch("northstack.application.build.ModelGateway", MagicMock()):
        components = build_company(_wired_config(), tmp_path)
    try:
        assert not isinstance(components.company._planner, ModelBackedPlanner)
    finally:
        components.ledger.close()


def test_build_model_planner_without_routed_profile_degrades(tmp_path: Path):
    config = NorthStackConfig(
        name="no-planner",
        profiles=_profiles()[:1],
        run=__import__("northstack.config", fromlist=["RunConfig"]).RunConfig(planner_mode="model"),
    )
    with patch("northstack.application.build.ModelGateway", MagicMock()):
        components = build_company(config, tmp_path)
    try:
        assert not isinstance(components.company._planner, ModelBackedPlanner)
    finally:
        components.ledger.close()


def test_build_wires_model_falsifier_when_configured(tmp_path: Path):
    with patch("northstack.application.build.ModelGateway", MagicMock()):
        components = build_company(_wired_config(falsifier_mode="model"), tmp_path)
    try:
        assert isinstance(components.company._compiler._falsifier, ModelBackedFalsifier)
    finally:
        components.ledger.close()


def test_build_loads_calibration_records(tmp_path: Path):
    records = [
        {
            "criterion_index": 0,
            "reviewer_agreement_rate": 0.9,
            "sample_count": 20,
            "min_reviewers": 2,
            "agreement_threshold": 0.8,
        }
    ]
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps({"records": records}), encoding="utf-8")

    with patch("northstack.application.build.ModelGateway", MagicMock()):
        components = build_company(_wired_config(calibration_path=str(cal)), tmp_path)
    try:
        checker = components.company._soft_checker
        assert 0 in checker._calibration
        assert checker._calibration[0].agreement_threshold == 0.8
    finally:
        components.ledger.close()


def test_build_rejects_broken_calibration_file(tmp_path: Path):
    cal = tmp_path / "calibration.json"
    cal.write_text("{not json", encoding="utf-8")
    with (
        patch("northstack.application.build.ModelGateway", MagicMock()),
        patch("northstack.application.build.Ledger") as ledger,
        pytest.raises(Exception),
    ):
        build_company(_wired_config(calibration_path=str(cal)), tmp_path)
    ledger.assert_not_called()


# CLI guard


def test_calibrate_cli_requires_two_routed_reviewers(tmp_path: Path):
    config = tmp_path / "c.toml"
    config.write_text(
        '[northstack]\nname = "x"\n\n[[northstack.profiles]]\n'
        'name = "p"\nprotocol = "openai_chat"\nbase_url = "http://localhost/v1"\n'
        'model = "m"\nroles = ["worker"]\nmax_concurrency = 2\n',
        encoding="utf-8",
    )
    samples = tmp_path / "s.json"
    samples.write_text(json.dumps({"samples": [_sample(0, "e", True)]}), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "calibrate",
            "--config",
            str(config),
            "--samples",
            str(samples),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 1
    assert "two routed reviewer" in result.output
