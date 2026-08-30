"""Reviewer calibration: measure agreement and accuracy on labeled samples.

Feeds the soft-review law's calibration gate (ADR 0006): soft rubrics may
only verify under a ``CalibrationRecord`` whose thresholds were *measured*,
not guessed. This module is that measurement.

Method (pinned by the experiment protocol):

  - samples carry ground truth (``label`` = the evidence truly satisfies the
    criterion) and are separate from evaluation tasks;
  - every reviewer profile in the operator's ``reviewer`` chain reviews every
    sample through the production :class:`ModelBackedReviewer` path;
  - per criterion index we report sample count, inter-reviewer agreement
    rate, majority-vs-label accuracy, false acceptance (majority passes bad
    evidence) and false rejection (majority fails good evidence);
  - the suggested ``agreement_threshold`` is ``max(0.5, 1 - false
    acceptance rate)``: a panel that falsely accepts 10% of bad evidence must
    then agree on at least 90% of what it accepts. It is a heuristic floor
    for the operator to inspect, not a law.

``northstack calibrate`` writes ``calibration.json`` (records +
measurements) and a Markdown report. Setting
``[northstack.run] calibration_path`` to the JSON makes ``build_company``
load the records into the soft-review panel; a malformed file is a hard
error, never a silent uncalibrated panel.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from northstack.application.verification.model_review import ModelBackedReviewer
from northstack.domain.outcome import CalibrationRecord

logger = logging.getLogger(__name__)


class CalibrationSample(BaseModel):
    """One labeled judgment task for the reviewer panel."""

    model_config = ConfigDict(frozen=True)

    criterion_index: int = Field(ge=0)
    objective: str = Field(min_length=1)
    description: str = Field(min_length=1)
    evidence_content: str = Field(min_length=1)
    label: bool = Field(description="Ground truth: does the evidence satisfy the criterion?")


class CalibrationSuite(BaseModel):
    """A named collection of labeled samples."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(default="calibration", min_length=1)
    samples: list[CalibrationSample] = Field(min_length=1)


class ReviewerMeasurement(BaseModel):
    """Per-reviewer accuracy against the labels."""

    model_config = ConfigDict(frozen=True)

    profile: str
    samples: int = Field(ge=0)
    agreements_with_label: int = Field(ge=0)

    @property
    def accuracy(self) -> float:
        return self.agreements_with_label / self.samples if self.samples else 0.0


class CriterionMeasurement(BaseModel):
    """Aggregate panel behavior for one criterion index."""

    model_config = ConfigDict(frozen=True)

    criterion_index: int = Field(ge=0)
    sample_count: int = Field(ge=1)
    agreement_rate: float = Field(ge=0.0, le=1.0)
    majority_accuracy: float = Field(ge=0.0, le=1.0)
    false_acceptance_rate: float = Field(ge=0.0, le=1.0)
    false_rejection_rate: float = Field(ge=0.0, le=1.0)
    suggested_threshold: float = Field(ge=0.0, le=1.0)

    def to_record(self, min_reviewers: int) -> CalibrationRecord:
        """The measured CalibrationRecord the soft-review law consumes."""
        return CalibrationRecord(
            criterion_index=self.criterion_index,
            reviewer_agreement_rate=self.agreement_rate,
            sample_count=self.sample_count,
            min_reviewers=min_reviewers,
            agreement_threshold=self.suggested_threshold,
        )


class CalibrationReport(BaseModel):
    """Complete measurement output; serialized as calibration.json."""

    model_config = ConfigDict(frozen=True)

    suite_name: str
    reviewer_profiles: list[str]
    generated_at: float
    per_reviewer: list[ReviewerMeasurement]
    per_criterion: list[CriterionMeasurement]

    def records(self) -> list[CalibrationRecord]:
        min_reviewers = len(self.reviewer_profiles)
        return [m.to_record(min_reviewers) for m in self.per_criterion]

    def to_json(self) -> dict[str, Any]:
        """The persisted shape: measurements plus ready-to-load records."""
        return {
            "suite_name": self.suite_name,
            "reviewer_profiles": list(self.reviewer_profiles),
            "generated_at": self.generated_at,
            "per_reviewer": [r.model_dump() for r in self.per_reviewer],
            "per_criterion": [m.model_dump() for m in self.per_criterion],
            "records": [r.model_dump() for r in self.records()],
        }

    def to_markdown(self) -> str:
        header = (
            "| Criterion | Samples | Agreement | Majority accuracy | "
            "False accept | False reject | Suggested threshold |"
        )
        lines = [
            "# Reviewer calibration report",
            "",
            f"Suite: {self.suite_name} · Reviewers: {', '.join(self.reviewer_profiles)}",
            "",
            header,
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for m in self.per_criterion:
            lines.append(
                f"| {m.criterion_index} | {m.sample_count} | {m.agreement_rate:.3f} | "
                f"{m.majority_accuracy:.3f} | {m.false_acceptance_rate:.3f} | "
                f"{m.false_rejection_rate:.3f} | {m.suggested_threshold:.3f} |"
            )
        lines.extend(["", "## Per-reviewer accuracy", ""])
        for r in self.per_reviewer:
            lines.append(f"- **{r.profile}**: {r.accuracy:.3f} over {r.samples} samples")
        lines.extend(
            [
                "",
                "Suggested threshold = max(0.5, 1 - false acceptance rate). Inspect before",
                "adopting: it is a floor derived from this suite, not a law.",
            ]
        )
        return "\n".join(lines) + "\n"


def load_calibration_suite(path: Path) -> CalibrationSuite:
    """Load a samples file; malformed input raises (never silently empty)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CalibrationSuite.model_validate(data)


def load_calibration_records(path: Path) -> list[CalibrationRecord]:
    """Load records from a ``northstack calibrate`` output file.

    Every record validates through :class:`CalibrationRecord`; a malformed
    file raises so the operator learns their calibration is broken instead of
    running an accidentally uncalibrated panel.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"calibration file {path} carries no records")
    adapter: TypeAdapter[CalibrationRecord] = TypeAdapter(CalibrationRecord)
    records = []
    for i, raw in enumerate(raw_records):
        try:
            records.append(adapter.validate_python(raw))
        except ValidationError as exc:
            raise ValueError(f"calibration record {i} in {path} is invalid: {exc}") from exc
    return records


class CalibrationRunner:
    """Measures a reviewer panel against labeled samples."""

    def __init__(self, gateway: Any, reviewer_profiles: list[str]) -> None:
        if not reviewer_profiles:
            raise ValueError("calibration requires at least one reviewer profile")
        self._gateway = gateway
        self._reviewers = [ModelBackedReviewer(gateway, name) for name in reviewer_profiles]

    async def measure(self, suite: CalibrationSuite) -> CalibrationReport:
        per_reviewer = {r.profile_name: [0, 0] for r in self._reviewers}  # hits, total
        samples_by_criterion: dict[int, list[CalibrationSample]] = {}
        panels_by_criterion: dict[int, list[list[bool]]] = {}

        for sample in suite.samples:
            samples_by_criterion.setdefault(sample.criterion_index, []).append(sample)
            panel: list[bool] = []
            for reviewer in self._reviewers:
                verdict = await reviewer.review(
                    criterion_index=sample.criterion_index,
                    criterion_kind="soft_rubric",
                    description=sample.description,
                    objective=sample.objective,
                    evidence_content=sample.evidence_content,
                )
                panel.append(verdict.passed)
                hits, total = per_reviewer[reviewer.profile_name]
                per_reviewer[reviewer.profile_name] = [
                    hits + int(verdict.passed == sample.label),
                    total + 1,
                ]
            panels_by_criterion.setdefault(sample.criterion_index, []).append(panel)

        per_criterion: list[CriterionMeasurement] = []
        for criterion_index, samples in sorted(samples_by_criterion.items()):
            panels = panels_by_criterion[criterion_index]
            agreement_hits = sum(1 for panel in panels if len(set(panel)) == 1)
            majorities = [sum(panel) * 2 > len(panel) for panel in panels]
            accuracy_hits = sum(
                1
                for majority, sample in zip(majorities, samples, strict=True)
                if majority == sample.label
            )
            negatives = sum(1 for sample in samples if not sample.label)
            positives = len(samples) - negatives
            false_accepts = sum(
                1
                for majority, sample in zip(majorities, samples, strict=True)
                if majority and not sample.label
            )
            false_rejects = sum(
                1
                for majority, sample in zip(majorities, samples, strict=True)
                if (not majority) and sample.label
            )
            n = len(samples)
            fa_rate = false_accepts / negatives if negatives else 0.0
            fr_rate = false_rejects / positives if positives else 0.0
            per_criterion.append(
                CriterionMeasurement(
                    criterion_index=criterion_index,
                    sample_count=n,
                    agreement_rate=agreement_hits / n,
                    majority_accuracy=accuracy_hits / n,
                    false_acceptance_rate=fa_rate,
                    false_rejection_rate=fr_rate,
                    suggested_threshold=max(0.5, round(1.0 - fa_rate, 4)),
                )
            )

        return CalibrationReport(
            suite_name=suite.name,
            reviewer_profiles=[r.profile_name for r in self._reviewers],
            generated_at=time.time(),
            per_reviewer=[
                ReviewerMeasurement(
                    profile=name,
                    samples=totals[1],
                    agreements_with_label=totals[0],
                )
                for name, totals in per_reviewer.items()
            ],
            per_criterion=per_criterion,
        )
