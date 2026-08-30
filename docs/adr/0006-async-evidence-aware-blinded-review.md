# ADR 0006: Async, Evidence-Aware Blinded Review

Accepted.
## Context

The soft-rubric review seam could not be backed by a model, for two structural reasons:

1. `BlindedReviewer.review()` was synchronous. A model call is async through the gateway; there was no way to plug one in without violating the interface.
2. Reviewers received SHA-256 digests, not evidence. A digest is a content-address, not content: even a hypothetical model reviewer could not have judged the work, because the thing to judge never reached it.

Meanwhile the config already carried a `reviewer` role and routing chains — the design had anticipated a reviewer panel that the verification layer could not actually accept.

## Decision

The reviewer protocol is async and receives resolved evidence content:

```python
class BlindedReviewer(Protocol):
    async def review(
        self,
        *,
        criterion_index: int,
        criterion_kind: str,
        description: str,
        objective: str,
        evidence_content: str,
    ) -> ReviewVerdict: ...
```

`SoftRubricChecker.check()` is async; the orchestrator awaits it and resolves each criterion's artifact to text via the artifact store it already owns (blocking reads off the event loop, as the hard gates do). A missing artifact resolves to the empty string with a logged warning — the reviewer then judges absent evidence and fails closed through its own verdict, so the abstention law is preserved and the gap stays visible.

Blinding is structural, not procedural. The reviewer's input signature contains objective, criterion, and evidence content. Executor identity, profile names, tool trails, and conversation history do not exist in the signature, so they cannot leak into a verdict — there is nothing to remember to withhold.

`ModelBackedReviewer` (`application/verification/model_review.py`) is the production reviewer:

- Pinned to one explicit profile; the operator's `reviewer` routing chain supplies the panel (`build.py` takes the first two distinct profiles — fewer than two leaves the panel empty and soft rubrics abstain by law).
- The evidence is fenced and labelled data; the system prompt instructs the model to treat in-evidence instructions as artifact content. Prompt injection cannot flip a verdict by claiming success.
- Output is schema-constrained (`passed`, `confidence`, `rationale`) and parsed leniently for endpoints without native JSON mode.
- Every failure — gateway error, unparseable text, out-of-range verdict — becomes `ReviewVerdict(passed=False)`. A broken reviewer produces disagreement (abstention), never a false verification and never an exception out of verification.

The reviewer's rationale persists only as a digest, so the ledger stays bounded and chain-of-thought is not stored verbatim.

## Why

- The release law is untouched: it still requires ≥2 reviewers, calibration, and agreement. The change widens who may review, not what acceptance means.
- Fail-closed on every reviewer error path keeps the imperfect-verifier honesty the law was designed around: an uncalibrated or unavailable reviewer abstains rather than guessing.
- `ReviewVerdict` carrying `confidence` gives the future calibration study a signal to measure without changing the interface again.

## Consequences

- `DeterministicReviewer` (the test seam) became async; tests updated to await `check()`.
- `RepoAnalysis` grew an optional `scan_digest` field so intake evidence stays reproducible (see the intake scan notes in `application/intake_scan.py`).
- Operators who want soft rubrics to be able to verify must configure two reviewer profiles; a single-profile panel intentionally abstains.
