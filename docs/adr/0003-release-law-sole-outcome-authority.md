# ADR 0003: Release Law — Sole `RunOutcome` Authority (KD2)

Accepted.
## Context

`RunOutcome` (`verified` / `abstained` / `failed`) used to be assembled in several places: the orchestrator could set it from the budget-exhaustion path, the verifier could set it from the hard-gate result, and the worker's final text could be read as a de-facto verdict. The control plane's central promise — the control plane, not an LLM, is authoritative — was only as strong as the weakest of these call sites. Two paths that both thought they owned the verdict could disagree, and the cheaper one could win.

## Decision

`src/northstack/application/release_law.py` is the sole constructor of `RunOutcome`. Nothing else in the control plane builds one; the orchestrator calls `release_law.decide(evidence) -> Verdict` and emits whatever it returns.

The law decides from the evidence manifest — never from the worker's text or the budget path:

- Hard gates run real workspace operations and cannot be waived. A failed hard gate → `FAILED`.
- Soft rubrics run only after hard gates pass. Missing calibration or material disagreement → `ABSTAINED`.
- All acceptance criteria satisfied → `VERIFIED`.

Invariants the law upholds (pinned by a matrix test in `tests/test_release_law.py`): the outcome is a function of the evidence, the `reason` is human-auditable, and the worker's final text is an evidence input only — never a verdict.

## Why

- One owner cannot disagree with itself. Centralising the verdict removes the bug class where two call sites produce different outcomes for the same evidence.
- The authority claim becomes load-bearing: the verdict is decided by code that reads evidence, not by a model that asserts success.
- Budget exhaustion reaching `ABSTAINED` (via `VERIFYING`, per ADR 0001 decision 3) still routes through the release law — it does not bypass it by constructing `RunOutcome` directly.

## Consequences

- A new terminal path that wants to set an outcome must go through `release_law`; it cannot shortcut by constructing `RunOutcome`.
- The worker stays unable to declare the project verified: its output is evidence, and the release law is what turns evidence into a verdict.
