# ADR 0007: Live Benchmark Law and Baseline Operationalization

Accepted.
## Context

The benchmark runner existed but ran only fixtures — none of the four preregistered configurations (`strong_single`, `cheap_best_of_n`, `singleton_expert`, `company`) could execute against real endpoints, so the experiment protocol's central question was untestable. Two design questions had to be answered first:

1. What exactly is a baseline? "One end-to-end worker" must not smuggle in company machinery (contract analysis, routing, recovery), or the comparison is rigged in the company's favor.
2. What outcome is retained when the system's own claim and the hidden checks disagree?

## Decision

**The strategy seam is async.** `BenchmarkStrategy.run` and `BenchmarkRunner.run` are async; fixture strategies return without awaiting. Live strategies drive real gateways, workers, and ledgers. The CLI (`northstack benchmark --live --config ...`) awaits them; fixture mode stays the default.

**Every configuration starts from a clean snapshot.** Each task ships an immutable workspace template; every run copies it to a fresh `<output>/runs/<task>-<index>-<config>/workspace` (the SWE-bench reproducibility model: derived state like `.git` and caches is excluded from both the copy and the template digest). A `meta.json` sidecar records the template digest, snapshot path, and timestamp.

**Baselines are bare worker loops.** `LiveWorkerStrategy` instantiates one `NativeWorker` pinned to a single profile with the full tool registry (the same mediated tool policy as the company), a single-cell contract carrying only the task request and ceilings, and bounded loop limits. No contract analysis, no routing, no recovery. The claimed outcome is worker completion under the ceilings; the token ceiling is enforced in-loop, the cost ceiling post-hoc. Profile selection is deterministic and inspectable: the operator's routing chains win when present (worker → cheap, specialist → expert, orchestrator → strong); otherwise tier scoring with price tie-breaks.

**The retained-outcome law: hidden checks decide, claims audit.**

- The retained `RunResult.outcome` is derived from the hidden-check score: all pass → `verified`, none → `failed`, partial → `abstained`.
- The system's own claim is audited against that truth. `false_acceptance` = claimed verified while checks fail; `false_rejection` = claimed failed while checks all pass. The company's claim is its ledger outcome; a baseline's claim is worker completion.
- Best-of-N selects among candidates by hidden-check score — an oracle-based selector, stronger than any deployable verifier could be, so a company win against it is conservative. All candidates' resources are accounted regardless of selection.

**A crashed run is retained, never dropped.** Worker exceptions inside a strategy are caught and recorded as failed runs with the error string; no run disappears because it crashed.

**The shipped suite is integrity-gated.** `benchmarks/live-pilot.json`'s tasks (bug fix, feature, refactor, docs) use stdlib `unittest` hidden checks so no snapshot needs pytest installed. `tests/test_live_suite_integrity.py` pins that every pristine template is partially/fully unsolved and every intended solution reaches score 1.0 — template drift fails CI before a live run burns budget on a broken task.

## Why

- Symmetric ceilings and tool policy mean the only variable between configurations is the control plane itself — which is the hypothesis.
- Oracle-based baseline selection biases against the company, so a positive result is meaningful; a negative result is reported as-is.
- Retaining crashes, failures, and abstentions follows the protocol's "no run may be dropped" law mechanically, not by discipline.

## Consequences

- `BenchmarkTask` grew a structured `checks` field (`HiddenCheck`: exact-argv command with expected exit code, or file presence/content); the legacy `hidden_checks: list[str]` stays for fixture suites.
- `WorkerResult` gained `api_calls` so baselines report real call counts.
- Baseline cost enforcement is post-hoc (a bare loop has no wave boundary); the protocol documents this asymmetry and its direction — baselines may overshoot the cost ceiling by at most one worker turn.
