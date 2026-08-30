# ADR 0004: Single Retry and Budget Authority (KD4)

Accepted.
## Context

Retry and budget accounting used to be spread across three layers:

- The worker ran an internal retry loop — it would back off and retry a cell on its own, invisible to the orchestrator.
- The orchestrator also had retry logic, so a single cell failure could be retried at two levels with no shared cap, no shared dedup, and no shared attempt accounting.
- Budget (`tokens` / `cost_usd`) was enforced in both the worker and the router, each with its own guard, so a spend could be allowed by one and rejected by another, or counted twice.

The result was a class of bugs that is uniquely hard to reason about: how many attempts a cell got depended on which retry loop fired, and the spent total depended on which enforcement path it hit. `max_retries` was a ceiling only by accident — both loops had to honour it, and they didn't share state.

## Decision

**Retry has one owner: `CellRunner`** (`src/northstack/application/cell_runner.py`).

- `CellRunner.run_cell` owns the per-cell retry loop. It is the sole reader of `contract.budget.max_retries`: `max_retries == 0` means a single attempt with no retry; a further failure is `TERMINATE`, not a would-be retry.
- On failure it selects a recovery action from `RetryPolicy` / `RECOVERY_POLICY` (ADR 0001, decision 2), records spend via `BudgetAuthority`, and reroutes to a different profile on a repeated signature. The dedup is keyed on the real `AttemptSignature`, so the same strategy cannot fire twice.
- The worker's internal retry loop was deleted. The worker makes exactly one attempt and returns; there is no second loop kept "just in case".

**Spend has one owner: `BudgetAuthority`** (`src/northstack/application/budget_authority.py`).

- `BudgetAuthority` reserves an estimate (`reserve`), commits the actual (`commit`), and answers `remaining()` as a remaining snapshot. It is the only place tokens and cost are accumulated.
- The worker no longer enforces cost/token limits; it reports `WorkerResult` tallies and `BudgetAuthority` decides. The `max_calls` / wall-time guards stay with the worker — they are run-level bounds, not spend.
- The router reads `remaining_budget` as a read-only snapshot and is kept out of the accounting path (ADR 0001, decision 4).

## Why

- A single owner of the retry cap makes `max_retries` a real ceiling, not a wish: there is no other loop to honour or forget it.
- A single owner of spend cannot double-count or disagree with itself. The worker reports, the authority decides — the split is explicit and tested.
- The `resume_from_messages` transient carry (a crashed attempt resumes from the last succeeded tool round) lives in `CellRunner`, in memory, and is never written to the ledger — the ledger stays the source of truth for committed state, and in-flight conversation is a transient, not a durable fact.

## Consequences

- Two tests that asserted the worker retried internally were updated, not deleted, to assert the no-retry behaviour — the old behaviour was the bug.
- A cell's attempt count is a single number owned by `CellRunner`; reading it from anywhere else is reading a stale copy.
