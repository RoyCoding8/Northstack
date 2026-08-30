import assert from "node:assert/strict";
import test from "node:test";
import { storage } from "./support.mjs";

globalThis.localStorage = storage();
globalThis.sessionStorage = storage();
globalThis.document = { addEventListener() {}, hidden: false };
const { budgetProgress, collectLedger } = await import("../../src/northstack/interfaces/web/static/js/views/run_detail.js");

test("ledger export collects every page exactly once", async () => {
  const paths = [];
  const ledger = await collectLedger("run-1", async path => {
    paths.push(path);
    return paths.length === 1
      ? { run_id: "run-1", events: [{ seq: 1 }, { seq: 2 }], next_seq: 2, truncated: true }
      : { run_id: "run-1", events: [{ seq: 3 }], next_seq: 3, truncated: false };
  });
  assert.deepEqual(paths, [
    "/runs/run-1/ledger.json?since=0&limit=5000",
    "/runs/run-1/ledger.json?since=2&limit=5000",
  ]);
  assert.deepEqual(ledger.events.map(event => event.seq), [1, 2, 3]);
});

test("ledger export rejects stalled pagination and oversized output", async () => {
  await assert.rejects(
    collectLedger("run-1", async () => ({ events: [], next_seq: 0, truncated: true })),
    /made no progress/,
  );
  await assert.rejects(
    collectLedger("run-1", async () => ({ events: [{ seq: 1 }, { seq: 2 }], truncated: false }), 1),
    /exceeds 1 events/,
  );
});

test("budget progress uses the run snapshot rather than current config", () => {
  assert.deepEqual(budgetProgress({
    usage: { total_input_tokens: 20, total_output_tokens: 30, total_cost_usd: 1.25 },
    budget: { token_limit: 200, cost_limit_usd: 3 },
  }), {
    tokens: 50,
    token_limit: 200,
    token_percent: 25,
    cost: 1.25,
    cost_limit: 3,
  });
  assert.equal(budgetProgress({ usage: {} }).token_percent, null);
});
