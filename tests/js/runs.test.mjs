import assert from "node:assert/strict";
import test from "node:test";

import { collectRunsCsv, toggleCompareSelection } from "../../src/northstack/interfaces/web/static/js/views/runs.js";

test("comparison selection stores ids and supports deselect and replacement", () => {
  const selected = new Set();
  toggleCompareSelection(selected, "run-a", true);
  toggleCompareSelection(selected, "run-b", true);
  assert.deepEqual([...selected], ["run-a", "run-b"]);
  toggleCompareSelection(selected, "run-a", false);
  assert.deepEqual([...selected], ["run-b"]);
  toggleCompareSelection(selected, "run-c", true);
  toggleCompareSelection(selected, "run-d", true);
  assert.deepEqual([...selected], ["run-d"]);
});

test("CSV export collects every page with one header", async () => {
  const paths = [];
  const csv = await collectRunsCsv(async path => {
    paths.push(path);
    return new Response(`run_id,status\r\nrun-${paths.length},ok\r\n`, {
      headers: {
        "X-NorthStack-Truncated": paths.length === 1 ? "true" : "false",
        "X-NorthStack-Next-Offset": String(paths.length),
      },
    });
  });
  assert.deepEqual(paths, [
    "/runs/export?format=csv&limit=10000&offset=0",
    "/runs/export?format=csv&limit=10000&offset=1",
  ]);
  assert.equal(csv, "run_id,status\r\nrun-1,ok\r\nrun-2,ok\r\n");
});
