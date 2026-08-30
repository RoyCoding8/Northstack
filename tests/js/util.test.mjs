import assert from "node:assert/strict";
import test from "node:test";
import { storage } from "./support.mjs";

globalThis.localStorage = storage();
globalThis.sessionStorage = storage();
globalThis.document = { addEventListener() {}, hidden: false };
const { fmtSpan, fmtElapsed } = await import("../../src/northstack/interfaces/web/static/js/util.js");

test("fmtSpan renders a finished run's duration", () => {
  assert.equal(fmtSpan(0), "0s");
  assert.equal(fmtSpan(45), "45s");
  assert.equal(fmtSpan(90), "1m 30s");
  assert.equal(fmtSpan(3661), "1h 1m 1s");
});

test("fmtSpan never renders a negative span", () => {
  assert.equal(fmtSpan(-10), "0s");
});

test("fmtElapsed falls back to an em dash without a start", () => {
  assert.equal(fmtElapsed(null), "—");
  assert.equal(fmtElapsed(0), "—");
});
