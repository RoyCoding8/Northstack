import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";
import { deferred, storage, waitFor } from "./support.mjs";

globalThis.localStorage = storage();
globalThis.sessionStorage = storage();
globalThis.document = { addEventListener() {}, hidden: false };

const timers = new Map();
let nextTimer = 1;
globalThis.setTimeout = (fn, ms) => {
  const id = nextTimer++;
  timers.set(id, { fn, ms });
  return id;
};
globalThis.clearTimeout = id => timers.delete(id);

const { http, store } = await import("../../src/northstack/interfaces/web/static/js/api.js");
const { poll } = await import("../../src/northstack/interfaces/web/static/js/poll.js");
const originalGet = http.get;

beforeEach(() => {
  timers.clear();
  poll._timers = {};
  poll._inflight = { history: false, runs: new Set() };
  poll._watched = new Set();
  poll._enabled = true;
  store.state.live = {};
  store.state.settings.eventsPollMs = 25;
});

afterEach(() => {
  http.get = originalGet;
});

test("unwatching an in-flight run prevents its request from restarting polling", async () => {
  const snapshot = deferred();
  http.get = path => path.endsWith("/events?since=0&limit=500")
    ? Promise.resolve({ events: [], next_seq: 0 })
    : snapshot.promise;

  poll.watchRun("run-1");
  await waitFor(() => poll._inflight.runs.has("run-1"));
  poll.unwatchRun("run-1");
  snapshot.resolve({ status: "executing", active: true });
  await waitFor(() => !poll._inflight.runs.has("run-1"));

  assert.equal(poll._watched.has("run-1"), false);
  assert.equal(poll._timers["run-run-1"], undefined);
  assert.equal(timers.size, 0);
});

test("stopping during an in-flight run prevents late timer resurrection", async () => {
  const snapshot = deferred();
  http.get = path => path.endsWith("/events?since=0&limit=500")
    ? Promise.resolve({ events: [], next_seq: 0 })
    : snapshot.promise;

  poll.watchRun("run-1");
  await waitFor(() => poll._inflight.runs.has("run-1"));
  poll.stop();
  snapshot.resolve({ status: "executing", active: true });
  await waitFor(() => !poll._inflight.runs.has("run-1"));

  assert.equal(poll._timers["run-run-1"], undefined);
  assert.equal(timers.size, 0);
});
