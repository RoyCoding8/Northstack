import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";
import { storage } from "./support.mjs";

globalThis.localStorage = storage();
globalThis.sessionStorage = storage();
const { http, loadSettings } = await import("../../src/northstack/interfaces/web/static/js/api.js");
const originalFetch = globalThis.fetch;

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

test("http.get requests and parses JSON through the API boundary", async () => {
  let request;
  globalThis.fetch = async (...args) => {
    request = args;
    return Response.json({ runs: ["run-1"] });
  };

  assert.deepEqual(await http.get("/runs"), { runs: ["run-1"] });
  assert.equal(request[0], "/api/runs");
  assert.equal(request[1].headers.Accept, "application/json");
});

test("http.get returns null for a no-content response", async () => {
  globalThis.fetch = async () => new Response(null, { status: 204 });

  assert.equal(await http.get("/empty"), null);
});

test("http.get returns null for an empty successful response", async () => {
  globalThis.fetch = async () => new Response("", { status: 200 });

  assert.equal(await http.get("/empty"), null);
});

test("successful malformed JSON rejects with endpoint and status context", async () => {
  globalThis.fetch = async () => new Response("not-json", {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });

  await assert.rejects(http.get("/broken"), error => {
    assert.equal(error.name, "ApiError");
    assert.equal(error.status, 200);
    assert.equal(error.path, "/broken");
    assert.match(error.message, /invalid JSON/);
    return true;
  });
});

test("successful non-JSON content rejects before parsing", async () => {
  globalThis.fetch = async () => new Response('{"looks":"valid"}', {
    status: 200,
    headers: { "Content-Type": "text/html; charset=utf-8" },
  });

  await assert.rejects(http.get("/wrong-type"), error => {
    assert.equal(error.name, "ApiError");
    assert.equal(error.status, 200);
    assert.equal(error.path, "/wrong-type");
    assert.match(error.message, /expected JSON.*text\/html/);
    return true;
  });
});

test("loadSettings rejects invalid persisted polling delays", () => {
  localStorage.setItem("mc.settings", JSON.stringify({
    eventsPollMs: 0,
    historyPollMs: "fast",
    theme: "system",
    landingPage: "#/runs",
  }));

  assert.deepEqual(loadSettings(), {
    theme: "system",
    eventsPollMs: 700,
    historyPollMs: 3000,
    landingPage: "#/runs",
  });
});
