import assert from "node:assert/strict";

export function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => ([resolve, reject] = [yes, no]));
  return { promise, resolve, reject };
}

export function storage() {
  const values = new Map();
  return {
    clear: () => values.clear(),
    getItem: key => values.get(key) ?? null,
    removeItem: key => values.delete(key),
    setItem: (key, value) => values.set(key, String(value)),
  };
}

export async function waitFor(predicate, turns = 20) {
  for (let turn = 0; turn < turns && !predicate(); turn += 1) await Promise.resolve();
  assert.ok(predicate(), "condition was not reached before the microtask limit");
}
