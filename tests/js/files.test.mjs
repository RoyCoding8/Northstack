import assert from "node:assert/strict";
import test from "node:test";

import { joinTreePath } from "../../src/northstack/interfaces/web/static/js/views/files.js";

test("nested tree paths use the row parent rather than the current root", () => {
  assert.equal(joinTreePath(".", "src"), "src");
  assert.equal(joinTreePath("src", "northstack"), "src/northstack");
  assert.equal(joinTreePath("src/northstack", "config.py"), "src/northstack/config.py");
});
