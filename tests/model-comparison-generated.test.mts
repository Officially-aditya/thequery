import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("generated comparisons do not append generic bottom-line boilerplate", async () => {
  const source = await readFile(path.join(root, "lib/model-comparison.ts"), "utf8");

  assert.doesNotMatch(source, /database-comparison-bottom-line/);
  assert.doesNotMatch(source, /This comparison is generated from TheQuery's verified model catalog/);
  assert.doesNotMatch(source, /## Bottom line/);
});
