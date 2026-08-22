import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

test("runtime database access uses the new Neon project variable", async () => {
  const source = await readFile(path.join(import.meta.dirname, "../lib/db.ts"), "utf8");

  assert.match(source, /process\.env\.NEW_DATABASE_URL/);
  assert.doesNotMatch(source, /process\.env\.DATABASE_URL/);
});
