import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

test("Claude Fable 5.1 glossary migration is registered and keeps its body block in sync", async () => {
  const [runner, migration] = await Promise.all([
    readFile(path.join(import.meta.dirname, "../scripts/migrate.mjs"), "utf8"),
    readFile(path.join(import.meta.dirname, "../db/migrations/008_add_claude_fable_51.sql"), "utf8"),
  ]);

  assert.match(runner, /008_add_claude_fable_51/);
  assert.match(migration, /## Benchmark profile/);
  assert.match(migration, /## Pricing and cache economics/);
  assert.match(migration, /'content', fable_51_entry\.body/);
  assert.match(migration, /'claude-fable-51'/);
  assert.match(migration, /slug = 'claude-fable-5'/);
});
