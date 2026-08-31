import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

test("Benchmark database update is registered and keeps rendered content in sync", async () => {
  const [runner, migration] = await Promise.all([
    readFile(path.join(import.meta.dirname, "../scripts/migrate.mjs"), "utf8"),
    readFile(path.join(import.meta.dirname, "../db/migrations/005_update_benchmark.sql"), "utf8"),
  ]);

  assert.match(runner, /005_update_benchmark/);
  assert.match(migration, /## Benchmark vs evaluation vs leaderboard/);
  assert.match(migration, /## How to compare two benchmark claims/);
  assert.match(migration, /'content', benchmark_entry\.body/);
  assert.match(migration, /WHERE item\.kind = 'glossary'[\s\S]+item\.slug = 'benchmark'/);
});
