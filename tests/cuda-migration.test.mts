import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

test("CUDA database update is registered and keeps rendered content in sync", async () => {
  const [runner, migration] = await Promise.all([
    readFile(path.join(import.meta.dirname, "../scripts/migrate.mjs"), "utf8"),
    readFile(path.join(import.meta.dirname, "../db/migrations/004_update_cuda.sql"), "utf8"),
  ]);

  assert.match(runner, /004_update_cuda/);
  assert.match(migration, /## CUDA platform vs CUDA Toolkit vs driver/);
  assert.match(migration, /## CUDA version and driver compatibility/);
  assert.match(migration, /'content', cuda_entry\.body/);
  assert.match(migration, /WHERE item\.kind = 'glossary'[\s\S]+item\.slug = 'cuda'/);
});
