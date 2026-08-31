import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

test("dense retrieval database update is registered and keeps body blocks in sync", async () => {
  const [runner, migration] = await Promise.all([
    readFile(path.join(import.meta.dirname, "../scripts/migrate.mjs"), "utf8"),
    readFile(path.join(import.meta.dirname, "../db/migrations/003_update_dense_retrieval.sql"), "utf8"),
  ]);

  assert.match(runner, /003_update_dense_retrieval/);
  assert.match(migration, /## Dense retrieval vs sparse retrieval/);
  assert.match(migration, /## Dense retrieval in RAG/);
  assert.match(migration, /'content', dense_retrieval\.body/);
  assert.match(migration, /WHERE item\.kind = 'glossary'[\s\S]+item\.slug = 'dense-retrieval'/);
});
