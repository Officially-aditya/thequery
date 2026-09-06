import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

async function source(file: string) {
  return readFile(path.join(root, file), "utf8");
}

test("token counts are normalized in stored comparison data", async () => {
  const [runner, migration] = await Promise.all([
    source("scripts/migrate.mjs"),
    source("db/migrations/017_compact_token_counts.sql"),
  ]);

  assert.match(runner, /017_compact_token_counts/);
  assert.match(migration, /'1,050,000 tokens', '1\.05M tokens'/);
  assert.match(migration, /'1,048,576 tokens', '1M tokens'/);
  assert.match(migration, /'131,072 input tokens', '128K input tokens'/);
  assert.match(migration, /'128,000 tokens', '128K tokens'/);
  assert.match(migration, /'65,536 tokens', '64K tokens'/);
  assert.match(migration, /UPDATE models/);
  assert.match(migration, /WHERE kind = 'comparison'/);
});

test("spec tables compact raw future token counts at render time", async () => {
  const renderer = await source("components/content/ContentBlocksRenderer.tsx");

  assert.match(renderer, /function compactTokenCounts/);
  assert.match(renderer, /\[131_072, "128K"\]/);
  assert.match(renderer, /\[65_536, "64K"\]/);
  assert.match(renderer, /value >= 1_000_000/);
  assert.match(renderer, /value % 1_000 === 0/);
  assert.match(renderer, /compactTokenCounts\(text\)/);
});
