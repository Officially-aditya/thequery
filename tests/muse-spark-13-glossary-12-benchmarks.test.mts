import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");

test("glossary 1.2 benchmark fill migration is registered", async () => {
  const runner = await source("scripts/migrate.mjs");
  assert.match(runner, /079_fill_muse_spark_13_glossary_12_benchmarks/);
});

test("empty 1.2 cells are filled from normalized benchmark rows", async () => {
  const migration = await source("db/migrations/079_fill_muse_spark_13_glossary_12_benchmarks.sql");

  assert.match(migration, /\| SWEAtlas CodeBase QnA \| 59\.4 \| 46\.2 \| 53\.5 \| 52\.7 \|/);
  assert.match(migration, /\| Terminal-Bench 2\.1 \| 88\.8 \| 82\.9\u2020 \| 88\.8 \(tie\) \| 86\.7 \|/);
  // "not reported" appears 3x total: once in the header comment and twice in
  // the REPLACE search patterns — never as a new cell value.
  assert.strictEqual(migration.split("not reported").length - 1, 3);
  assert.match(migration, /muse-spark-1-3/);
  assert.match(migration, /blocks/);
  assert.match(migration, /updated_at = NOW\(\)/);
});
