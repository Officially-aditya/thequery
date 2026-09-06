import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

async function source(file: string) {
  return readFile(path.join(root, file), "utf8");
}

test("migration 016 enriches the requested OpenAI comparison models", async () => {
  const [runner, migration] = await Promise.all([
    source("scripts/migrate.mjs"),
    source("db/migrations/016_enrich_openai_gpt56_gpt55_gpt54_small.sql"),
  ]);

  assert.match(runner, /016_enrich_openai_gpt56_gpt55_gpt54_small/);

  for (const slug of [
    "gpt-5-6-sol",
    "gpt-5-6-terra",
    "gpt-5-6-luna",
    "gpt-5-5",
    "gpt-5-4-mini",
    "gpt-5-4-nano",
  ]) {
    assert.match(migration, new RegExp(`\\"slug\\":\\"${slug}\\"`));
  }

  assert.match(migration, /1,050,000 tokens/);
  assert.match(migration, /400,000 tokens/);
  assert.match(migration, /2026-02-16/);
  assert.match(migration, /2025-12-01/);
  assert.match(migration, /2025-08-31/);
  assert.match(migration, /\$4\.00 \(current promotional API price\)/);
  assert.match(migration, /\$2\.00/);
  assert.match(migration, /\$0\.20/);
  assert.match(migration, /gpt-5\.4-nano[\s\S]*?\"Computer use\":\"No\"/);
  assert.match(migration, /gpt-5\.4-mini[\s\S]*?\"Computer use\":\"Yes\"/);
});

test("migration 016 preserves benchmark conditions and versions", async () => {
  const migration = await source("db/migrations/016_enrich_openai_gpt56_gpt55_gpt54_small.sql");

  for (const benchmark of [
    "SWE-bench Pro",
    "DeepSWE v1.1",
    "Terminal-Bench 2.1",
    "FrontierMath",
    "GPQA Diamond",
    "Humanity''s Last Exam",
    "OSWorld 2.0",
    "BrowseComp",
    "GDPval-AA v2",
  ]) {
    assert.match(migration, new RegExp(benchmark.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  }

  assert.match(migration, /'v2 Tier 1-3'/);
  assert.match(migration, /'v2 Tier 4'/);
  assert.match(migration, /FALSE,'xhigh'/);
  assert.match(migration, /TRUE,'xhigh'/);
  assert.match(migration, /Single-model GPT-5\.6 launch comparison suite/);
  assert.match(migration, /Do not confuse with Sol Ultra \/ multi-agent 92\.2%/);
  assert.match(migration, /'Terminal-Bench','2\.0'/);
  assert.match(migration, /'OSWorld','Verified'/);
});
