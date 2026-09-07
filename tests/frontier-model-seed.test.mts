import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");

test("frontier model seed migration is registered", async () => {
  const runner = await source("scripts/migrate.mjs");
  assert.match(runner, /026_seed_frontier_models/);
});

test("frontier model seed upserts all requested canonical models", async () => {
  const migration = await source("db/migrations/026_seed_frontier_models.sql");

  for (const slug of [
    "kimi-k3",
    "minimax-m3",
    "glm-5-3",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "qwen3-8-max",
    "qwen3-8-27b",
  ]) {
    assert.ok(migration.includes(`\"slug\":\"${slug}\"`), `${slug} should be seeded or enriched`);
  }

  assert.match(migration, /ON CONFLICT \(slug\) DO UPDATE/);
  assert.match(migration, /comparison_data = COALESCE\(models\.comparison_data, '\{\}'::jsonb\) \|\| EXCLUDED\.comparison_data/);
  assert.match(migration, /\"slug\":\"glm-5-3\"[\s\S]*?\"Context window\":\"1M tokens\"[\s\S]*?\"Max output\":\"128K tokens\"/);
  assert.match(migration, /\"slug\":\"qwen3-8-max\"[\s\S]*?\"Release date\":\"2026-08-02\"/);
  assert.match(migration, /\"slug\":\"qwen3-8-27b\"[\s\S]*?\"Max output\":\"128K tokens\"/);
  assert.match(migration, /\"slug\":\"deepseek-v4-pro\"[\s\S]*?\"Max output\":\"384K tokens\"/);
  assert.match(migration, /\"slug\":\"deepseek-v4-flash\"[\s\S]*?\"Max output\":\"384K tokens\"/);
  assert.match(migration, /\"slug\":\"kimi-k3\"[\s\S]*?\"Weights \/ license\":\"Open weights — Kimi K3 License\"/);
  assert.match(migration, /\"slug\":\"minimax-m3\"[\s\S]*?\"Weights \/ license\":\"Open weights — MiniMax Community License\"/);
});

test("frontier model seed stores normalized benchmark evidence", async () => {
  const migration = await source("db/migrations/026_seed_frontier_models.sql");

  for (const score of [
    "93.5%",
    "88.3%",
    "1686 Elo",
    "59.0%",
    "83.5%",
    "88.2%",
    "1769 Elo",
    "82.7%",
    "87.9%",
    "86.6%",
    "1739 Elo",
    "73.0%",
    "90.3%",
    "84.3%",
  ]) {
    assert.ok(migration.includes(`\"score_display\":\"${score}\"`), `${score} should be normalized`);
  }

  for (const evaluator of ["Moonshot AI", "MiniMax", "Z.ai", "DeepSeek", "Alibaba / Qwen", "Artificial Analysis"]) {
    assert.ok(migration.includes(`\"evaluator\":\"${evaluator}\"`), `${evaluator} provenance should be retained`);
  }

  assert.match(migration, /\"benchmark_name\":\"Toolathlon\",\"benchmark_version\":\"Verified\"/);
  assert.match(migration, /\"model_slug\":\"qwen3-8-27b\"[\s\S]*?\"benchmark_name\":\"OSWorld 2\.0\",\"benchmark_version\":\"Binary\"/);
  assert.match(migration, /\"model_slug\":\"qwen3-8-27b\"[\s\S]*?\"benchmark_name\":\"OSWorld 2\.0\",\"benchmark_version\":\"Partial\"/);
  assert.match(migration, /INSERT INTO model_benchmarks/);
  assert.match(migration, /ON CONFLICT \(id\) DO UPDATE/);
});

test("frontier seed only fills empty authored comparison benchmark cells", async () => {
  const migration = await source("db/migrations/026_seed_frontier_models.sql");

  assert.match(migration, /WHEN COALESCE\(row_entry\.row_value->>1, ''\) = '' AND bench_a\.rendered IS NOT NULL THEN bench_a\.rendered/);
  assert.match(migration, /WHEN COALESCE\(row_entry\.row_value->>2, ''\) = '' AND bench_b\.rendered IS NOT NULL THEN bench_b\.rendered/);
  assert.doesNotMatch(migration, /THEN bench_a\.rendered\s+WHEN/);
  assert.doesNotMatch(migration, /THEN bench_b\.rendered\s+WHEN/);
});
