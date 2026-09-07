import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");

test("Fable GLM Kimi and Grok enrichment migration is registered", async () => {
  const runner = await source("scripts/migrate.mjs");
  assert.match(runner, /028_enrich_fable_glm_kimi_grok/);
});

test("follow-up frontier seed enriches all five requested models", async () => {
  const migration = await source("db/migrations/028_enrich_fable_glm_kimi_grok.sql");

  for (const slug of ["claude-fable-5", "glm-5-2", "kimi-k2-6", "grok-4-6", "grok-4-5"]) {
    assert.ok(migration.includes(`\"slug\":\"${slug}\"`), `${slug} should be enriched`);
  }

  assert.match(migration, /ON CONFLICT \(slug\) DO UPDATE/);
  assert.match(migration, /comparison_data = COALESCE\(models\.comparison_data, '\{\}'::jsonb\) \|\| EXCLUDED\.comparison_data/);

  assert.match(migration, /\"slug\":\"claude-fable-5\"[\s\S]*?\"Context window\":\"1M tokens\"[\s\S]*?\"Max output\":\"128K tokens\"[\s\S]*?\"Input \/ 1M tokens\":\"\$10\"[\s\S]*?\"Output \/ 1M tokens\":\"\$50\"/);
  assert.match(migration, /\"slug\":\"glm-5-2\"[\s\S]*?\"Context window\":\"1M tokens\"[\s\S]*?\"Max output\":\"128K tokens\"[\s\S]*?\"Input \/ 1M tokens\":\"\$1\.40\"[\s\S]*?\"Cached input \/ 1M\":\"\$0\.26\"[\s\S]*?\"Output \/ 1M tokens\":\"\$4\.40\"[\s\S]*?Open source — MIT/);
  assert.match(migration, /\"slug\":\"kimi-k2-6\"[\s\S]*?\"Context window\":\"256K tokens\"[\s\S]*?Open weights — Modified MIT/);
  assert.match(migration, /\"slug\":\"grok-4-6\"[\s\S]*?\"Context window\":\"500K tokens\"[\s\S]*?\"Max output\":\"No text output limit\"[\s\S]*?\"Input \/ 1M tokens\":\"\$2 <=200K \/ \$4 >200K\"[\s\S]*?\"Cached input \/ 1M\":\"\$0\.50 <=200K \/ \$1 >200K\"/);
  assert.match(migration, /\"slug\":\"grok-4-5\"[\s\S]*?\"Context window\":\"500K tokens\"[\s\S]*?\"Input \/ 1M tokens\":\"\$2 <=200K \/ \$4 >200K\"[\s\S]*?\"Cached input \/ 1M\":\"\$0\.30 <=200K \/ \$0\.60 >200K\"/);

  const kimiBlock = migration.match(/\"slug\":\"kimi-k2-6\"[\s\S]*?\"metadata\":\{\"verification_pass\":\"frontier-followup-2026-09-07\"[\s\S]*?\}\n  \}/)?.[0] ?? "";
  assert.doesNotMatch(kimiBlock, /Input \/ 1M tokens|Cached input \/ 1M|Output \/ 1M tokens/);
});

test("follow-up frontier seed stores 40 normalized benchmark rows", async () => {
  const migration = await source("db/migrations/028_enrich_fable_glm_kimi_grok.sql");
  const ids = migration.match(/\"id\":\"tq-20260907-/g) ?? [];
  assert.equal(ids.length, 40);

  for (const score of [
    "84.3%",
    "80.4%",
    "1741 Elo",
    "81.0%",
    "62.1%",
    "1504 Elo",
    "80.2%",
    "58.6%",
    "66.7%",
    "96.4%",
    "90.5%",
    "73.1%",
    "1753 Elo",
    "69.9%",
    "65.9%",
    "83.3%",
    "64.7%",
    "1526 Elo",
  ]) {
    assert.ok(migration.includes(`\"score_display\":\"${score}\"`), `${score} should be normalized`);
  }

  for (const evaluator of ["Anthropic", "Z.ai", "Moonshot AI", "SpaceXAI"]) {
    assert.ok(migration.includes(`\"evaluator\":\"${evaluator}\"`), `${evaluator} provenance should be retained`);
  }

  assert.match(migration, /\"model_slug\":\"kimi-k2-6\"[\s\S]*?\"benchmark_name\":\"Terminal-Bench 2\.0\"/);
  assert.match(migration, /\"model_slug\":\"glm-5-2\"[\s\S]*?\"benchmark_name\":\"Toolathlon\",\"benchmark_version\":\"Verified\"/);
  assert.doesNotMatch(migration, /\"score_display\":\"86\.3%\"/);
  assert.match(migration, /INSERT INTO model_benchmarks/);
  assert.match(migration, /ON CONFLICT \(id\) DO UPDATE/);
});

test("follow-up frontier seed only fills blank authored benchmark cells", async () => {
  const migration = await source("db/migrations/028_enrich_fable_glm_kimi_grok.sql");

  assert.match(migration, /WHEN COALESCE\(row_entry\.row_value->>1, ''\) = '' AND bench_a\.rendered IS NOT NULL/);
  assert.match(migration, /WHEN COALESCE\(row_entry\.row_value->>2, ''\) = '' AND bench_b\.rendered IS NOT NULL/);
  assert.match(migration, /ELSE COALESCE\(row_entry\.row_value->>1, ''\)/);
  assert.match(migration, /ELSE COALESCE\(row_entry\.row_value->>2, ''\)/);
});
