import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");

function benchmarkBlock(migration: string): string {
  const match = migration.match(/WITH b\(model_slug,category,benchmark_name[\s\S]*?\n\)\nINSERT INTO model_benchmarks/);
  assert.ok(match, "migration should contain normalized benchmark VALUES");
  return match[0];
}

function benchmarkRowCount(migration: string): number {
  return (benchmarkBlock(migration).match(/^\('/gm) ?? []).length;
}

test("Qwen and Gemma enrichment migrations are registered in order", async () => {
  const runner = await source("scripts/migrate.mjs");
  const qwen = runner.indexOf("032_enrich_qwen_remaining");
  const gemma = runner.indexOf("033_enrich_gemma4_family");
  assert.ok(qwen >= 0, "032 should be registered");
  assert.ok(gemma > qwen, "033 should run after 032");
});

test("remaining Qwen enrichment covers all fifteen catalog models", async () => {
  const migration = await source("db/migrations/032_enrich_qwen_remaining.sql");
  for (const slug of [
    "qwen3-5-397b-a17b",
    "qwen3-5-122b-a10b",
    "qwen3-5-35b-a3b",
    "qwen3-5-27b",
    "qwen3-5-plus",
    "qwen3-5-flash",
    "qwen3-5-9b",
    "qwen3-5-4b",
    "qwen3-5-2b",
    "qwen3-5-0-8b",
    "qwen3-6-plus",
    "qwen3-6-flash",
    "qwen3-6-max-preview",
    "qwen3-7-max",
    "qwen3-7-plus",
  ]) assert.ok(migration.includes(`'${slug}'`), `${slug} should be enriched`);

  assert.match(migration, /'qwen3-5-397b-a17b'[\s\S]*?'397B'[\s\S]*?'17B'/);
  assert.match(migration, /'qwen3-5-397b-a17b'[\s\S]*?'262K native; extensible to ~1\.01M'/);
  assert.match(migration, /'qwen3-5-plus'[\s\S]*?'1M','64K'[\s\S]*?'\$0\.40 ≤256K \/ \$0\.50 >256K'/);
  assert.match(migration, /'qwen3-5-flash'[\s\S]*?'1M','64K'[\s\S]*?'\$0\.10'/);
  assert.match(migration, /'qwen3-6-max-preview'[\s\S]*?'256K','64K'[\s\S]*?'No','No'/);
  assert.match(migration, /'qwen3-7-max'[\s\S]*?'1M','128K'[\s\S]*?'\$2\.50'/);
  assert.match(migration, /'Weights \/ license','Open source — Apache 2\.0'/);
  assert.match(migration, /ON CONFLICT\(slug\) DO UPDATE/);
});

test("Qwen enrichment stores normalized evidence without fabricating hosted-model benchmark rows", async () => {
  const migration = await source("db/migrations/032_enrich_qwen_remaining.sql");
  const benchmarks = benchmarkBlock(migration);
  assert.equal(benchmarkRowCount(migration), 62, "expected 62 normalized Qwen benchmark rows");

  for (const fragment of [
    "'qwen3-5-397b-a17b','coding','SWE-bench Verified','',76.2",
    "'qwen3-5-397b-a17b','knowledge','MMLU-Pro','',87.8",
    "'qwen3-5-122b-a10b','knowledge','GPQA Diamond','',86.6",
    "'qwen3-5-9b','coding','LiveCodeBench','v6',65.6",
    "'qwen3-5-2b','knowledge','MMLU-Pro','',55.3,'false','non-thinking'",
    "'qwen3-5-2b','knowledge','MMLU-Pro','',66.5,'false','thinking'",
    "'qwen3-7-plus','agentic_computer_use','Toolathlon','Verified',50.6",
    "'qwen3-7-plus','agentic_computer_use','OSWorld 2.0','Binary',2.8",
    "'qwen3-7-plus','agentic_computer_use','OSWorld 2.0','Partial',21.5",
  ]) assert.ok(benchmarks.includes(fragment), `${fragment} should be represented`);

  assert.ok(benchmarks.includes("'Alibaba / Qwen'"));
  for (const hosted of ["qwen3-5-plus", "qwen3-5-flash", "qwen3-6-plus", "qwen3-6-flash", "qwen3-6-max-preview", "qwen3-7-max"])
    assert.equal((benchmarks.match(new RegExp(`'${hosted}'`, "g")) ?? []).length, 0, `${hosted} should not receive inferred benchmark rows`);

  assert.match(benchmarks, /,'false',/);
  assert.match(benchmarks, /,'true',/);
  assert.match(migration, /CASE WHEN tools='' THEN NULL ELSE tools::boolean END/);
});

test("Gemma 4 enrichment covers all five variants and their architecture differences", async () => {
  const migration = await source("db/migrations/033_enrich_gemma4_family.sql");
  for (const slug of ["gemma-4-e2b", "gemma-4-e4b", "gemma-4-12b", "gemma-4-26b-a4b", "gemma-4-31b"])
    assert.ok(migration.includes(`'${slug}'`), `${slug} should be enriched`);

  assert.match(migration, /'gemma-4-e2b'[\s\S]*?'128K','Yes'/);
  assert.match(migration, /'gemma-4-e4b'[\s\S]*?'128K','Yes'/);
  assert.match(migration, /'gemma-4-12b','Gemma 4 12B','2026-06-03'[\s\S]*?'256K','Yes'/);
  assert.match(migration, /'gemma-4-26b-a4b'[\s\S]*?"parameters_total":"25\.2B"[\s\S]*?"parameters_active":"3\.8B"/);
  assert.match(migration, /'gemma-4-31b'[\s\S]*?"parameters_total":"30\.7B"/);
  assert.match(migration, /'open_source','Gemma 4'/);
  assert.match(migration, /'Weights \/ license','Open source — Apache 2\.0'/);
});

test("Gemma 4 enrichment keeps Google and cross-vendor evaluations distinct", async () => {
  const migration = await source("db/migrations/033_enrich_gemma4_family.sql");
  const benchmarks = benchmarkBlock(migration);
  assert.equal(benchmarkRowCount(migration), 28, "expected 28 normalized Gemma benchmark rows");

  for (const fragment of [
    "'gemma-4-31b','knowledge','MMLU-Pro','',85.2",
    "'gemma-4-31b','math_reasoning','AIME','2026',89.2",
    "'gemma-4-31b','coding','LiveCodeBench','v6',80.0",
    "'gemma-4-31b','knowledge','GPQA Diamond','',84.3",
    "'gemma-4-26b-a4b','knowledge','MMLU-Pro','',82.6",
    "'gemma-4-12b','coding','LiveCodeBench','v6',72.0",
    "'gemma-4-e4b','knowledge','GPQA Diamond','',58.6",
    "'gemma-4-e2b','math_reasoning','AIME','2026',37.5",
  ]) assert.ok(benchmarks.includes(fragment), `${fragment} should be represented`);

  assert.match(benchmarks, /'gemma-4-31b','coding','SWE-bench Verified','',52\.0,'','','Qwen agent scaffold','Alibaba \/ Qwen'/);
  assert.match(benchmarks, /Cross-vendor rerun published by Qwen; not a Google evaluation/);
  assert.ok(benchmarks.includes("'Google DeepMind'"));
  assert.ok(benchmarks.includes("'Alibaba / Qwen'"));
});

test("Qwen and Gemma enrichment only fills blank authored comparison cells", async () => {
  const [qwen, gemma] = await Promise.all([
    source("db/migrations/032_enrich_qwen_remaining.sql"),
    source("db/migrations/033_enrich_gemma4_family.sql"),
  ]);

  for (const migration of [qwen, gemma]) {
    assert.match(migration, /COALESCE\(re\.row_value->>1,''\)='' AND a\.rendered IS NOT NULL/);
    assert.match(migration, /COALESCE\(re\.row_value->>2,''\)='' AND d\.rendered IS NOT NULL/);
    assert.doesNotMatch(migration, /old_a\.value/);
    assert.doesNotMatch(migration, /old_b\.value/);
  }
});
