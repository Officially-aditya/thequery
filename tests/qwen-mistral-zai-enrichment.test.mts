import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");

function payload<T>(sql: string, marker: string): T[] {
  const escaped = marker.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = sql.match(new RegExp(`\\$${escaped}\\$\\s*([\\s\\S]*?)\\s*\\$${escaped}\\$`));
  assert.ok(match?.[1], `${marker} payload should exist`);
  return JSON.parse(match[1]) as T[];
}

type ModelSeed = {
  slug: string;
  access: string;
  comparison_data: Record<string, string>;
  metadata: Record<string, string>;
};

type BenchmarkSeed = {
  model_slug: string;
  benchmark_name: string;
  benchmark_version: string | null;
  score_display: string;
  evaluator: string | null;
};

const migrations = [
  "db/migrations/029_enrich_qwen36_qwen38.sql",
  "db/migrations/030_enrich_mistral_small_medium.sql",
  "db/migrations/031_enrich_zai_glm_catalog.sql",
] as const;

test("Qwen, Mistral and Z.ai enrichment migrations are registered in order", async () => {
  const runner = await source("scripts/migrate.mjs");
  const p29 = runner.indexOf("029_enrich_qwen36_qwen38");
  const p30 = runner.indexOf("030_enrich_mistral_small_medium");
  const p31 = runner.indexOf("031_enrich_zai_glm_catalog");
  assert.ok(p29 > 0 && p30 > p29 && p31 > p30);
});

test("the enrichment batch covers all 12 requested canonical models", async () => {
  const sql = await Promise.all(migrations.map(source));
  const models = sql.flatMap((migration) => payload<ModelSeed>(migration, "models"));
  const slugs = new Set(models.map((model) => model.slug));

  for (const slug of [
    "qwen3-6-35b-a3b",
    "qwen3-6-27b",
    "qwen3-8-flash-next",
    "qwen3-8-2-4t-a95b",
    "mistral-small-4",
    "mistral-medium-3-5",
    "glm-4-7-flash",
    "glm-5",
    "glm-5-turbo",
    "glm-5v-turbo",
    "glm-5-1",
    "glm-5-3-flash",
  ]) {
    assert.ok(slugs.has(slug), `${slug} should be enriched`);
  }
  assert.equal(models.length, 12);
});

test("corrected licenses, context limits and pricing are retained", async () => {
  const sql = await Promise.all(migrations.map(source));
  const models = new Map(
    sql.flatMap((migration) => payload<ModelSeed>(migration, "models")).map((model) => [model.slug, model]),
  );

  assert.equal(models.get("glm-4-7-flash")?.access, "open_source");
  assert.equal(models.get("glm-4-7-flash")?.comparison_data["Weights / license"], "Open source — MIT");
  assert.equal(models.get("glm-5-1")?.access, "open_source");
  assert.equal(models.get("glm-5-1")?.comparison_data["Weights / license"], "Open source — MIT");
  assert.equal(models.get("glm-5-3-flash")?.access, "open_source");
  assert.equal(models.get("glm-5-3-flash")?.comparison_data["Context window"], "1M tokens");
  assert.equal(models.get("glm-5-3-flash")?.comparison_data["Max output"], "128K tokens");

  assert.equal(models.get("qwen3-8-2-4t-a95b")?.comparison_data["Weights / license"], "Open weights — Qwen3.8-Max License");
  assert.equal(models.get("qwen3-6-35b-a3b")?.comparison_data["Input / 1M tokens"], "$0.375 (Alibaba Model Studio Singapore)");
  assert.equal(models.get("qwen3-6-27b")?.comparison_data["Output / 1M tokens"], "$3.60 (Alibaba Model Studio Singapore)");

  assert.equal(models.get("mistral-small-4")?.metadata.parameters_total, "119B");
  assert.equal(models.get("mistral-small-4")?.metadata.parameters_active, "6.5B");
  assert.equal(models.get("mistral-medium-3-5")?.comparison_data["Weights / license"], "Open weights — Modified MIT");
});

test("the enrichment batch stores 69 normalized benchmark rows with representative evidence", async () => {
  const sql = await Promise.all(migrations.map(source));
  const benchmarks = sql.flatMap((migration) => payload<BenchmarkSeed>(migration, "benchmarks"));
  assert.equal(benchmarks.length, 69);

  for (const score of [
    "77.2%",
    "59.3%",
    "73.5%",
    "19.4%",
    "52.3%",
    "92.6%",
    "86.1%",
    "0.72",
    "77.6%",
    "91.4%",
    "59.2%",
    "77.8%",
    "61.1%",
    "79.3%",
    "84.3%",
    "63.4%",
    "78.4%",
    "48.8%",
    "1773 Elo",
  ]) {
    assert.ok(benchmarks.some((row) => row.score_display === score), `${score} should be normalized`);
  }

  for (const evaluator of ["Alibaba / Qwen", "Mistral AI", "Z.ai"]) {
    assert.ok(benchmarks.some((row) => row.evaluator === evaluator), `${evaluator} provenance should be retained`);
  }

  assert.ok(benchmarks.some((row) => row.model_slug === "qwen3-8-flash-next" && row.benchmark_name === "OSWorld 2.0" && row.benchmark_version === "Binary"));
  assert.ok(benchmarks.some((row) => row.model_slug === "qwen3-8-flash-next" && row.benchmark_name === "OSWorld 2.0" && row.benchmark_version === "Partial"));
  assert.ok(benchmarks.some((row) => row.model_slug === "glm-5-3-flash" && row.benchmark_name === "GDPval-AA v2" && row.score_display === "1773 Elo"));
});

test("each enrichment migration only fills blank authored comparison cells", async () => {
  const sql = await Promise.all(migrations.map(source));
  for (const migration of sql) {
    assert.match(migration, /COALESCE\(re\.row_value->>1,''\)='' AND a\.rendered IS NOT NULL/);
    assert.match(migration, /COALESCE\(re\.row_value->>2,''\)='' AND d\.rendered IS NOT NULL/);
    assert.match(migration, /ELSE COALESCE\(re\.row_value->>1,''\) END/);
    assert.match(migration, /ELSE COALESCE\(re\.row_value->>2,''\) END/);
  }
});

test("hosted GLM Turbo variants are enriched without fabricated benchmark rows", async () => {
  const zai = await source("db/migrations/031_enrich_zai_glm_catalog.sql");
  const benchmarks = payload<BenchmarkSeed>(zai, "benchmarks");
  assert.equal(benchmarks.filter((row) => row.model_slug === "glm-5-turbo").length, 0);
  assert.equal(benchmarks.filter((row) => row.model_slug === "glm-5v-turbo").length, 0);
});
