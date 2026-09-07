import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");

function parsePayload(sourceText: string, tag: string): unknown[] {
  const escaped = tag.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = sourceText.match(new RegExp(`\\$${escaped}\\$\\s*(\\[[\\s\\S]*?\\])\\s*\\$${escaped}\\$`));
  assert.ok(match?.[1], `expected $${tag}$ payload`);
  return JSON.parse(match[1]);
}

test("final catalog enrichment migration is registered", async () => {
  const runner = await source("scripts/migrate.mjs");
  assert.match(runner, /036_enrich_remaining_catalog/);
});

test("final catalog enrichment upserts all eight remaining models", async () => {
  const migration = await source("db/migrations/036_enrich_remaining_catalog.sql");
  const models = parsePayload(migration, "models") as Array<{ slug: string }>;

  assert.deepEqual(
    new Set(models.map((model) => model.slug)),
    new Set([
      "kimi-k2-5",
      "deepseek-v4-flash-vision-exp",
      "command-a-plus",
      "jamba2-3b",
      "jamba2-mini",
      "minimax-m2-5",
      "minimax-m2-7",
      "ernie-5-1",
    ]),
  );

  assert.match(migration, /ON CONFLICT \(slug\) DO UPDATE/);
  assert.match(migration, /comparison_data=COALESCE\(models\.comparison_data,'\{\}'::jsonb\)\|\|EXCLUDED\.comparison_data/);
});

test("final model profiles contain representative verified specifications", async () => {
  const migration = await source("db/migrations/036_enrich_remaining_catalog.sql");
  const models = parsePayload(migration, "models") as Array<{
    slug: string;
    comparison_data: Record<string, string>;
    metadata: Record<string, unknown>;
  }>;
  const bySlug = new Map(models.map((model) => [model.slug, model]));

  assert.equal(bySlug.get("kimi-k2-5")?.comparison_data["Context window"], "256K");
  assert.equal(bySlug.get("kimi-k2-5")?.comparison_data["Input / 1M tokens"], "¥4.00");
  assert.equal(bySlug.get("kimi-k2-5")?.comparison_data["Output / 1M tokens"], "¥21.00");
  assert.match(bySlug.get("kimi-k2-5")?.comparison_data["Weights / license"] ?? "", /Modified MIT/);

  assert.equal(bySlug.get("deepseek-v4-flash-vision-exp")?.comparison_data["Context window"], "1M");
  assert.equal(bySlug.get("deepseek-v4-flash-vision-exp")?.comparison_data["Max output"], "384K");
  assert.match(bySlug.get("deepseek-v4-flash-vision-exp")?.comparison_data["Weights / license"] ?? "", /MIT/);

  assert.equal(bySlug.get("command-a-plus")?.comparison_data["Context window"], "128K");
  assert.equal(bySlug.get("command-a-plus")?.comparison_data["Max output"], "64K");
  assert.equal(bySlug.get("command-a-plus")?.metadata.parameters_total, "218B");
  assert.equal(bySlug.get("command-a-plus")?.metadata.parameters_active, "25B");

  assert.equal(bySlug.get("jamba2-3b")?.comparison_data["Context window"], "256K");
  assert.equal(bySlug.get("jamba2-mini")?.comparison_data["Context window"], "256K");
  assert.match(bySlug.get("jamba2-3b")?.comparison_data["Weights / license"] ?? "", /Apache 2\.0/);
  assert.match(bySlug.get("jamba2-mini")?.comparison_data["Weights / license"] ?? "", /Apache 2\.0/);

  assert.equal(bySlug.get("minimax-m2-5")?.comparison_data["Input / 1M tokens"], "$0.30");
  assert.equal(bySlug.get("minimax-m2-5")?.comparison_data["Cached input / 1M"], "$0.03");
  assert.equal(bySlug.get("minimax-m2-7")?.comparison_data["Cached input / 1M"], "$0.06");
  assert.equal(bySlug.get("minimax-m2-7")?.comparison_data["Output / 1M tokens"], "$1.20");

  assert.equal(bySlug.get("ernie-5-1")?.comparison_data["Context window"], "128K");
  assert.equal(bySlug.get("ernie-5-1")?.comparison_data["Max output"], "64K");
  assert.match(bySlug.get("ernie-5-1")?.comparison_data["Input / 1M tokens"] ?? "", /¥4\.00/);
});

test("final enrichment stores 37 normalized benchmark rows with provenance", async () => {
  const migration = await source("db/migrations/036_enrich_remaining_catalog.sql");
  const benchmarks = parsePayload(migration, "benchmarks") as Array<{
    model_slug: string;
    benchmark_name: string;
    benchmark_version?: string | null;
    score_display: string;
    evaluator?: string | null;
  }>;

  assert.equal(benchmarks.length, 37);

  const rowsFor = (slug: string) => benchmarks.filter((row) => row.model_slug === slug);
  assert.equal(rowsFor("kimi-k2-5").length, 12);
  assert.equal(rowsFor("deepseek-v4-flash-vision-exp").length, 10);
  assert.equal(rowsFor("command-a-plus").length, 6);
  assert.equal(rowsFor("minimax-m2-5").length, 3);
  assert.equal(rowsFor("minimax-m2-7").length, 4);
  assert.equal(rowsFor("ernie-5-1").length, 2);
  assert.equal(rowsFor("jamba2-3b").length, 0);
  assert.equal(rowsFor("jamba2-mini").length, 0);

  assert.ok(rowsFor("kimi-k2-5").some((row) => row.benchmark_name === "BrowseComp" && row.benchmark_version === "Agent Swarm" && row.score_display === "78.4%"));
  assert.ok(rowsFor("deepseek-v4-flash-vision-exp").some((row) => row.benchmark_name === "Toolathlon" && row.score_display === "75.9%" && row.evaluator === "DeepSeek"));
  assert.ok(rowsFor("command-a-plus").some((row) => row.benchmark_name === "τ²-bench Telecom" && row.score_display === "85%" && row.evaluator === "Cohere"));
  assert.ok(rowsFor("ernie-5-1").some((row) => row.benchmark_name === "Arena Search" && row.score_display === "1223 Elo" && row.evaluator === "LMArena"));
});

test("MiniMax M2.7 keeps GDPval-AA distinct from GDPval-AA v2", async () => {
  const migration = await source("db/migrations/036_enrich_remaining_catalog.sql");
  const benchmarks = parsePayload(migration, "benchmarks") as Array<{
    model_slug: string;
    benchmark_name: string;
    score_display: string;
  }>;
  const m27 = benchmarks.filter((row) => row.model_slug === "minimax-m2-7");

  assert.ok(m27.some((row) => row.benchmark_name === "GDPval-AA" && row.score_display === "1495 Elo"));
  assert.ok(!m27.some((row) => row.benchmark_name === "GDPval-AA v2"));
  assert.match(migration, /comparison_data = COALESCE\(comparison_data, '\{\}'::jsonb\) - 'GDPval-AA v2'/);
});

test("Jamba2 does not receive numeric scores inferred from published charts", async () => {
  const migration = await source("db/migrations/036_enrich_remaining_catalog.sql");
  const benchmarks = parsePayload(migration, "benchmarks") as Array<{ model_slug: string }>;

  assert.ok(!benchmarks.some((row) => row.model_slug === "jamba2-3b"));
  assert.ok(!benchmarks.some((row) => row.model_slug === "jamba2-mini"));
  assert.match(migration, /no numeric rows are inferred/i);
  assert.match(migration, /Numeric chart values are not inferred from images/i);
});

test("final enrichment only fills blank authored comparison cells", async () => {
  const migration = await source("db/migrations/036_enrich_remaining_catalog.sql");

  assert.match(migration, /WHEN COALESCE\(re\.row_value->>1,''\)='' AND a\.rendered IS NOT NULL/);
  assert.match(migration, /WHEN COALESCE\(re\.row_value->>2,''\)='' AND d\.rendered IS NOT NULL/);
  assert.doesNotMatch(migration, /old_a\.value/);
  assert.doesNotMatch(migration, /old_b\.value/);
});
