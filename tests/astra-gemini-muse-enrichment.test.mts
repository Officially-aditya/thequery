import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");

test("Astra, Fable, Gemini and Muse enrichment migrations are registered", async () => {
  const runner = await source("scripts/migrate.mjs");
  for (const id of [
    "018_enrich_astra_fable",
    "019_enrich_gemini_core",
    "020_enrich_gemini_specialized",
    "021_enrich_muse_family",
    "022_refresh_enriched_comparisons",
    "023_backfill_agentic_benchmark_labels",
  ]) assert.match(runner, new RegExp(id));
});

test("Astra and Fable have complete comparable catalog profiles", async () => {
  const migration = await source("db/migrations/018_enrich_astra_fable.sql");
  assert.match(migration, /\"slug\":\"gpt-6-astra\"/);
  assert.match(migration, /\"Context window\":\"1\.05M tokens\"/);
  assert.match(migration, /\"Knowledge cutoff\":\"Apr 30, 2026\"/);
  assert.match(migration, /\"slug\":\"claude-fable-5-1\"/);
  assert.match(migration, /\"Context window\":\"1M tokens\"/);
  assert.match(migration, /FrontierCode 1\.1 Main/);
  assert.match(migration, /FrontierMath Tier 4 \(v2\)/);
  assert.match(migration, /\"evaluator\":\"OpenAI\"/);
});

test("Gemini core and specialized catalog records are enriched", async () => {
  const [core, specialized] = await Promise.all([
    source("db/migrations/019_enrich_gemini_core.sql"),
    source("db/migrations/020_enrich_gemini_specialized.sql"),
  ]);
  for (const slug of [
    "gemini-3-1-pro",
    "gemini-3-1-flash-lite",
    "gemini-3-5-flash",
    "gemini-3-5-flash-lite",
    "gemini-3-6-flash",
    "gemini-3-7-flash",
    "gemini-3-8-flash",
  ]) assert.match(core, new RegExp(`\\"slug\\":\\"${slug}\\"`));

  for (const slug of [
    "gemini-3-1-flash-image",
    "gemini-3-1-flash-live",
    "gemini-3-1-flash-tts",
    "gemini-3-1-flash-lite-image",
    "gemini-3-5-audio",
    "gemini-omni-flash",
    "gemini-omni-1-1-flash",
  ]) assert.match(specialized, new RegExp(`\\"slug\\":\\"${slug}\\"`));

  assert.match(core, /GDPval-AA v2/);
  assert.match(core, /OSWorld-Verified/);
  assert.match(core, /MCP Atlas/);
  assert.match(specialized, /\"Context window\":\"128K tokens\"/);
  assert.match(specialized, /\"Context window\":\"64K tokens\"/);
});

test("Muse catalog preserves verified access distinctions", async () => {
  const migration = await source("db/migrations/021_enrich_muse_family.sql");
  for (const slug of ["muse-spark", "muse-spark-1-1", "muse-spark-1-2", "muse-spark-1-3", "muse-glimmer"]) {
    assert.match(migration, new RegExp(`\\"slug\\":\\"${slug}\\"`));
  }
  assert.match(migration, /\"slug\":\"muse-spark-1-2\"[\s\S]*?\"access\":\"proprietary\"/);
  assert.match(migration, /open weights were forthcoming but had not released/);
  assert.match(migration, /\"slug\":\"muse-glimmer\"[\s\S]*?\"access\":\"open_source\"/);
  assert.match(migration, /Open source — Apache 2\.0/);
});

test("comparison vocabulary exposes enriched benchmarks and refresh preserves edits", async () => {
  const [generated, admin, refresh, backfill] = await Promise.all([
    source("lib/model-comparison.ts"),
    source("components/admin/admin-client.ts"),
    source("db/migrations/022_refresh_enriched_comparisons.sql"),
    source("db/migrations/023_backfill_agentic_benchmark_labels.sql"),
  ]);
  for (const label of [
    "FrontierCode 1.1 Main",
    "FrontierCode 1.1 Extended",
    "Terminal-Bench 3.0",
    "Terminal-Bench Science 0.1",
    "MLE-Bench",
    "FrontierMath Tier 4 (v2)",
    "OSWorld-Verified",
    "GDPval-AA",
    "GDPval-AA v2",
    "AutomationBench",
    "Agents' Last Exam",
    "MCP Atlas",
    "Toolathlon",
  ]) {
    assert.ok(generated.includes(label), `${label} should be shown on database comparisons`);
    assert.ok(admin.includes(label), `${label} should be available in new comparison templates`);
    assert.ok(refresh.includes(label), `${label} should be materialized into fresh comparisons`);
    assert.ok(backfill.includes(label), `${label} should be materialized into already-migrated comparisons`);
  }
  assert.match(refresh, /model_benchmarks/);
  assert.match(refresh, /COALESCE\(NULLIF\(er\.values_by_label->lower\(l\.label\)->>0, ''\), p\.model_a_data->>l\.label, ''\)/);
  assert.match(refresh, /COALESCE\(NULLIF\(er\.values_by_label->lower\(l\.label\)->>1, ''\), p\.model_b_data->>l\.label, ''\)/);
  assert.match(backfill, /model_benchmarks/);
});
