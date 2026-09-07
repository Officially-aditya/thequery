import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");

test("Anthropic frontier enrichment migration is registered", async () => {
  const runner = await source("scripts/migrate.mjs");
  assert.match(runner, /027_enrich_anthropic_frontier/);
});

test("Anthropic frontier enrichment upserts all requested Claude models", async () => {
  const migration = await source("db/migrations/027_enrich_anthropic_frontier.sql");

  for (const slug of [
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
  ]) {
    assert.ok(migration.includes(`\"slug\":\"${slug}\"`), `${slug} should be enriched`);
  }

  assert.match(migration, /ON CONFLICT \(slug\) DO UPDATE/);
  assert.match(migration, /comparison_data = COALESCE\(models\.comparison_data, '\{\}'::jsonb\) \|\| EXCLUDED\.comparison_data/);
  assert.match(migration, /\"slug\":\"claude-opus-5\"[\s\S]*?\"Context window\":\"1M tokens\"[\s\S]*?\"Max output\":\"128K tokens\"[\s\S]*?\"Knowledge cutoff\":\"May 2026\"/);
  assert.match(migration, /\"slug\":\"claude-sonnet-5\"[\s\S]*?\"Input \/ 1M tokens\":\"\$2\"[\s\S]*?\"Output \/ 1M tokens\":\"\$10\"/);
  assert.match(migration, /\"slug\":\"claude-sonnet-4-6\"[\s\S]*?\"Input \/ 1M tokens\":\"\$3\"[\s\S]*?\"Output \/ 1M tokens\":\"\$15\"/);
  assert.match(migration, /\"slug\":\"claude-opus-4-8\"[\s\S]*?\"catalog_status\":\"legacy_active\"/);
  assert.match(migration, /\"Batch \/ flex discount\":\"Batch API: 50% discount\"/);
  assert.match(migration, /\"Long-context surcharge\":\"None — standard pricing through 1M context\"/);
});

test("Anthropic frontier enrichment stores broad normalized benchmark evidence", async () => {
  const migration = await source("db/migrations/027_enrich_anthropic_frontier.sql");
  const benchmarkRows = migration.match(/\"model_slug\":/g) ?? [];
  assert.equal(benchmarkRows.length, 58, "expected 58 normalized benchmark records");

  for (const score of [
    "96.0%", "79.2%", "90.8%", "85.8%", "1861 Elo",
    "88.6%", "69.2%", "82.7%", "83.4%", "1615 Elo",
    "87.6%", "64.3%", "69.4%", "82.3%", "77.3%",
    "80.8%", "53.4%", "65.4%", "72.7%", "1606 Elo",
    "85.2%", "63.2%", "80.4%", "81.2%", "1618 Elo", "54.3%", "13.5%",
    "79.6%", "58.1%", "59.1%", "67.0%", "74.01%", "78.5%", "1395 Elo",
  ]) {
    assert.ok(migration.includes(`\"score_display\":\"${score}\"`), `${score} should be seeded`);
  }

  for (const label of [
    "SWE-bench Verified",
    "SWE-bench Pro",
    "Terminal-Bench 2.0",
    "Terminal-Bench 2.1",
    "Humanity's Last Exam",
    "BrowseComp",
    "OSWorld-Verified",
    "GDPval-AA",
    "GDPval-AA v2",
    "MCP Atlas",
    "Toolathlon",
    "AutomationBench",
  ]) assert.ok(migration.includes(label), `${label} should be represented`);

  assert.ok(migration.includes(`\"evaluator\":\"Anthropic\"`));
  assert.ok(migration.includes(`\"evaluator\":\"Artificial Analysis\"`));
  assert.match(migration, /Original GDPval-AA benchmark; do not compare as if it were GDPval-AA v2/);
  assert.match(migration, /First-attempt success rate; keep distinct from partial-credit and other OSWorld 2\.0 methodologies/);
});

test("Terminal-Bench 2.0 is a first-class comparison row", async () => {
  const [generated, admin] = await Promise.all([
    source("lib/model-comparison.ts"),
    source("components/admin/admin-client.ts"),
  ]);
  assert.ok(generated.includes("Terminal-Bench 2.0"));
  assert.ok(admin.includes("Terminal-Bench 2.0"));
});

test("Anthropic enrichment only backfills empty authored benchmark cells", async () => {
  const migration = await source("db/migrations/027_enrich_anthropic_frontier.sql");

  assert.match(migration, /WHEN COALESCE\(row_entry\.row_value->>1, ''\) = ''[\s\S]*?AND bench_a\.rendered IS NOT NULL[\s\S]*?THEN bench_a\.rendered/);
  assert.match(migration, /WHEN COALESCE\(row_entry\.row_value->>2, ''\) = ''[\s\S]*?AND bench_b\.rendered IS NOT NULL[\s\S]*?THEN bench_b\.rendered/);
  assert.doesNotMatch(migration, /old_a\.value/);
  assert.doesNotMatch(migration, /old_b\.value/);
});
