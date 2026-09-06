import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

async function source(file: string) {
  return readFile(path.join(root, file), "utf8");
}

function payload<T>(sql: string, marker: string): T[] {
  const escaped = marker.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = sql.match(new RegExp(`\\$${escaped}\\$\\s*([\\s\\S]*?)\\s*\\$${escaped}\\$`));
  assert.ok(match?.[1], `${marker} payload should exist`);
  return JSON.parse(match[1]) as T[];
}

test("migration 015 reconciles the comprehensive 2026 comparison catalog", async () => {
  const [runner, migration] = await Promise.all([
    source("scripts/migrate.mjs"),
    source("db/migrations/015_comprehensive_model_catalog.sql"),
  ]);

  assert.match(runner, /015_comprehensive_model_catalog/);
  assert.match(migration, /CREATE TABLE IF NOT EXISTS model_benchmarks/);
  assert.match(migration, /ALTER TABLE models ADD COLUMN IF NOT EXISTS family/);
  assert.match(migration, /ALTER TABLE models ADD COLUMN IF NOT EXISTS ga_date/);
  assert.match(migration, /DELETE FROM models WHERE slug = 'ernie-5-0'/);

  const newModels = payload<{
    slug: string;
    name: string;
    developer: string;
    release_date: string;
    access: string;
    family: string;
    comparison_data: Record<string, string>;
    sources: Array<{ title: string; url: string }>;
  }>(migration, "newmodels");
  const patches = payload<{ slug: string }>(migration, "patches");
  const benchmarks = payload<{
    id: string;
    model_slug: string;
    category: string;
    benchmark_name: string;
    score_display: string;
    source: string | null;
  }>(migration, "benchmarks");

  assert.equal(newModels.length, 19);
  assert.equal(patches.length, 76);
  assert.equal(benchmarks.length, 100);
  assert.equal(new Set(newModels.map((model) => model.slug)).size, newModels.length);
  assert.equal(new Set(benchmarks.map((benchmark) => benchmark.id)).size, benchmarks.length);

  for (const model of newModels) {
    assert.match(model.release_date, /^2026-\d{2}-\d{2}$/);
    assert.ok(["proprietary", "restricted", "open_weights", "open_source"].includes(model.access));
    assert.equal(model.comparison_data.Developer, model.developer);
    assert.equal(model.comparison_data["Release date"], model.release_date);
    assert.ok(model.sources.length > 0, `${model.name} should have at least one verification source`);
    for (const item of model.sources) assert.match(item.url, /^https:\/\//);
  }

  for (const expected of [
    "Gemini 3.1 Flash Image",
    "Gemini 3.1 Flash Live",
    "Gemini 3.5 Audio",
    "GLM-5",
    "GLM-5-Turbo",
    "GLM-5V-Turbo",
    "MiniMax M3",
    "Muse Glimmer",
    "Qwen3.6-27B",
    "Qwen3.6-35B-A3B",
    "Qwen3.8-2.4T-A95B",
  ]) {
    assert.ok(newModels.some((model) => model.name === expected), `${expected} should be added`);
  }

  assert.ok(benchmarks.some((benchmark) => benchmark.benchmark_name === "SWE-bench Pro"));
  assert.ok(benchmarks.some((benchmark) => benchmark.benchmark_name === "Terminal-Bench 4.0"));
  assert.ok(benchmarks.some((benchmark) => benchmark.benchmark_name === "Humanity's Last Exam"));
  assert.ok(benchmarks.some((benchmark) => benchmark.benchmark_name === "OSWorld 2.0"));
  assert.ok(benchmarks.some((benchmark) => benchmark.benchmark_name === "BrowseComp"));
  assert.ok(benchmarks.every((benchmark) => benchmark.score_display.trim().length > 0));
});

test("model reads preserve benchmark conditions and make them available to comparison autofill", async () => {
  const models = await source("lib/models.ts");

  assert.match(models, /FROM model_benchmarks/);
  assert.match(models, /benchmarkVersion/);
  assert.match(models, /reasoningEffort/);
  assert.match(models, /evidence\.tools === true/);
  assert.match(models, /evidence\.harness/);
  assert.match(models, /evidence\.evaluator/);
  assert.match(models, /enrichComparisonData/);
  assert.match(models, /values\.join\(" · "\)/);
  assert.match(models, /enrichSources/);
});
