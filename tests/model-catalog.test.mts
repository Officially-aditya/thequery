import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

async function source(file: string) {
  return readFile(path.join(root, file), "utf8");
}

test("model catalog migration is registered and seeds the verified 2026 catalog", async () => {
  const [runner, migration] = await Promise.all([
    source("scripts/migrate.mjs"),
    source("db/migrations/011_model_catalog.sql"),
  ]);

  assert.match(runner, /011_model_catalog/);
  assert.match(migration, /CREATE TABLE IF NOT EXISTS models/);
  assert.match(migration, /comparison_data JSONB NOT NULL/);
  assert.match(migration, /sources JSONB NOT NULL/);
  assert.match(migration, /ON CONFLICT \(slug\) DO UPDATE/);

  const payload = migration.match(/\$models\$\s*([\s\S]*?)\s*\$models\$/)?.[1];
  assert.ok(payload, "migration should contain the model seed JSON");
  const models = JSON.parse(payload) as Array<{
    slug: string;
    name: string;
    developer: string;
    release_date: string;
    access: string;
    comparison_data: Record<string, string>;
    sources: Array<{ title: string; url: string }>;
  }>;

  assert.equal(models.length, 75);
  assert.equal(new Set(models.map((model) => model.slug)).size, models.length);

  for (const model of models) {
    assert.ok(model.slug);
    assert.ok(model.name);
    assert.ok(model.developer);
    assert.match(model.release_date, /^2026-\d{2}-\d{2}$/);
    assert.ok(["proprietary", "restricted", "open_weights", "open_source"].includes(model.access));
    assert.equal(model.comparison_data.Developer, model.developer);
    assert.equal(model.comparison_data["Release date"], model.release_date);
    assert.ok(model.sources.length > 0, `${model.name} should have a verification source`);
    for (const item of model.sources) assert.match(item.url, /^https:\/\//);
  }

  for (const expected of [
    "GPT-6 Astra",
    "Claude Fable 5.1",
    "Gemini 3.8 Flash",
    "Grok 4.6",
    "Kimi K3",
    "DeepSeek V4 Pro",
    "Mistral Small 4",
    "Command A+",
    "Jamba2 Mini",
    "GLM-5.3-Flash",
  ]) {
    assert.ok(models.some((model) => model.name === expected), `${expected} should be seeded`);
  }
});

test("comparison editor loads catalog models and materializes them into spec blocks", async () => {
  const [collection, picker, route, data] = await Promise.all([
    source("components/admin/EditorialCollection.tsx"),
    source("components/admin/ComparisonModelPicker.tsx"),
    source("app/api/admin/models/route.ts"),
    source("lib/models.ts"),
  ]);

  assert.match(collection, /apiRequest<ModelCatalogEntry\[]>\("\/api\/admin\/models"\)/);
  assert.match(collection, /<ComparisonModelPicker/);
  assert.match(picker, /Model \{side\.toUpperCase\(\)\}/);
  assert.match(picker, /Custom \/ manual/);
  assert.match(picker, /modelA: nextModelASlug/);
  assert.match(picker, /modelB: nextModelBSlug/);
  assert.match(picker, /modelA\.comparisonData\[label\] \?\? ""/);
  assert.match(picker, /modelB\.comparisonData\[label\] \?\? ""/);
  assert.match(picker, /new Map\(\[\.\.\.current, \.\.\.additions\]/);
  assert.match(route, /isAuthenticated/);
  assert.match(route, /getModels/);
  assert.match(data, /FROM models/);
  assert.match(data, /ORDER BY developer ASC, release_date DESC/);
});
