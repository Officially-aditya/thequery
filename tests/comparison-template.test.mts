import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

async function source(file: string) {
  return readFile(path.join(root, file), "utf8");
}

test("new comparisons use the standard spec template", async () => {
  const [client, collection] = await Promise.all([
    source("components/admin/admin-client.ts"),
    source("components/admin/EditorialCollection.tsx"),
  ]);

  const specifications = client.indexOf('title: "Specifications"');
  const pricing = client.indexOf('title: "Pricing"');
  const capabilities = client.indexOf('title: "Capabilities & access"');
  const benchmarks = client.indexOf('title: "Benchmarks"');

  assert.ok(specifications >= 0);
  assert.ok(pricing > specifications);
  assert.ok(capabilities > pricing);
  assert.ok(benchmarks > capabilities);

  for (const row of [
    "Text input",
    "Image / vision input",
    "Audio input",
    "Video input",
    "Text output",
    "Audio output",
    "Tool / function calling",
    "Computer use",
    "API access",
    "Product access",
    "Weights / license",
  ]) {
    assert.match(client, new RegExp(row.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  }

  assert.match(collection, /kind === "comparison" \? comparisonTemplateBlocks\(\) : undefined/);
});

test("the existing Fable Astra comparison gets capabilities immediately after pricing", async () => {
  const [runner, migration] = await Promise.all([
    source("scripts/migrate.mjs"),
    source("db/migrations/010_comparison_template_capabilities.sql"),
  ]);

  assert.match(runner, /010_comparison_template_capabilities/);
  assert.match(migration, /'title', 'Capabilities & access'/);
  assert.match(migration, /lower\(pricing\.element->>'title'\) = 'pricing'/);
  assert.match(migration, /comparison\.title ILIKE '%Fable 5\.1%'/);
  assert.match(migration, /comparison\.title ILIKE '%Astra%'/);
  assert.match(migration, /'Image \/ vision input'/);
  assert.match(migration, /'Weights \/ license'/);
});
