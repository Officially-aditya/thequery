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
  const coding = client.indexOf('title: "Coding"');
  const math = client.indexOf('title: "Math & reasoning"');
  const knowledge = client.indexOf('title: "Knowledge"');
  const agentic = client.indexOf('title: "Agentic & computer use"');

  assert.ok(specifications >= 0);
  assert.ok(pricing > specifications);
  assert.ok(capabilities > pricing);
  assert.ok(coding > capabilities);
  assert.ok(math > coding);
  assert.ok(knowledge > math);
  assert.ok(agentic > knowledge);
  assert.doesNotMatch(client, /title: "Benchmarks"/);

  for (const row of [
    "Text input",
    "Image / vision input",
    "Audio input",
    "Video input",
    "Text output",
    "Image output",
    "Audio output",
    "Video output",
    "Tool / function calling",
    "Computer use",
    "API access",
    "Product access",
    "Weights / license",
    "SWE-bench Verified",
    "DeepSWE v1.1",
    "Terminal-Bench 2.1",
    "Terminal-Bench 4.0",
    "FrontierMath",
    "GPQA Diamond",
    "HLE-Verified",
    "OSWorld 2.0",
    "GDPval-AA v2",
  ]) {
    assert.match(client, new RegExp(row.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  }

  assert.match(collection, /kind === "comparison" \? comparisonTemplateBlocks\(\) : undefined/);
});

test("the existing Fable Astra comparison gets capabilities immediately after pricing", async () => {
  const [runner, migration, comparisons] = await Promise.all([
    source("scripts/migrate.mjs"),
    source("db/migrations/010_comparison_template_capabilities.sql"),
    source("lib/comparisons.ts"),
  ]);

  assert.match(runner, /010_comparison_template_capabilities/);
  assert.match(migration, /'title', 'Capabilities & access'/);
  assert.match(migration, /lower\(pricing\.element->>'title'\) = 'pricing'/);
  assert.match(migration, /comparison\.title ILIKE '%Fable 5\.1%'/);
  assert.match(migration, /comparison\.title ILIKE '%Astra%'/);
  assert.match(migration, /'Image \/ vision input'/);
  assert.match(migration, /'Weights \/ license'/);

  assert.match(comparisons, /function withFlagshipCapabilities/);
  assert.match(comparisons, /pricingIndex \+ 1/);
  assert.match(comparisons, /title: "Capabilities & access"/);
  assert.match(comparisons, /alreadyHasCapabilities/);
});

test("existing model comparisons are migrated to the same canonical layout", async () => {
  const [runner, migration] = await Promise.all([
    source("scripts/migrate.mjs"),
    source("db/migrations/014_canonicalize_existing_comparisons.sql"),
  ]);

  assert.match(runner, /014_canonicalize_existing_comparisons/);
  assert.match(migration, /split_part\(c\.title, ' vs ', 1\)/);
  assert.match(migration, /split_part\(c\.title, ' vs ', 2\)/);
  assert.match(migration, /m\.slug = c\.metadata->>'modelA'/);
  assert.match(migration, /m\.slug = c\.metadata->>'modelB'/);
  assert.match(migration, /NULLIF\(er\.values_by_label->lower\(l\.label\)->>0, ''\)/);
  assert.match(migration, /p\.model_a_data->>l\.label/);
  assert.match(migration, /'Capabilities & access'/);
  assert.match(migration, /'Coding'/);
  assert.match(migration, /'Math & reasoning'/);
  assert.match(migration, /'Knowledge'/);
  assert.match(migration, /'Agentic & computer use'/);
  assert.match(migration, /'DeepSWE v1\.1'/);
  assert.match(migration, /'Terminal-Bench 4\.0'/);
  assert.match(migration, /'OSWorld 2\.0'/);
  assert.match(migration, /'GDPval-AA v2'/);
  assert.match(migration, /'markdown-canonical-bottom-line'/);
});
