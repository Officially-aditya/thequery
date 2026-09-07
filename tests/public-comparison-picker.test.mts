import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

async function source(file: string) {
  return readFile(path.join(root, file), "utf8");
}

test("comparison model picker is only the two database-backed spec-header dropdowns", async () => {
  const [index, detail, renderer, picker] = await Promise.all([
    source("app/comparisons/page.tsx"),
    source("app/comparisons/[slug]/page.tsx"),
    source("components/content/ContentBlocksRenderer.tsx"),
    source("components/comparisons/ModelPicker.tsx"),
  ]);

  assert.doesNotMatch(index, /<ModelPicker/);
  assert.doesNotMatch(index, /Compare models/);
  assert.match(detail, /getModelOptions\(\)/);
  assert.match(detail, /getComparisonPairs\(\)/);
  assert.doesNotMatch(detail, /getModels\(\)/);
  assert.match(detail, /comparisonPicker=\{/);
  assert.match(renderer, /<ModelHeaderSelect/);
  assert.match(renderer, /side="a"/);
  assert.match(renderer, /side="b"/);
  assert.match(picker, /<select/);
  assert.match(picker, /optgroup/);
  assert.match(picker, /canonicalComparisonSlug/);
  assert.doesNotMatch(picker, /\/comparisons\/compare\?/);
  assert.match(picker, /comparisonByPair/);
  assert.doesNotMatch(picker, /access:/);
  assert.doesNotMatch(picker, /Compare models/);
  assert.doesNotMatch(picker, /<button/);
});

test("database-only pairs render from the canonical comparison slug route", async () => {
  const [page, legacy, builder, routing] = await Promise.all([
    source("app/comparisons/[slug]/page.tsx"),
    source("app/comparisons/compare/page.tsx"),
    source("lib/model-comparison.ts"),
    source("lib/model-comparison-route.ts"),
  ]);

  assert.match(page, /resolveComparisonSlug/);
  assert.match(page, /getModelsBySlugs\(\[modelASlug, modelBSlug\]\)/);
  assert.match(page, /canonicalComparisonSlug/);
  assert.match(page, /alternates: \{ canonical: canonicalUrl \}/);
  assert.match(page, /buildModelComparisonBlocks/);
  assert.match(page, /modelComparisonSources/);
  assert.match(page, /permanentRedirect\(`\/comparisons\/\$\{canonicalSlug\}`\)/);
  assert.doesNotMatch(page, /getModels\(\)/);

  assert.match(legacy, /permanentRedirect/);
  assert.match(legacy, /canonicalComparisonSlug/);
  assert.doesNotMatch(legacy, /ContentBlocksRenderer/);
  assert.doesNotMatch(legacy, /getModelsBySlugs/);

  assert.match(routing, /normalized\.startsWith\("claude-"\)/);
  assert.match(routing, /canonicalComparisonModels/);
  assert.match(routing, /resolveComparisonSlug/);

  assert.match(builder, /title: "Specifications"/);
  assert.match(builder, /title: "Pricing"/);
  assert.match(builder, /title: "Capabilities & access"/);
  assert.match(builder, /title: "Coding"/);
  assert.match(builder, /title: "Math & reasoning"/);
  assert.match(builder, /title: "Knowledge"/);
  assert.match(builder, /title: "Agentic & computer use"/);
});
