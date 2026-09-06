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
  assert.match(detail, /getModels\(\)/);
  assert.match(detail, /comparisonPicker=\{/);
  assert.match(renderer, /<ModelHeaderSelect/);
  assert.match(renderer, /side="a"/);
  assert.match(renderer, /side="b"/);
  assert.match(picker, /<select/);
  assert.match(picker, /optgroup/);
  assert.match(picker, /\/comparisons\/compare\?/);
  assert.match(picker, /comparisonByPair/);
  assert.doesNotMatch(picker, /Compare models/);
  assert.doesNotMatch(picker, /<button/);
});

test("database-only pairs render with the canonical comparison sections and inline picker", async () => {
  const [page, builder] = await Promise.all([
    source("app/comparisons/compare/page.tsx"),
    source("lib/model-comparison.ts"),
  ]);

  assert.match(page, /buildModelComparisonBlocks/);
  assert.match(page, /modelComparisonSources/);
  assert.match(page, /ContentBlocksRenderer/);
  assert.match(page, /comparisonPicker=\{/);
  assert.doesNotMatch(page, /<ModelPicker/);
  assert.match(builder, /title: "Specifications"/);
  assert.match(builder, /title: "Pricing"/);
  assert.match(builder, /title: "Capabilities & access"/);
  assert.match(builder, /title: "Coding"/);
  assert.match(builder, /title: "Math & reasoning"/);
  assert.match(builder, /title: "Knowledge"/);
  assert.match(builder, /title: "Agentic & computer use"/);
});
