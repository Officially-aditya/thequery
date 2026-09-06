import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

async function source(file: string) {
  return readFile(path.join(root, file), "utf8");
}

test("public comparisons load model options from the database", async () => {
  const [index, detail, picker] = await Promise.all([
    source("app/comparisons/page.tsx"),
    source("app/comparisons/[slug]/page.tsx"),
    source("components/comparisons/ModelPicker.tsx"),
  ]);

  assert.match(index, /getModels\(\)/);
  assert.match(index, /<ModelPicker/);
  assert.match(detail, /getModels\(\)/);
  assert.match(detail, /initialModelA=\{comparison\.modelA\}/);
  assert.match(detail, /initialModelB=\{comparison\.modelB\}/);
  assert.match(picker, /Choose a model/);
  assert.match(picker, /optgroup/);
  assert.match(picker, /\/comparisons\/compare\?/);
  assert.match(picker, /comparisonByPair/);
});

test("database-only pairs render with the canonical comparison sections", async () => {
  const [page, builder] = await Promise.all([
    source("app/comparisons/compare/page.tsx"),
    source("lib/model-comparison.ts"),
  ]);

  assert.match(page, /buildModelComparisonBlocks/);
  assert.match(page, /modelComparisonSources/);
  assert.match(page, /ContentBlocksRenderer/);
  assert.match(builder, /title: "Specifications"/);
  assert.match(builder, /title: "Pricing"/);
  assert.match(builder, /title: "Capabilities & access"/);
  assert.match(builder, /title: "Coding"/);
  assert.match(builder, /title: "Math & reasoning"/);
  assert.match(builder, /title: "Knowledge"/);
  assert.match(builder, /title: "Agentic & computer use"/);
});
