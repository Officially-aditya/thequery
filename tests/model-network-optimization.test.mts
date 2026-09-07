import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");

test("model option reads never select heavyweight catalog columns", async () => {
  const models = await source("lib/models.ts");
  const optionQuery = models.match(/async function queryModelOptions[\s\S]*?\) as ModelCatalogOption\[\];/)?.[0] ?? "";

  assert.match(optionQuery, /SELECT slug, name, developer, access/);
  assert.doesNotMatch(optionQuery, /comparison_data/);
  assert.doesNotMatch(optionQuery, /sources/);
  assert.doesNotMatch(optionQuery, /notes/);
  assert.doesNotMatch(optionQuery, /model_benchmarks/);
  assert.match(models, /MODEL_OPTION_CACHE_SECONDS = 300/);
});

test("full model detail reads are bounded to requested slugs", async () => {
  const models = await source("lib/models.ts");

  assert.match(models, /export async function getModelsBySlugs/);
  assert.match(models, /WHERE slug = ANY\(\$1::text\[\]\)/);
  assert.match(models, /WHERE model_slug = ANY\(\$1::text\[\]\)/);
  assert.match(models, /export async function getModelBySlug[\s\S]*getModelsBySlugs\(\[slug\]\)/);
});

test("public comparison pages do not load the full model catalog", async () => {
  const [generated, authored] = await Promise.all([
    source("app/comparisons/compare/page.tsx"),
    source("app/comparisons/[slug]/page.tsx"),
  ]);

  assert.match(generated, /getModelsBySlugs\(\[modelASlug, modelBSlug\]\)/);
  assert.match(generated, /getModelOptions\(\)/);
  assert.match(generated, /getComparisonPairs\(\)/);
  assert.doesNotMatch(generated, /getModels\(\)/);

  assert.match(authored, /getModelOptions\(\)/);
  assert.match(authored, /getComparisonPairs\(\)/);
  assert.doesNotMatch(authored, /getModels\(\)/);
  assert.doesNotMatch(authored, /getModelsBySlugs/);
});

test("public client receives identity-only model options", async () => {
  const picker = await source("components/comparisons/ModelPicker.tsx");
  const optionType = picker.match(/export type PublicModelOption = \{[\s\S]*?\};/)?.[0] ?? "";

  assert.match(optionType, /slug: string/);
  assert.match(optionType, /name: string/);
  assert.match(optionType, /developer: string/);
  assert.doesNotMatch(optionType, /access/);
  assert.doesNotMatch(optionType, /comparisonData/);
  assert.doesNotMatch(optionType, /sources/);
});

test("admin loads model details lazily and does not persistently cache them", async () => {
  const [route, picker] = await Promise.all([
    source("app/api/admin/models/route.ts"),
    source("components/admin/ComparisonModelPicker.tsx"),
  ]);

  assert.match(route, /getModelOptions/);
  assert.match(route, /getModelBySlug/);
  assert.match(route, /private, max-age=300/);
  assert.match(route, /private, no-store/);
  assert.match(picker, /\/api\/admin\/models\?slug=/);
  assert.match(picker, /useRef\(new Map<string, ModelCatalogDetail>\(\)\)/);
  assert.doesNotMatch(picker, /comparisonData: Record<string, string>[\s\S]*export interface ModelCatalogEntry/);
});

test("authored-comparison routing reuses the existing cached summary index", async () => {
  const comparisons = await source("lib/comparisons.ts");
  const pairHelper = comparisons.match(/export async function getComparisonPairs[\s\S]*?\n\}/)?.[0] ?? "";

  assert.match(pairHelper, /getContentSummaries\("comparison"\)/);
  assert.match(pairHelper, /metadataModelSlug/);
  assert.doesNotMatch(pairHelper, /getSql/);
  assert.doesNotMatch(pairHelper, /SELECT /);
});
