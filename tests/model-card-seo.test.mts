import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");

test("glossary model pages target model-card search intent without relabeling architecture terms", async () => {
  const page = await source("app/glossary/[term]/page.tsx");

  assert.match(page, /MODEL_GLOSSARY_CATEGORY = "Models & Architectures"/);
  assert.match(page, /if \(term\.category !== MODEL_GLOSSARY_CATEGORY\) return false/);
  assert.match(page, /getModelOptions\(\)/);
  assert.match(page, /normalizedModelName\(model\.name\) === modelName/);
  assert.match(page, /`\$\{term\.name\} Model Card`/);
  assert.match(page, /`\$\{term\.name\} model card`/);
  assert.match(page, /`\$\{term\.name\} specs`/);
  assert.match(page, /`\$\{term\.name\} benchmarks`/);
  assert.match(page, /`\$\{term\.name\} pricing`/);
  assert.match(page, /alternates: \{ canonical: canonicalUrl \}/);
  assert.match(page, /"@type": "TechArticle"/);
});
