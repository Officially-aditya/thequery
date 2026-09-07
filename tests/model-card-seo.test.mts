import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");

test("glossary model pages target model-card intent without depending on the comparison catalog", async () => {
  const page = await source("app/glossary/[term]/page.tsx");

  assert.match(page, /MODEL_GLOSSARY_CATEGORY = "Models & Architectures"/);
  assert.match(page, /MODEL_CARD_SIGNAL_MINIMUM = 2/);
  assert.match(page, /if \(term\.category !== MODEL_GLOSSARY_CATEGORY\) return false/);
  assert.match(page, /context window/);
  assert.match(page, /priced at\|pricing/);
  assert.match(page, /benchmarks/);
  assert.match(page, /api/);
  assert.match(page, /open\[- \]weights/);
  assert.match(page, /signalCount >= MODEL_CARD_SIGNAL_MINIMUM/);
  assert.doesNotMatch(page, /getModelOptions/);
  assert.doesNotMatch(page, /normalizedModelName/);
  assert.match(page, /`\$\{term\.name\} Model Card`/);
  assert.match(page, /`\$\{term\.name\} model card`/);
  assert.match(page, /`\$\{term\.name\} specs`/);
  assert.match(page, /`\$\{term\.name\} benchmarks`/);
  assert.match(page, /`\$\{term\.name\} pricing`/);
  assert.match(page, /alternates: \{ canonical: canonicalUrl \}/);
  assert.match(page, /"@type": "TechArticle"/);
});
