import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

const migrationPath = path.join(root, "db/migrations/090_add_gpt_6_1_sol.sql");

const glossary = JSON.parse(await readFile(path.join(root, "data/glossary.json"), "utf8")) as Array<{
  name: string;
  slug: string;
  category: string;
  lastUpdated: string;
  fullDef: string;
  references: Array<{ title: string; url: string }>;
  seoDescription: string;
  seoKeywords: string[];
  relatedTerms: string[];
}>;

test("GPT-6.1 Sol migration is registered and builds the glossary row", async () => {
  const [runner, migration] = await Promise.all([
    readFile(path.join(root, "scripts/migrate.mjs"), "utf8"),
    readFile(migrationPath, "utf8"),
  ]);

  assert.match(runner, /090_add_gpt_6_1_sol/);
  assert.match(migration, /'glossary:gpt-6-1-sol'/);
  assert.match(migration, /'gpt-6-1-sol'/);
  assert.match(migration, /'glossary\/gpt-6-1-sol'/);
  assert.match(migration, /'content', gpt_6_1_sol_entry\.body/);
  assert.match(migration, /ON CONFLICT \(kind, slug, parent_slug\) DO UPDATE SET/);
  assert.match(migration, /DATE '2026-09-30'/);
  assert.match(migration, /'category', 'Models & Architectures'/);
});

test("GPT-6.1 Sol migration keeps the statement splitter intact", async () => {
  const migration = await readFile(migrationPath, "utf8");
  const statements = migration
    .split(/;\s*(?:\r?\n|$)/)
    .map((statement) => statement.trim())
    .filter(Boolean);

  assert.equal(statements.length, 2);
  assert.match(statements[0], /INSERT INTO content_items/);
  assert.match(statements[1], /UPDATE content_items/);
  assert.ok(!statements[1].includes("SELECT $body$"));
});

test("GPT-6.1 Sol entry carries the release facts and cross-links", async () => {
  const migration = await readFile(migrationPath, "utf8");

  assert.match(migration, /GPT-6\.1 Sol is OpenAI's September 2026 \[large language model\]\(\/glossary\/large-language-model\)/);
  assert.match(migration, /available in the API as `gpt-6\.1-sol`/);
  assert.match(migration, /## Core profile/);
  assert.match(migration, /## Benchmark profile/);
  assert.match(migration, /## Pricing and efficiency/);
  assert.match(migration, /## Safeguards/);
  assert.match(migration, /## API and behavior changes/);
  assert.match(migration, /## Applications and workflow fit/);
  assert.match(migration, /## Bottom line/);
  assert.match(migration, /1,050,000-token \[context window\]\(\/glossary\/context-window\)/);
  assert.match(migration, /knowledge cutoff of April 30, 2026/);
  assert.match(migration, /Introducing GPT-6\.1 Sol/);
  assert.match(migration, /'relatedTerms', jsonb_build_array\('gpt-6-sol'/);
  assert.match(migration, /slug IN \('gpt-6-sol', 'gpt-6-astra'\)/);
});

test("GPT-6.1 Sol entry keeps the vendor caveats in the body", () => {
  const entry = glossary.find(({ slug }) => slug === "gpt-6-1-sol");

  assert.ok(entry);
  assert.equal(entry.name, "GPT-6.1 Sol");
  assert.equal(entry.category, "Models & Architectures");
  assert.equal(entry.lastUpdated, "2026-09-30");

  assert.match(entry.fullDef, /as transcribed by Handy AI/);
  assert.match(entry.fullDef, /DeepSWE's 75\.2% is at high effort and falls to 71\.9% at max/);
  assert.match(entry.fullDef, /at max effort Opus 5\.5 leads by 6\.4 points/);
  assert.match(entry.fullDef, /structured-output bug that Artificial Analysis says it will re-run/);
  assert.match(entry.fullDef, /may be inflated by contamination from historical vulnerabilities/);
  assert.match(entry.fullDef, /1\.50 percent, against 0\.51 percent for Astra/);
});

test("GPT-6.1 Sol entry has no inline title or duplicated reference sections", () => {
  const entry = glossary.find(({ slug }) => slug === "gpt-6-1-sol");
  assert.ok(entry);

  assert.ok(!/^# /.test(entry.fullDef));
  assert.ok(!entry.fullDef.includes("## References"));
  assert.ok(!entry.fullDef.includes("## Related Terms"));
  assert.ok(!entry.fullDef.includes("https://www.thequery.in/glossary/"));
  assert.equal(entry.references.length, 4);
});

test("GPT-6.1 Sol entry keeps the three comparison tables and the SEO fields", () => {
  const entry = glossary.find(({ slug }) => slug === "gpt-6-1-sol");
  assert.ok(entry);

  assert.equal(entry.fullDef.match(/^\| ---/gm)?.length, 4);
  assert.equal(entry.fullDef.match(/^\| /gm)?.length, 31);
  assert.ok(entry.fullDef.includes("| DeepSWE v1.1 (high) | **75.2%** |"));
  assert.ok(entry.fullDef.includes("| Intelligence Index v4.3.2 |"));
  assert.ok(entry.fullDef.includes("| Cached input | $0.10 | $0.20 |"));
  assert.ok(entry.fullDef.includes("| max | 51.8 / $0.72 | 52.7 / $3.26 |"));
  assert.ok(entry.seoDescription.length >= 140);
  assert.ok(entry.seoDescription.length <= 160);
  assert.ok(entry.seoKeywords.includes("gpt-6-1-sol"));
});

test("every GPT-6.1 Sol related term resolves in the glossary", () => {
  const entry = glossary.find(({ slug }) => slug === "gpt-6-1-sol");
  const slugs = new Set(glossary.map((term) => term.slug));
  // gpt-6-astra is renamed in the database by migration 072 and is not seeded
  // from data/glossary.json, so it cannot be resolved from this file.
  const databaseOnly = new Set(["gpt-6-astra"]);

  assert.ok(entry);
  for (const slug of entry.relatedTerms) {
    assert.ok(slugs.has(slug) || databaseOnly.has(slug), `unresolved related term: ${slug}`);
  }
});
