import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const migrationPath = path.join(root, "db/migrations/133_add_gpt_6_2_astra.sql");

const glossary = JSON.parse(await readFile(path.join(root, "data/glossary.json"), "utf8")) as Array<{
  name: string;
  slug: string;
  category: string;
  lastUpdated: string;
  shortDef: string;
  fullDef: string;
  references: Array<{ title: string; url: string }>;
  seoDescription: string;
  seoKeywords: string[];
  relatedTerms: string[];
}>;

test("GPT-6.2 Astra migration is registered and builds the glossary, model, and benchmark rows", async () => {
  const [runner, migration] = await Promise.all([
    readFile(path.join(root, "scripts/migrate.mjs"), "utf8"),
    readFile(migrationPath, "utf8"),
  ]);

  assert.match(runner, /133_add_gpt_6_2_astra/);
  assert.match(migration, /'glossary:gpt-6-2-astra'/);
  assert.match(migration, /'gpt-6-2-astra'/);
  assert.match(migration, /'glossary\/gpt-6-2-astra'/);
  assert.match(migration, /'content', gpt_6_2_astra_entry\.body/);
  assert.match(migration, /ON CONFLICT \(kind, slug, parent_slug\) DO UPDATE SET/);
  assert.match(migration, /DATE '2026-11-17'/);
  assert.match(migration, /'category', 'Models & Architectures'/);
  assert.match(migration, /'relatedTerms', jsonb_build_array/);
  assert.match(migration, /slug IN \('gpt-6-astra', 'gpt-6-1-sol', 'gpt-6-1-terra', 'gpt-6-sol', 'openai'\)/);
  assert.match(migration, /INSERT INTO models/);
  assert.match(migration, /INSERT INTO model_benchmarks/);
});

test("GPT-6.2 Astra migration keeps the statement splitter intact", async () => {
  const migration = await readFile(migrationPath, "utf8");
  const statements = migration
    .split(/;\s*(?:\r?\n|$)/)
    .map((statement) => statement.trim())
    .filter(Boolean);

  assert.equal(statements.length, 4);
  assert.match(statements[0], /INSERT INTO content_items/);
  assert.match(statements[1], /UPDATE content_items/);
  assert.match(statements[2], /INSERT INTO models/);
  assert.match(statements[3], /INSERT INTO model_benchmarks/);
});

test("GPT-6.2 Astra seed body matches the migration body", async () => {
  const migration = await readFile(migrationPath, "utf8");
  const body = migration.match(/\$body\$([\s\S]*?)\$body\$::text AS body/)?.[1];
  const entry = glossary.find(({ slug }) => slug === "gpt-6-2-astra");

  assert.ok(body);
  assert.ok(entry);
  assert.equal(entry.fullDef, body);
});

test("GPT-6.2 Astra entry carries specifications, benchmarks, and frontier capability facts", () => {
  const entry = glossary.find(({ slug }) => slug === "gpt-6-2-astra");

  assert.ok(entry);
  assert.equal(entry.name, "GPT-6.2 Astra");
  assert.equal(entry.category, "Models & Architectures");
  assert.equal(entry.lastUpdated, "2026-11-17");

  assert.match(entry.fullDef, /GPT-6\.2 Astra is OpenAI's November 2026 frontier flagship/);
  assert.match(entry.fullDef, /`gpt-6\.2-astra`/);
  assert.match(entry.fullDef, /## Core profile/);
  assert.match(entry.fullDef, /## Lineup positioning and generation changes/);
  assert.match(entry.fullDef, /## Benchmark profile/);
  assert.match(entry.fullDef, /## Pricing and cost efficiency/);
  assert.match(entry.fullDef, /## Safeguards and system card profile/);
  assert.match(entry.fullDef, /## API and behavior changes/);
  assert.match(entry.fullDef, /## Applications and workflow fit/);
  assert.match(entry.fullDef, /## Bottom line/);

  assert.match(entry.fullDef, /Terminal-Bench-Science 0\.1/);
  assert.match(entry.fullDef, /77\.1%/);
  assert.match(entry.fullDef, /Agents' Last Exam/);
  assert.match(entry.fullDef, /77\.6%/);
  assert.match(entry.fullDef, /GPQA Diamond/);
  assert.match(entry.fullDef, /97\.0%/);
  assert.match(entry.fullDef, /FrontierCode 2\.0/);
  assert.match(entry.fullDef, /49\.8%/);
  assert.match(entry.fullDef, /Control Room/);
  assert.match(entry.fullDef, /Preparedness Framework/);
  assert.match(entry.fullDef, /8\.00 base input \/ .*40\.00 base output/);
  assert.match(entry.fullDef, /0\.40 per million tokens/);
});

test("GPT-6.2 Astra entry has no inline title or duplicated reference sections", () => {
  const entry = glossary.find(({ slug }) => slug === "gpt-6-2-astra");
  assert.ok(entry);

  assert.ok(!/^# /.test(entry.fullDef));
  assert.ok(!entry.fullDef.includes("## References"));
  assert.ok(!entry.fullDef.includes("## Related Terms"));
  assert.ok(!entry.fullDef.includes("https://www.thequery.in/glossary/"));
  assert.equal(entry.references.length, 4);
});

test("GPT-6.2 Astra meets SEO length and keyword expectations", () => {
  const entry = glossary.find(({ slug }) => slug === "gpt-6-2-astra");
  assert.ok(entry);

  assert.ok(entry.seoDescription.length >= 140, `Description too short: ${entry.seoDescription.length}`);
  assert.ok(entry.seoDescription.length <= 160, `Description too long: ${entry.seoDescription.length}`);
  assert.ok(entry.seoKeywords.includes("gpt-6-2-astra"));
  assert.ok(entry.seoKeywords.includes("gpt-6.2-astra"));
  assert.ok(entry.seoKeywords.includes("GPT-6.2 Astra"));
});

test("GPT-6.2 Astra is recognized as a model card by page logic", () => {
  const entry = glossary.find(({ slug }) => slug === "gpt-6-2-astra");
  assert.ok(entry);

  const MODEL_GLOSSARY_CATEGORY = "Models & Architectures";
  const MODEL_CARD_SIGNALS = [
    /\b(?:released|launched|introduced|announced)\b/i,
    /\bcontext window\b/i,
    /\b(?:input|output) tokens?\b/i,
    /\b(?:priced at|pricing|per million tokens?|cache reads?|cache writes?)\b/i,
    /\bbenchmarks?\b/i,
    /\bapi\b/i,
    /\b(?:open[- ]weights?|downloadable weights?|model weights?)\b/i,
    /\b(?:knowledge cutoff|max output|reasoning effort|tool calling)\b/i,
  ];
  const MODEL_CARD_STRONG_SIGNALS = [
    /\bcontext window\b/i,
    /\b(?:input|output) tokens?\b/i,
    /\b(?:priced at|pricing|per million tokens?|cache reads?|cache writes?)\b/i,
    /\b(?:open[- ]weights?|downloadable weights?|model weights?)\b/i,
    /\b(?:knowledge cutoff|max output|reasoning effort|tool calling)\b/i,
  ];
  const MODEL_CARD_VERSION_CUE = /\d/;

  const searchable = [
    entry.name,
    entry.shortDef,
    entry.fullDef,
    ...(entry.seoKeywords ?? []),
  ].join(" ");

  const signalCount = MODEL_CARD_SIGNALS.reduce(
    (count, signal) => count + (signal.test(searchable) ? 1 : 0),
    0,
  );
  assert.ok(signalCount >= 2);
  const hasStrongSignal = MODEL_CARD_STRONG_SIGNALS.some((signal) => signal.test(searchable));
  assert.ok(hasStrongSignal);
  assert.ok(MODEL_CARD_VERSION_CUE.test(entry.name));
  assert.equal(entry.category, MODEL_GLOSSARY_CATEGORY);
});

test("every GPT-6.2 Astra related term resolves in glossary or known database entries", () => {
  const entry = glossary.find(({ slug }) => slug === "gpt-6-2-astra");
  const slugs = new Set(glossary.map((term) => term.slug));
  const databaseOnly = new Set([
    "gemini-4-carbon",
    "gemini-4-argon",
    "gemini-3-8-flash",
    "claude-fable-51",
    "claude-sonnet-5-5",
    "gpt-6-astra",
  ]);

  assert.ok(entry);
  for (const slug of entry.relatedTerms) {
    assert.ok(slugs.has(slug) || databaseOnly.has(slug), `unresolved related term: ${slug}`);
  }
});
