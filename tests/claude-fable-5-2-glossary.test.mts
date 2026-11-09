import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const migrationPath = path.join(root, "db/migrations/130_add_claude_fable_5_2.sql");

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

test("Claude Fable 5.2 migration is registered and builds the glossary, model, and benchmark rows", async () => {
  const [runner, migration] = await Promise.all([
    readFile(path.join(root, "scripts/migrate.mjs"), "utf8"),
    readFile(migrationPath, "utf8"),
  ]);

  assert.match(runner, /130_add_claude_fable_5_2/);
  assert.match(migration, /'glossary:claude-fable-5-2'/);
  assert.match(migration, /'claude-fable-5-2'/);
  assert.match(migration, /'glossary\/claude-fable-5-2'/);
  assert.match(migration, /'content', fable_52_entry\.body/);
  assert.match(migration, /ON CONFLICT \(kind, slug, parent_slug\) DO UPDATE SET/);
  assert.match(migration, /DATE '2026-11-10'/);
  assert.match(migration, /'category', 'Models & Architectures'/);
  assert.match(migration, /'relatedTerms', jsonb_build_array/);
  assert.match(migration, /slug IN \('claude-fable-51', 'claude-fable-5', 'claude-opus-5-5', 'anthropic'\)/);
  assert.match(migration, /INSERT INTO models/);
  assert.match(migration, /INSERT INTO model_benchmarks/);
});

test("Claude Fable 5.2 migration keeps the statement splitter intact", async () => {
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

test("Claude Fable 5.2 seed body matches the migration body", async () => {
  const migration = await readFile(migrationPath, "utf8");
  const body = migration.match(/\$body\$([\s\S]*?)\$body\$::text AS body/)?.[1];
  const entry = glossary.find(({ slug }) => slug === "claude-fable-5-2");

  assert.ok(body);
  assert.ok(entry);
  assert.equal(entry.fullDef, body);
});

test("Claude Fable 5.2 entry carries specifications, benchmarks, and architectural nuance", () => {
  const entry = glossary.find(({ slug }) => slug === "claude-fable-5-2");

  assert.ok(entry);
  assert.equal(entry.name, "Claude Fable 5.2");
  assert.equal(entry.category, "Models & Architectures");
  assert.equal(entry.lastUpdated, "2026-11-10");

  assert.match(entry.fullDef, /Claude Fable 5\.2 is Anthropic's November 2026 premium frontier/);
  assert.match(entry.fullDef, /`claude-fable-5-2`/);
  assert.match(entry.fullDef, /## Core profile/);
  assert.match(entry.fullDef, /## Lineup positioning and generation changes/);
  assert.match(entry.fullDef, /## Benchmark profile/);
  assert.match(entry.fullDef, /## Pricing and cache economics/);
  assert.match(entry.fullDef, /## Safeguards and Mythos 5\.2/);
  assert.match(entry.fullDef, /## API and behavior changes/);
  assert.match(entry.fullDef, /## Applications and workflow fit/);
  assert.match(entry.fullDef, /## Bottom line/);

  assert.match(entry.fullDef, /Terminal-Bench 4\.0/);
  assert.match(entry.fullDef, /68\.2%/);
  assert.match(entry.fullDef, /Humanity's Last Exam/);
  assert.match(entry.fullDef, /FrontierCode v1\.1/);
  assert.match(entry.fullDef, /SWE-sweep/);
  assert.match(entry.fullDef, /\$10\.00 base input \/ \$50\.00 base output/);
  assert.match(entry.fullDef, /\$0\.25 per million tokens/);
});

test("Claude Fable 5.2 entry has no inline title or duplicated reference sections", () => {
  const entry = glossary.find(({ slug }) => slug === "claude-fable-5-2");
  assert.ok(entry);

  assert.ok(!/^# /.test(entry.fullDef));
  assert.ok(!entry.fullDef.includes("## References"));
  assert.ok(!entry.fullDef.includes("## Related Terms"));
  assert.ok(!entry.fullDef.includes("https://www.thequery.in/glossary/"));
  assert.equal(entry.references.length, 4);
});

test("Claude Fable 5.2 meets SEO length and keyword expectations", () => {
  const entry = glossary.find(({ slug }) => slug === "claude-fable-5-2");
  assert.ok(entry);

  assert.ok(entry.seoDescription.length >= 140, `Description too short: ${entry.seoDescription.length}`);
  assert.ok(entry.seoDescription.length <= 160, `Description too long: ${entry.seoDescription.length}`);
  assert.ok(entry.seoKeywords.includes("claude-fable-5-2"));
  assert.ok(entry.seoKeywords.includes("Claude Fable 5.2"));
});

test("Claude Fable 5.2 is recognized as a model card by page logic", () => {
  const entry = glossary.find(({ slug }) => slug === "claude-fable-5-2");
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

test("every Claude Fable 5.2 related term resolves in glossary or known database entries", () => {
  const entry = glossary.find(({ slug }) => slug === "claude-fable-5-2");
  const slugs = new Set(glossary.map((term) => term.slug));
  const databaseOnly = new Set(["claude-fable-51", "claude-sonnet-5-5", "gemini-4-argon", "gemini-4-carbon", "gpt-6-astra"]);

  assert.ok(entry);
  for (const slug of entry.relatedTerms) {
    assert.ok(slugs.has(slug) || databaseOnly.has(slug), `unresolved related term: ${slug}`);
  }
});
