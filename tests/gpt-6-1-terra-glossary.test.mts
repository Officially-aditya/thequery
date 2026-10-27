import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const migrationPath = path.join(root, "db/migrations/125_add_gpt_6_1_terra.sql");

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

test("GPT-6.1 Terra migration is registered and builds the glossary and model rows", async () => {
  const [runner, migration] = await Promise.all([
    readFile(path.join(root, "scripts/migrate.mjs"), "utf8"),
    readFile(migrationPath, "utf8"),
  ]);

  assert.match(runner, /125_add_gpt_6_1_terra/);
  assert.match(migration, /'glossary:gpt-6-1-terra'/);
  assert.match(migration, /'gpt-6-1-terra'/);
  assert.match(migration, /'glossary\/gpt-6-1-terra'/);
  assert.match(migration, /'content', gpt_6_1_terra_entry\.body/);
  assert.match(migration, /ON CONFLICT \(kind, slug, parent_slug\) DO UPDATE SET/);
  assert.match(migration, /DATE '2026-10-26'/);
  assert.match(migration, /'category', 'Models & Architectures'/);
  assert.match(migration, /'relatedTerms', jsonb_build_array\('gpt-6-1-sol'/);
  assert.match(migration, /slug IN \('gpt-6-1-sol', 'gpt-6-luna', 'gpt-6-sol'\)/);
  assert.match(migration, /Introducing GPT-6\.1 Terra/);
  assert.match(migration, /INSERT INTO models/);
  assert.match(migration, /INSERT INTO model_benchmarks/);
});

test("GPT-6.1 Terra migration keeps the statement splitter intact", async () => {
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
  assert.ok(!statements[1].includes("SELECT $body$"));
});

test("GPT-6.1 Terra seeds only the verified benchmark row", async () => {
  const migration = await readFile(migrationPath, "utf8");

  assert.match(migration, /"tq-20261026-gpt61terra-terminalbench40"/);

  for (const id of [
    "tq-20261026-gpt61terra-osworld21",
    "tq-20261026-gpt61terra-ale-medium",
    "tq-20261026-gpt61terra-mmmupro",
    "tq-20261026-gpt61terra-dualentry",
    "tq-20261026-gpt61terra-omniscience",
  ]) {
    assert.ok(!migration.includes(id), `migration should not seed ${id}`);
  }
});

test("GPT-6.1 Terra uses a single migration, not a duplicate correction entry", async () => {
  const runner = await readFile(path.join(root, "scripts/migrate.mjs"), "utf8");
  const files = (await readdir(path.join(root, "db/migrations"))).filter((name) =>
    name.includes("gpt_6_1_terra"),
  );

  assert.deepEqual(files, ["125_add_gpt_6_1_terra.sql"]);
  assert.match(runner, /125_add_gpt_6_1_terra/);
  assert.doesNotMatch(runner, /126_correct_gpt_6_1_terra/);
});

test("the seed body matches the migration body", async () => {
  const migration = await readFile(migrationPath, "utf8");
  const body = migration.match(/\$body\$([\s\S]*?)\$body\$::text AS body/)?.[1];
  const entry = glossary.find(({ slug }) => slug === "gpt-6-1-terra");

  assert.ok(body);
  assert.ok(entry);
  assert.equal(entry.fullDef, body);
});

test("GPT-6.1 Terra entry carries the release facts, specifications and cross-links", async () => {
  const migration = await readFile(migrationPath, "utf8");

  assert.match(migration, /GPT-6\.1 Terra is OpenAI's October 2026 \[large language model\]\(\/glossary\/large-language-model\)/);
  assert.match(migration, /available in the API as `gpt-6\.1-terra`/);
  assert.match(migration, /## Key specifications/);
  assert.match(migration, /## Lineup positioning and generation changes/);
  assert.match(migration, /## Pricing and cost efficiency/);
  assert.match(migration, /## Benchmark profile and independent evaluations/);
  assert.match(migration, /## API behavior and reasoning configuration/);
  assert.match(migration, /## Safeguards and system card profile/);
  assert.match(migration, /## Workflow fit and bottom line/);
  assert.match(migration, /1,050,000 tokens/);
  assert.match(migration, /June 1, 2026/);
});

test("GPT-6.1 Terra entry keeps the vendor and evaluation caveats in the body", () => {
  const entry = glossary.find(({ slug }) => slug === "gpt-6-1-terra");

  assert.ok(entry);
  assert.equal(entry.name, "GPT-6.1 Terra");
  assert.equal(entry.category, "Models & Architectures");
  assert.equal(entry.lastUpdated, "2026-10-26");

  assert.match(entry.fullDef, /reasoning\.effort: "none"/);
  assert.match(entry.fullDef, /Terminal-Bench 4\.0 at maximum effort.*50\.3%/);
  assert.match(entry.fullDef, /OSWorld 2\.1.*surpasses GPT-5\.6 Terra/);
  assert.match(entry.fullDef, /48 on its Intelligence Index v4\.3\.2/);
  assert.match(entry.fullDef, /internal factuality error rate.*53\.6% to 16\.7%/);
  assert.match(entry.fullDef, /Treated as Critical capability/);
  assert.match(entry.fullDef, /same safeguards stack to all three models/);
});

test("GPT-6.1 Terra entry has no inline title or duplicated reference sections", () => {
  const entry = glossary.find(({ slug }) => slug === "gpt-6-1-terra");
  assert.ok(entry);

  assert.ok(!/^# /.test(entry.fullDef));
  assert.ok(!entry.fullDef.includes("## References"));
  assert.ok(!entry.fullDef.includes("## Related Terms"));
  assert.ok(!entry.fullDef.includes("https://www.thequery.in/glossary/"));
  assert.equal(entry.references.length, 4);
});

test("GPT-6.1 Terra entry preserves tables and SEO field bounds", () => {
  const entry = glossary.find(({ slug }) => slug === "gpt-6-1-terra");
  assert.ok(entry);

  assert.ok((entry.fullDef.match(/^\| ---/gm)?.length ?? 0) >= 2);
  assert.ok(entry.fullDef.includes("solves 50.3% of tasks"));
  assert.ok(entry.fullDef.includes("48 on its Intelligence Index v4.3.2"));
  assert.ok(entry.fullDef.includes("| Standard input | $1.00 | $2.00 |"));
  assert.ok(entry.seoDescription.length >= 140);
  assert.ok(entry.seoDescription.length <= 160);
  assert.ok(entry.seoKeywords.includes("gpt-6-1-terra"));
  assert.ok(entry.seoKeywords.includes("GPT-6.1 Terra model card"));
});

test("every GPT-6.1 Terra related term resolves in the glossary", () => {
  const entry = glossary.find(({ slug }) => slug === "gpt-6-1-terra");
  const slugs = new Set(glossary.map((term) => term.slug));
  const databaseOnly = new Set(["gpt-6-astra"]);

  assert.ok(entry);
  for (const slug of entry.relatedTerms) {
    assert.ok(slugs.has(slug) || databaseOnly.has(slug), `unresolved related term: ${slug}`);
  }
});
