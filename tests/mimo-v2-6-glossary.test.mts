import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

const migrationPath = path.join(root, "db/migrations/093_add_mimo_v2_6.sql");

const glossary = JSON.parse(
  await readFile(path.join(root, "data/glossary.json"), "utf8"),
) as Array<{
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

const series = glossary.find(({ slug }) => slug === "mimo-v2-6");
const pro = glossary.find(({ slug }) => slug === "mimo-v2-6-pro");

test("MiMo V2.6 migration is registered and builds both glossary rows", async () => {
  const [runner, migration] = await Promise.all([
    readFile(path.join(root, "scripts/migrate.mjs"), "utf8"),
    readFile(migrationPath, "utf8"),
  ]);

  assert.match(runner, /093_add_mimo_v2_6/);
  assert.match(migration, /'glossary:mimo-v2-6'/);
  assert.match(migration, /'glossary:mimo-v2-6-pro'/);
  assert.match(migration, /'glossary\/mimo-v2-6'/);
  assert.match(migration, /'glossary\/mimo-v2-6-pro'/);
  assert.match(migration, /'content', mimo_v2_6_entry\.body/);
  assert.match(migration, /'content', mimo_v2_6_pro_entry\.body/);
  assert.match(
    migration,
    /ON CONFLICT \(kind, slug, parent_slug\) DO UPDATE SET/,
  );
  assert.match(migration, /DATE '2026-09-30'/);
  assert.equal((migration.match(/'category', 'Models & Architectures'/g) ?? []).length, 2);
});

test("MiMo V2.6 migration keeps the statement splitter intact", async () => {
  const migration = await readFile(migrationPath, "utf8");
  const statements = migration
    .split(/;\s*(?:\r?\n|$)/)
    .map((statement) => statement.trim())
    .filter(Boolean);

  assert.equal(statements.length, 2);
  for (const statement of statements) {
    assert.match(statement, /INSERT INTO content_items/);
  }
  assert.equal((migration.match(/\$body\$/g) ?? []).length, 4);
});

test("MiMo V2.6 series entry carries the release facts", async () => {
  const migration = await readFile(migrationPath, "utf8");

  assert.match(migration, /MiMo V2\.6 is Xiaomi's September 2026 model series/);
  assert.match(migration, /MIT License/);
  assert.match(migration, /Artificial Analysis records the release as September 21, 2026/);
  assert.match(migration, /## What is in the series/);
  assert.match(migration, /## The training run is the actual story/);
  assert.match(migration, /### Groupwise agentic grading/);
  assert.match(migration, /## Benchmarks/);
  assert.match(migration, /## Pricing/);
  assert.match(migration, /## Availability/);
  assert.match(migration, /## Bottom line/);
  assert.match(migration, /1,568 samples per update/);
  assert.match(migration, /Introducing MiMo-V2\.6 series/);
  assert.match(migration, /MiMo-V2\.6-Pro on Artificial Analysis/);
});

test("MiMo V2.6 Pro entry carries the architecture and pricing facts", async () => {
  const migration = await readFile(migrationPath, "utf8");

  assert.match(migration, /## Core profile/);
  assert.match(migration, /## Architecture/);
  assert.match(migration, /## Benchmarks/);
  assert.match(migration, /## The cyber numbers/);
  assert.match(migration, /## Pricing/);
  assert.match(migration, /## Where it fits/);
  assert.match(migration, /## Bottom line/);
  assert.match(migration, /1\.02 trillion total parameters and 42 billion active/);
  assert.match(migration, /70 \/ 60 \/ 10/);
  assert.match(migration, /384 \/ 8/);
  assert.match(migration, /MiMo-V2\.6-Pro model page/);
  assert.match(migration, /MiMo-V2\.6-Pro on Artificial Analysis/);
});

test("MiMo V2.6 entries keep the source caveats in the bodies", () => {
  assert.ok(series);
  assert.ok(pro);

  assert.equal(series.name, "MiMo V2.6");
  assert.equal(pro.name, "MiMo V2.6 Pro");
  assert.equal(series.category, "Models & Architectures");
  assert.equal(pro.category, "Models & Architectures");
  assert.equal(series.lastUpdated, "2026-09-30");
  assert.equal(pro.lastUpdated, "2026-09-30");

  // Vendor transcription, not independently verified numbers.
  assert.match(series.fullDef, /Xiaomi's transcription, and most of those labs do not publish/);
  assert.match(pro.fullDef, /Most of those competitors do not publish the underlying numbers/);
  // Three in-house benchmarks nobody outside Xiaomi can reproduce.
  assert.match(series.fullDef, /Three of these rows are Xiaomi's own in-house benchmarks/);
  assert.match(pro.fullDef, /Three rows cannot be reproduced by anyone outside Xiaomi at all/);
  // The post's own prose and per-step table disagree on the DeepSWE endpoints.
  assert.match(series.fullDef, /48\.8 to 65\.68 and 58\.4 to 72\.57/);
  assert.match(series.fullDef, /48\.7 and 65\.7 for the same two endpoints/);
  // License is not stated on either model page.
  assert.match(series.fullDef, /under the MIT License/);
  assert.match(pro.fullDef, /MIT licensed on Hugging Face/);
  // Independent evaluator contradicts the marketing claim on speed.
  assert.match(pro.fullDef, /The model is slow, and it is verbose/);
  assert.match(pro.fullDef, /140M tokens/);
});

test("MiMo V2.6 entries have no inline title or duplicated reference sections", () => {
  assert.ok(series);
  assert.ok(pro);
  for (const entry of [series, pro]) {
    assert.ok(entry);
    assert.ok(!/^# /.test(entry.fullDef));
    assert.ok(!entry.fullDef.includes("## References"));
    assert.ok(!entry.fullDef.includes("## Related Terms"));
    assert.ok(!entry.fullDef.includes("https://www.thequery.in/glossary/"));
  }
  assert.equal(series.references.length, 3);
  assert.equal(pro.references.length, 3);
});

test("MiMo V2.6 entries keep their tables and SEO fields", () => {
  assert.ok(series);
  assert.ok(pro);

  assert.equal(series.fullDef.match(/^\| ---/gm)?.length, 3);
  assert.equal(pro.fullDef.match(/^\| ---/gm)?.length, 4);
  assert.ok(series.fullDef.includes("| MiMo V2.6 Pro | 1.02T / 42B |"));
  assert.ok(series.fullDef.includes("| MiMo V2.6 Flash | $0.0028 | $0.14 | $0.28 |"));
  assert.ok(series.fullDef.includes("| Terminal-Bench 4.0 | 34.9 |"));
  assert.ok(pro.fullDef.includes("| Input (cache hit) | $0.0036 |"));
  assert.ok(pro.fullDef.includes("| Routed experts (total / activated) | 384 / 8 |"));
  assert.ok(pro.fullDef.includes("| ExploitGym | 17.8 | 6.0 |"));
  assert.ok(pro.fullDef.includes("| SEC Bench Pro | 66.3 | 47.5 |"));

  for (const entry of [series, pro]) {
    assert.ok(entry.seoDescription.length >= 140, `${entry.slug} seoDescription too short`);
    assert.ok(entry.seoDescription.length <= 160, `${entry.slug} seoDescription too long`);
    assert.ok(entry.seoKeywords.includes(entry.slug));
  }
});

test("the two MiMo V2.6 terms cross-link to each other", () => {
  assert.ok(series);
  assert.ok(pro);

  assert.ok(series.relatedTerms.includes("mimo-v2-6-pro"));
  assert.ok(pro.relatedTerms.includes("mimo-v2-6"));
  assert.match(series.fullDef, /\[MiMo V2\.6 Pro\]\(\/glossary\/mimo-v2-6-pro\)/);
  assert.match(pro.fullDef, /\[MiMo V2\.6 series\]\(\/glossary\/mimo-v2-6\)/);
});

test("every MiMo V2.6 related term resolves in the glossary", () => {
  const slugs = new Set(glossary.map((term) => term.slug));

  for (const entry of [series, pro]) {
    assert.ok(entry);
    for (const slug of entry.relatedTerms) {
      assert.ok(slugs.has(slug), `${entry.slug}: unresolved related term: ${slug}`);
    }
  }
});

test("every inline MiMo V2.6 glossary link resolves", () => {
  const slugs = new Set(glossary.map((term) => term.slug));

  for (const entry of [series, pro]) {
    assert.ok(entry);
    for (const [, target] of entry.fullDef.matchAll(/\]\(\/glossary\/([a-z0-9-]+)\)/g)) {
      assert.ok(slugs.has(target), `${entry.slug}: unresolved inline link: ${target}`);
    }
  }
});

test("the seed bodies match the migration bodies", async () => {
  const migration = await readFile(migrationPath, "utf8");
  const bodies = [...migration.matchAll(/\$body\$\n([\s\S]*?)\n\$body\$/g)].map(
    (match) => match[1],
  );

  assert.equal(bodies.length, 2);
  assert.equal(bodies[0], series?.fullDef);
  assert.equal(bodies[1], pro?.fullDef);
});
