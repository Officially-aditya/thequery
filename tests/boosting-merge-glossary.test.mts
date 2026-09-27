import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

const migrationPath = path.join(root, "db/migrations/076_merge_boosting_variants.sql");

async function readMigration() {
  return readFile(migrationPath, "utf8");
}

test("boosting merge migration is registered in the runner", async () => {
  const runner = await readFile(path.join(root, "scripts/migrate.mjs"), "utf8");

  assert.match(runner, /076_merge_boosting_variants/);
});

test("boosting merge migration expands the boosting row and deletes the three variants", async () => {
  const migration = await readMigration();

  assert.match(migration, /boosting_entry\.body/);
  assert.match(migration, /item\.slug = 'boosting'/);
  assert.match(migration, /DELETE FROM content_items/);
  assert.match(migration, /slug IN \('adaboost', 'xgboost', 'lightgbm'\)/);
  assert.match(migration, /kind = 'glossary'/);
  assert.match(migration, /jsonb_build_object\('id', 'markdown-1', 'type', 'markdown', 'content', boosting_entry\.body\)/);
  assert.match(migration, /updated_at = NOW\(\)/);
});

test("boosting body keeps body and blocks in sync with the same CTE entry", async () => {
  const migration = await readMigration();

  assert.equal(migration.match(/boosting_entry\.body/g)?.length, 2);
  assert.equal(migration.match(/\$body\$/g)?.length, 2);
  assert.doesNotMatch(migration, /\$body\$[\s\S]*?;[^\n]*\n[\s\S]*?\$body\$/);
});

test("merged boosting body covers each folded-in algorithm and the family framing", async () => {
  const migration = await readMigration();

  for (const heading of [
    "## How a boosting round works",
    "## AdaBoost: the original",
    "## Gradient boosting: the general form",
    "## XGBoost: regularization plus engineering",
    "## LightGBM: the same idea, a different growth policy",
    "## CatBoost and the rest of the family",
    "## The four side by side",
    "## Boosting versus bagging",
    "## What actually moves the score",
    "## Failure modes worth naming",
    "## Reading a boosted model",
    "## Where boosting sits in 2026",
    "## How to choose in one minute",
  ]) {
    assert.ok(migration.includes(heading), `missing heading: ${heading}`);
  }

  assert.match(migration, /Freund and Schapire introduced AdaBoost in 1997/);
  assert.match(migration, /Friedman's 2001 greedy function approximation/);
  assert.match(migration, /XGBoost \(2016, Tianqi Chen and Carlos Guestrin\)/);
  assert.match(migration, /LightGBM \(2017, Microsoft Research\)/);
  assert.match(migration, /CatBoost \(2018\)/);
  assert.match(migration, /up to 20 times faster than conventional GBDT/);
  assert.match(migration, /leaf-wise growth/);
  assert.match(migration, /Gradient-based One-Side Sampling/);
  assert.match(migration, /Exclusive Feature Bundling/);
  assert.match(migration, /early stopping/);
  assert.match(migration, /SHAP values are the honest version/);
});

test("merged boosting metadata drops the deleted slugs and keeps valid related terms", async () => {
  const migration = await readMigration();
  const glossary = JSON.parse(await readFile(path.join(root, "data/glossary.json"), "utf8"));
  const slugs = new Set(glossary.map((term: { slug: string }) => term.slug));

  const relatedBlock = migration.slice(migration.indexOf("'{relatedTerms}',"));
  const relatedTerms = [
    ...relatedBlock.slice(0, relatedBlock.indexOf("),\n    true\n  )")).matchAll(/'([a-z0-9-]+)'/g),
  ].map((match) => match[1]);

  assert.ok(relatedTerms.length > 0);
  for (const slug of relatedTerms) {
    assert.ok(slugs.has(slug), `related term does not exist: ${slug}`);
  }
  for (const removed of ["adaboost", "xgboost", "lightgbm"]) {
    assert.ok(!relatedTerms.includes(removed), `removed slug still in related terms: ${removed}`);
  }
});

test("seed JSON matches the merged migration and no longer ships the removed pages", async () => {
  const migration = await readMigration();
  const glossary = JSON.parse(await readFile(path.join(root, "data/glossary.json"), "utf8"));
  const wotd = JSON.parse(await readFile(path.join(root, "data/ai-word-of-the-day.json"), "utf8"));
  const body = migration.match(/\$body\$([\s\S]*?)\$body\$/)?.[1];
  const boosting = glossary.find((term: { slug: string }) => term.slug === "boosting");

  assert.ok(body);
  assert.ok(boosting);
  assert.equal(boosting.fullDef, body.trim());
  assert.equal(boosting.fullDef, boosting.fullDef.trim());
  assert.match(boosting.shortDef, /AdaBoost, XGBoost, LightGBM, and CatBoost/);
  assert.ok(Array.isArray(boosting.references) && boosting.references.length > 0);

  const slugs = new Set(glossary.map((term: { slug: string }) => term.slug));
  for (const removed of ["adaboost", "xgboost", "lightgbm"]) {
    assert.ok(!slugs.has(removed), `seed still ships removed term: ${removed}`);
    assert.ok(
      wotd.every((entry: { slug: string }) => entry.slug !== removed),
      `word of the day still points at a removed term: ${removed}`,
    );
  }

  assert.equal(
    wotd.filter((entry: { slug: string }) => entry.slug === "boosting").length,
    4,
    "the three remapped word-of-the-day entries should sit under the boosting slug",
  );
});

test("the three folded-in slugs redirect to the boosting page", async () => {
  const config = await readFile(path.join(root, "next.config.ts"), "utf8");

  for (const removed of ["adaboost", "xgboost", "lightgbm"]) {
    assert.match(
      config,
      new RegExp(`\\{ source: "/glossary/${removed}", destination: "/glossary/boosting", permanent: true \\}`),
      `missing redirect for /glossary/${removed}`,
    );
  }
});
