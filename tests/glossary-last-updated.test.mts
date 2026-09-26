import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("glossary last updated date prefers the update date over the publish date", async () => {
  const lib = await readFile(path.join(root, "lib/glossary.ts"), "utf8");

  assert.doesNotMatch(lib, /contentDisplayDate/);
  assert.match(
    lib,
    /function lastUpdatedDate\(publishedAt: string \| null, updatedAt: string\): string \{\s*return updatedAt \|\| publishedAt/,
  );
  assert.equal(lib.match(/lastUpdated: lastUpdatedDate\(/g)?.length, 2);
});

test("glossary page labels the field as last updated", async () => {
  const page = await readFile(path.join(root, "app/glossary/[term]/page.tsx"), "utf8");

  assert.match(page, /Last updated: \{new Date\(term\.lastUpdated\)/);
});

test("Artificial Analysis expansion stamps the update date on the row", async () => {
  const migration = await readFile(
    path.join(root, "db/migrations/074_expand_artificial_analysis_glossary.sql"),
    "utf8"
  );

  assert.match(migration, /updated_at = NOW\(\)/);
  assert.match(migration, /item\.slug = 'artificial-analysis'/);
});
