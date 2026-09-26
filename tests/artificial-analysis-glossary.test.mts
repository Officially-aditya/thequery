import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("Artificial Analysis glossary expansion is registered and keeps its body block in sync", async () => {
  const [runner, migration] = await Promise.all([
    readFile(path.join(root, "scripts/migrate.mjs"), "utf8"),
    readFile(path.join(root, "db/migrations/074_expand_artificial_analysis_glossary.sql"), "utf8"),
  ]);

  assert.match(runner, /074_expand_artificial_analysis_glossary/);
  assert.match(migration, /## Artificial Analysis leaderboard/);
  assert.match(migration, /## Artificial Analysis Intelligence Index/);
  assert.match(migration, /## Artificial Analysis Coding Agent Index/);
  assert.match(migration, /## Artificial Analysis Openness Index/);
  assert.match(migration, /'content', artificial_analysis_entry\.body/);
  assert.match(migration, /item\.kind = 'glossary'/);
  assert.match(migration, /item\.slug = 'artificial-analysis'/);
});

test("Artificial Analysis glossary expansion names the current index versions", async () => {
  const migration = await readFile(
    path.join(root, "db/migrations/074_expand_artificial_analysis_glossary.sql"),
    "utf8"
  );

  assert.match(migration, /Intelligence Index is the headline number, currently v4\.3\.2/);
  assert.match(migration, /Coding Agent Index, currently v1\.5/);
  assert.match(migration, /Endpoint Accuracy Index v1\.0/);
  assert.match(migration, /AA-Briefcase v1\.1/);
  assert.match(migration, /Terminal-Bench 4\.0/);
});
