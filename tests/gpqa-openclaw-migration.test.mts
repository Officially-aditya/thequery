import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

test("GPQA Diamond and OpenClaw database updates are registered and rendered from their bodies", async () => {
  const [runner, migration] = await Promise.all([
    readFile(path.join(import.meta.dirname, "../scripts/migrate.mjs"), "utf8"),
    readFile(path.join(import.meta.dirname, "../db/migrations/007_update_gpqa_diamond_and_openclaw.sql"), "utf8"),
  ]);

  assert.match(runner, /007_update_gpqa_diamond_and_openclaw/);
  assert.match(migration, /## How GPQA Diamond is scored/);
  assert.match(migration, /## OpenClaw 2\.0/);
  assert.match(migration, /v2026\.8\.1/);
  assert.match(migration, /'content', glossary_updates\.body/);
  assert.match(migration, /item\.slug = glossary_updates\.slug/);
});
