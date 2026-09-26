import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

const migrationPath = path.join(root, "db/migrations/075_add_gemini_3_8_flash.sql");

test("Gemini 3.8 Flash migration is registered and builds the glossary row", async () => {
  const [runner, migration] = await Promise.all([
    readFile(path.join(root, "scripts/migrate.mjs"), "utf8"),
    readFile(migrationPath, "utf8"),
  ]);

  assert.match(runner, /075_add_gemini_3_8_flash/);
  assert.match(migration, /'glossary:gemini-3-8-flash'/);
  assert.match(migration, /'gemini-3-8-flash'/);
  assert.match(migration, /'content', gemini_38_flash_entry\.body/);
  assert.match(migration, /ON CONFLICT \(kind, slug, parent_slug\) DO UPDATE SET/);
  assert.match(migration, /DATE '2026-09-02'/);
});

test("Gemini 3.8 Flash entry carries the release facts and cross-links", async () => {
  const migration = await readFile(migrationPath, "utf8");

  assert.match(migration, /Gemini 3\.8 Flash is Google's September 2, 2026 generally available/);
  assert.match(migration, /## Key specifications/);
  assert.match(migration, /## Reported benchmarks/);
  assert.match(migration, /gemini-3\.8-flash/);
  assert.match(migration, /1,048,576 tokens/);
  assert.match(migration, /65,536 tokens/);
  assert.match(migration, /\/glossary\/gemini-3-7-flash/);
  assert.match(migration, /item\.slug = 'gemini-3-7-flash'|AND slug = 'gemini-3-7-flash'/);
});

test("Gemini 3.8 Flash entry states the January 2027 price change", async () => {
  const migration = await readFile(migrationPath, "utf8");

  assert.match(migration, /\$0\.75 per million input tokens and \$3\.75 per million output tokens/);
  assert.match(migration, /December 31, 2026/);
  assert.match(migration, /January 1, 2027/);
  assert.match(migration, /\$1\.50 per million input tokens and \$7\.50 per million output tokens/);
});
