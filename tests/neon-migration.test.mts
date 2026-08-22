import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

test("Neon project copy requires explicit confirmation and separate URLs", async () => {
  const source = await readFile(path.join(import.meta.dirname, "../scripts/copy-neon-project.mjs"), "utf8");

  assert.match(source, /--confirm/);
  assert.match(source, /DATABASE_URL/);
  assert.match(source, /NEW_DATABASE_URL/);
  assert.match(source, /Source and target must be different Neon projects/);
  assert.match(source, /schema_migrations/);
  assert.match(source, /Admin sessions were intentionally not copied/);
});
