import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

test("admin content lists default to summaries and load full records on demand", async () => {
  const route = await readFile(path.join(import.meta.dirname, "../app/api/admin/content/[type]/route.ts"), "utf8");
  const clients = await Promise.all([
    readFile(path.join(import.meta.dirname, "../components/admin/EditorialCollection.tsx"), "utf8"),
    readFile(path.join(import.meta.dirname, "../components/admin/GlossaryManager.tsx"), "utf8"),
    readFile(path.join(import.meta.dirname, "../components/admin/BooksManager.tsx"), "utf8"),
  ]);

  assert.match(route, /getContentSummaries/);
  assert.match(route, /searchParams\.get\("full"\) === "1"/);
  assert.ok(clients.every((source) => source.includes("summary=1")));
  assert.ok(clients.every((source) => source.includes("slug=")));
});
