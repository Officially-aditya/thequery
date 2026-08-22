import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("public content reads use narrow projections and cache tags", async () => {
  const source = await readFile(path.join(root, "lib/content.ts"), "utf8");

  assert.match(source, /SELECT id, kind, slug, parent_slug, path, title, summary, metadata/);
  assert.match(source, /\["content-summaries", kind, parentSlug\]/);
  assert.match(source, /\["content-index", kind\]/);
  assert.match(source, /tags: \[`content:\$\{kind\}`\]/);
});

test("public pages no longer load the full glossary for navigation", async () => {
  const files = [
    "app/page.tsx",
    "app/ai-word-of-the-day/page.tsx",
    "app/articles/[slug]/page.tsx",
    "app/guides/[slug]/page.tsx",
    "app/books/[slug]/[chapter]/page.tsx",
    "app/glossary/page.tsx",
    "app/glossary/[term]/page.tsx",
    "app/sitemap.ts",
  ];

  const sources = await Promise.all(files.map((file) => readFile(path.join(root, file), "utf8")));
  assert.ok(sources.every((source) => !source.includes("getAllTerms")));
});
