import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("public content reads use narrow projections and cache tags", async () => {
  const source = await readFile(path.join(root, "lib/content.ts"), "utf8");

  assert.match(source, /SELECT id, kind, slug, parent_slug, path, title, summary, metadata/);
  assert.match(source, /published_at DESC NULLS LAST, created_at DESC, title ASC/);
  assert.match(source, /\["content-summaries-v2", kind, parentSlug\]/);
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

test("public database-backed pages use ISR instead of forced dynamic rendering", async () => {
  const files = [
    "app/page.tsx",
    "app/articles/page.tsx",
    "app/articles/[slug]/page.tsx",
    "app/guides/page.tsx",
    "app/guides/[slug]/page.tsx",
    "app/books/page.tsx",
    "app/books/[slug]/page.tsx",
    "app/books/[slug]/[chapter]/page.tsx",
    "app/glossary/page.tsx",
    "app/glossary/[term]/page.tsx",
    "app/ai-word-of-the-day/page.tsx",
    "app/sitemap.ts",
  ];

  const sources = await Promise.all(files.map((file) => readFile(path.join(root, file), "utf8")));
  assert.ok(sources.every((source) => !source.includes('dynamic = "force-dynamic"')));
  assert.ok(sources.every((source) => source.includes("revalidate")));
});
