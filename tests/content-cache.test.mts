import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("public content reads use narrow projections and on-demand cache tags", async () => {
  const source = await readFile(path.join(root, "lib/content.ts"), "utf8");

  assert.match(source, /SELECT id, kind, slug, parent_slug, path, title, summary, metadata/);
  assert.match(source, /published_at DESC NULLS LAST, created_at DESC, title ASC/);
  assert.match(source, /\["content-summaries-v2", kind, parentSlug\]/);
  assert.match(source, /\["content-item-v2", kind, resolvedParentSlug, slug\]/);
  assert.match(source, /\["content-index", kind\]/);
  assert.match(source, /tags: \[`content:\$\{kind\}`\]/);
  assert.doesNotMatch(source, /PUBLIC_CACHE_SECONDS/);
  assert.doesNotMatch(source, /revalidate:\s*\d+/);
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

test("editorial pages use on-demand static regeneration instead of timed ISR", async () => {
  const staticFiles = [
    "app/articles/page.tsx",
    "app/articles/[slug]/page.tsx",
    "app/guides/page.tsx",
    "app/guides/[slug]/page.tsx",
    "app/books/page.tsx",
    "app/books/[slug]/page.tsx",
    "app/books/[slug]/[chapter]/page.tsx",
    "app/glossary/page.tsx",
    "app/glossary/[term]/page.tsx",
    "app/comparisons/page.tsx",
    "app/comparisons/[slug]/page.tsx",
    "app/research/page.tsx",
    "app/research/data/[file]/route.ts",
    "app/sitemap.ts",
  ];

  const sources = await Promise.all(staticFiles.map((file) => readFile(path.join(root, file), "utf8")));
  assert.ok(sources.every((source) => source.includes("export const revalidate = false")));
  assert.ok(sources.every((source) => !source.includes('dynamic = "force-dynamic"')));
  assert.ok(sources.every((source) => !/export const revalidate = (?:300|900|3600)/.test(source)));
});

test("only daily content retains time-based regeneration", async () => {
  const [home, word] = await Promise.all([
    readFile(path.join(root, "app/page.tsx"), "utf8"),
    readFile(path.join(root, "app/ai-word-of-the-day/page.tsx"), "utf8"),
  ]);
  assert.match(home, /export const revalidate = 86400/);
  assert.match(word, /export const revalidate = 86400/);
});

test("known detail routes are prebuilt while allowing future on-demand slugs", async () => {
  const files = [
    "app/articles/[slug]/page.tsx",
    "app/guides/[slug]/page.tsx",
    "app/books/[slug]/page.tsx",
    "app/books/[slug]/[chapter]/page.tsx",
    "app/glossary/[term]/page.tsx",
    "app/comparisons/[slug]/page.tsx",
    "app/research/data/[file]/route.ts",
  ];
  const sources = await Promise.all(files.map((file) => readFile(path.join(root, file), "utf8")));
  assert.ok(sources.every((source) => source.includes("generateStaticParams")));
});
