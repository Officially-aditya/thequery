import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

async function source(file: string) {
  return readFile(path.join(root, file), "utf8");
}

test("article pagination no longer makes the server route request-specific", async () => {
  const [page, pagination] = await Promise.all([
    source("app/articles/page.tsx"),
    source("components/articles/ArticlePagination.tsx"),
  ]);
  assert.doesNotMatch(page, /searchParams/);
  assert.match(page, /export const revalidate = false/);
  assert.match(pagination, /useSearchParams/);
  assert.match(pagination, /\/articles\?page=\$\{page\}/);
});

test("model option cache no longer wakes up every five minutes", async () => {
  const models = await source("lib/models.ts");
  assert.doesNotMatch(models, /MODEL_OPTION_CACHE_SECONDS/);
  assert.doesNotMatch(models, /revalidate:\s*300/);
});

test("glossary matching compiles once and uses constant-time term lookup", async () => {
  const markdown = await source("components/MarkdownRenderer.tsx");
  assert.match(markdown, /function createGlossaryMatcher/);
  assert.match(markdown, /const glossaryMatcher = createGlossaryMatcher\(glossaryTerms\)/);
  assert.match(markdown, /byName\.get\(termKey\)/);
  assert.doesNotMatch(markdown, /sorted\.find/);
});

test("admin mutations invalidate dependent static route families", async () => {
  const admin = await source("app/api/admin/content/[type]/route.ts");
  assert.match(admin, /revalidateTag\(`content:\$\{kind\}`/);
  assert.match(admin, /revalidatePath\("\/articles\/\[slug\]", "page"\)/);
  assert.match(admin, /revalidatePath\("\/guides\/\[slug\]", "page"\)/);
  assert.match(admin, /revalidatePath\("\/books\/\[slug\]\/\[chapter\]", "page"\)/);
  assert.match(admin, /revalidatePath\("\/comparisons\/\[slug\]", "page"\)/);
  assert.match(admin, /revalidatePath\("\/research"\)/);
});

test("research exports are generated ahead of traffic", async () => {
  const route = await source("app/research/data/[file]/route.ts");
  assert.match(route, /export const revalidate = false/);
  assert.match(route, /generateStaticParams/);
  assert.match(route, /models\.json/);
  assert.match(route, /benchmarks\.csv/);
});
