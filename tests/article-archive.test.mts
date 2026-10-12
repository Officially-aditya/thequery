import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import test from "node:test";
import { runInNewContext } from "node:vm";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

const root = path.resolve(import.meta.dirname, "..");
const require = createRequire(import.meta.url);

function loadModule(file: string, mocks: Record<string, unknown> = {}) {
  const filename = path.join(root, file);
  const { outputText } = ts.transpileModule(readFileSync(filename, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
  });
  const loaded = { exports: {} };
  runInNewContext(outputText, {
    module: loaded,
    exports: loaded.exports,
    require: (id: string) => Object.hasOwn(mocks, id) ? mocks[id] : require(id),
  }, { filename });
  return loaded.exports;
}

function archive(articleCount = 23) {
  const issues = Array.from({ length: articleCount }, (_, index) => ({
    title: `Article ${index + 1}`,
    slug: `article-${index + 1}`,
    date: "2026-10-13",
    summary: `Summary ${index + 1}`,
  }));
  const pagination = loadModule("components/articles/ArticlePagination.tsx");
  return loadModule("app/articles/page.tsx", {
    "@/lib/articles": { getAllIssues: async () => issues },
    "@/components/articles/ArticlePagination": pagination,
    "@/lib/site": {
      SITE_URL: "https://www.thequery.in",
      createOpenGraphMetadata: (metadata: unknown) => metadata,
    },
    "next/navigation": { redirect: (url: string) => { throw Object.assign(new Error("Redirect"), { url }); } },
  }) as typeof import("../app/articles/page.tsx");
}

test("article archives render ten linked articles per page and crawlable navigation without hydration", async () => {
  const route = archive();
  for (const [page, firstArticle, lastArticle] of [[1, 1, 10], [2, 11, 20], [3, 21, 23]]) {
    const html = renderToStaticMarkup(await route.default({
      searchParams: Promise.resolve(page === 1 ? {} : { page: String(page) }),
    }));
    const hrefs = [...html.matchAll(/href="\/articles\/article-(\d+)"/g)].map((match) => Number(match[1]));
    assert.deepEqual(hrefs, Array.from({ length: lastArticle - firstArticle + 1 }, (_, index) => firstArticle + index));
    assert.match(html, /href="\/articles"/);
    assert.match(html, /href="\/articles\?page=2"/);
    assert.match(html, /href="\/articles\?page=3"/);
    assert.doesNotMatch(html, /Loading articles/);
  }
});

test("every valid archive page has its own canonical and page title", async () => {
  const route = archive();
  for (const page of [1, 2, 3]) {
    const metadata = await route.generateMetadata({
      searchParams: Promise.resolve(page === 1 ? {} : { page: String(page) }),
    });
    assert.equal(metadata.alternates?.canonical, `https://www.thequery.in/articles${page === 1 ? "" : `?page=${page}`}`);
    assert.equal(metadata.title, `AI News & Analysis${page === 1 ? "" : ` - Page ${page}`}`);
  }
});

test("duplicate first pages and invalid archive pages redirect to the first page", async () => {
  const route = archive();
  for (const page of ["", "0", "1", "01", "-1", "2.5", "4", "9007199254740992", "invalid", ["2", "3"]]) {
    const props = { searchParams: Promise.resolve({ page }) };
    for (const render of [route.default, route.generateMetadata]) {
      await assert.rejects(render(props), (error: unknown) => (
        typeof error === "object" && error !== null && "url" in error && error.url === "/articles"
      ));
    }
  }
});

test("an empty archive renders a stable first page", async () => {
  const route = archive(0);
  const html = renderToStaticMarkup(await route.default({ searchParams: Promise.resolve({}) }));
  assert.match(html, /No articles yet/);
  assert.doesNotMatch(html, /aria-label="Article pages"/);
});
