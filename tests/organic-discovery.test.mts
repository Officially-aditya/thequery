import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import test from "node:test";
import { runInNewContext } from "node:vm";
import { createElement } from "react";
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
    Response: globalThis.Response,
    Request: globalThis.Request,
    require: (id: string) => Object.hasOwn(mocks, id) ? mocks[id] : require(id),
  }, { filename });
  return loaded.exports;
}

test("chapter HTML includes every section, matching sidebar anchors, and adjacent chapter links", () => {
  const reader = loadModule("components/SectionReader.tsx", {
    "./MarkdownRenderer": ({ content }: { content: string }) => createElement("p", null, content),
  });
  const chapter = loadModule("components/ChapterView.tsx", {
    "./ChapterSidebar": loadModule("components/ChapterSidebar.tsx"),
    "./SectionReader": reader,
  }) as typeof import("../components/ChapterView.tsx");
  const sections = [
    { title: "", id: "", content: "Opening section body." },
    { title: "Models", id: "models", content: "Second section body." },
    { title: "Models", id: "models", content: "Repeated-heading section body." },
    { title: "Evaluation", id: "evaluation", content: "Final section body." },
  ];
  const html = renderToStaticMarkup(createElement(chapter.default, {
    bookSlug: "ai-book",
    bookTitle: "AI Book",
    chapters: [{ slug: "current", title: "Current chapter" }],
    currentChapter: "current",
    currentIdx: 1,
    sections,
    prevChapter: { slug: "previous", title: "Previous chapter" },
    nextChapter: { slug: "next", title: "Next chapter" },
  }));
  for (const section of sections) assert.ok(html.includes(section.content));
  const sidebar = html.match(/<aside\b[^>]*>([\s\S]*?)<\/aside>/)?.[1] ?? "";
  const anchors = [...sidebar.matchAll(/href="#([^"]+)"/g)].map((match) => match[1]);
  const renderedIds = [...html.matchAll(/<section\b[^>]*id="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(anchors.length, sections.length);
  assert.equal(new Set(anchors).size, sections.length);
  assert.deepEqual(anchors, renderedIds);
  const adjacent = html.match(/<nav aria-label="Adjacent chapters"[^>]*>([\s\S]*?)<\/nav>/)?.[1] ?? "";
  assert.match(adjacent, /href="\/books\/ai-book\/previous"/);
  assert.match(adjacent, /href="\/books\/ai-book\/next"/);
});

function editorialContent() {
  const publishedAt = "2026-04-01";
  const updates = { article: "2026-10-11", guide: "2026-10-12", comparison: "2026-10-13" };
  const records = Object.fromEntries(Object.entries(updates).map(([kind, updatedAt]) => [kind, {
    title: `${kind} title`, slug: `${kind}-slug`, summary: "Summary", body: "Content",
    blocks: [], sources: [], metadata: {}, publishedAt, updatedAt,
  }]));
  const content = loadModule("lib/content.ts", {
    "server-only": {}, "./db": {}, "./content-utils": {},
  }) as typeof import("../lib/content.ts");
  const mocks = {
    "server-only": {},
    "./content": {
      getContentItem: async (kind: string) => records[kind],
      getContentSummaries: async (kind: string) => [records[kind]],
      contentDisplayDate: content.contentDisplayDate,
    },
  };
  return {
    publishedAt, updates,
    articles: loadModule("lib/articles.ts", mocks) as typeof import("../lib/articles.ts"),
    guides: loadModule("lib/guides.ts", mocks) as typeof import("../lib/guides.ts"),
    comparisons: loadModule("lib/comparisons.ts", mocks) as typeof import("../lib/comparisons.ts"),
  };
}

test("editorial details and listings retain publication dates and expose actual update dates", async () => {
  const { articles, guides, comparisons, publishedAt, updates } = editorialContent();
  const values = [
    [await articles.getIssueBySlug("article-slug"), ...(await articles.getAllIssues())],
    [await guides.getGuideBySlug("guide-slug"), ...(await guides.getAllGuides())],
    [await comparisons.getComparisonBySlug("comparison-slug"), ...(await comparisons.getAllComparisons())],
  ];
  for (const [index, entries] of values.entries()) {
    for (const entry of entries) {
      assert.equal(entry?.date, publishedAt);
      assert.equal(entry?.lastModified, Object.values(updates)[index]);
    }
  }
});

test("sitemap collection and detail dates use updates without inventing static page freshness", async () => {
  const { articles, guides, comparisons, updates } = editorialContent();
  const sitemap = loadModule("app/sitemap.ts", {
    "@/lib/books": { getAllBooks: async () => [] },
    "@/lib/glossary": { getAllTermSummaries: async () => [] },
    "@/lib/articles": articles, "@/lib/guides": guides, "@/lib/comparisons": comparisons,
  }) as typeof import("../app/sitemap.ts");
  const entries = new Map((await sitemap.default()).map((entry) => [entry.url, entry]));
  const baseUrl = "https://www.thequery.in";
  for (const [kind, updatedAt] of Object.entries(updates)) {
    for (const pathname of [`/${kind}s`, `/${kind}s/${kind}-slug`]) {
      const entry = entries.get(`${baseUrl}${pathname}`);
      assert.ok(entry);
      assert.equal(new Date(entry.lastModified ?? 0).toISOString().slice(0, 10), updatedAt);
    }
  }
  for (const pathname of ["/privacy", "/about", "/research"]) {
    const entry = entries.get(`${baseUrl}${pathname}`);
    assert.ok(entry);
    assert.equal(entry.lastModified, undefined);
  }
});

test("book and chapter dates follow database edits in metadata and both sitemap formats", async () => {
  const book = {
    slug: "course", title: "Course", summary: "Course summary",
    metadata: { lastModified: "2026-02-01" }, updatedAt: "2026-10-10",
  };
  const chapters = [
    { slug: "first", title: "First", body: "Content", blocks: [], sources: [], metadata: { lastModified: "2026-02-02" }, updatedAt: "2026-10-12" },
    { slug: "second", title: "Second", body: "Content", blocks: [], sources: [], metadata: {}, updatedAt: "2026-10-11" },
  ];
  const books = loadModule("lib/books.ts", {
    "server-only": {},
    "./content": {
      getContentItem: async (kind: string, slug: string) => kind === "book" ? book : chapters.find((chapter) => chapter.slug === slug),
      getContentSummaries: async (kind: string) => kind === "book" ? [book] : chapters,
    },
  }) as typeof import("../lib/books.ts");
  assert.equal((await books.getBookMeta("course"))?.lastModified, "2026-10-12");
  assert.equal((await books.getChapterContent("course", "first"))?.meta.lastModified, "2026-10-12");
  assert.equal((await books.getChapterContent("course", "second"))?.meta.lastModified, "2026-10-11");
  const sitemap = loadModule("app/sitemap.ts", {
    "@/lib/books": books,
    "@/lib/glossary": { getAllTermSummaries: async () => [] },
    "@/lib/articles": { getAllIssues: async () => [] },
    "@/lib/guides": { getAllGuides: async () => [] },
    "@/lib/comparisons": { getAllComparisons: async () => [] },
  }) as typeof import("../app/sitemap.ts");
  const entries = await sitemap.default();
  const base = "https://www.thequery.in";
  for (const [pathname, date] of [
    ["/books", "2026-10-12"], ["/books/course", "2026-10-12"],
    ["/books/course/first", "2026-10-12"], ["/books/course/second", "2026-10-11"],
  ]) {
    assert.equal(new Date(entries.find((entry) => entry.url === base + pathname)?.lastModified ?? 0).toISOString().slice(0, 10), date);
  }
  const route = loadModule("app/sitemap/[id]/route.ts", {
    "../../sitemap.ts": sitemap.default,
  }) as typeof import("../app/sitemap/[id]/route.ts");
  const response = await route.GET(new Request(base + "/sitemap/books.xml"), { params: Promise.resolve({ id: "books.xml" }) });
  const xml = await response.text();
  assert.match(xml, /<loc>https:\/\/www\.thequery\.in\/books\/course\/first<\/loc>\s*<lastmod>2026-10-12T00:00:00\.000Z<\/lastmod>/);

  book.updatedAt = "2026-10-13";
  assert.equal((await books.getBookMeta("course"))?.lastModified, "2026-10-13");
  assert.equal((await books.getChapterContent("course", "second"))?.meta.lastModified, "2026-10-11");
  chapters.length = 0;
  assert.equal((await books.getBookMeta("course"))?.lastModified, "2026-10-13");
});
