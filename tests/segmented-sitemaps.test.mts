import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import test from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const root = path.resolve(import.meta.dirname, "..");
const require = createRequire(import.meta.url);

function loadModule(file: string, mocks: Record<string, unknown> = {}) {
  const filename = path.join(root, file);
  const { outputText } = ts.transpileModule(readFileSync(filename, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, esModuleInterop: true },
  });
  const loaded = { exports: {} };
  runInNewContext(outputText, {
    module: loaded,
    exports: loaded.exports,
    Response: globalThis.Response,
    Request: globalThis.Request,
    URL: globalThis.URL,
    Date: globalThis.Date,
    require: (id: string) => Object.hasOwn(mocks, id) ? mocks[id] : require(id),
  }, { filename });
  return loaded.exports;
}

test("segmented sitemaps serve valid XML for each section", async () => {
  const mockEntries = [
    { url: "https://www.thequery.in", lastModified: new Date("2026-10-13") },
    { url: "https://www.thequery.in/books/rag-book", lastModified: new Date("2026-10-12") },
    { url: "https://www.thequery.in/guides/guide-1", lastModified: new Date("2026-10-11") },
    { url: "https://www.thequery.in/articles/article-1", lastModified: new Date("2026-10-10") },
    { url: "https://www.thequery.in/comparisons/comp-1", lastModified: new Date("2026-10-09") },
    { url: "https://www.thequery.in/glossary/term-1", lastModified: new Date("2026-10-08") },
  ];

  const route = loadModule("app/sitemap/[id]/route.ts", {
    "../../sitemap.ts": async () => mockEntries,
  }) as {
    GET: (req: Request, props: { params: Promise<{ id: string }> }) => Promise<Response>;
    generateStaticParams: () => { id: string }[];
  };

  const sections = ["books", "guides", "articles", "comparisons", "glossary", "pages"];

  for (const section of sections) {
    const res = await route.GET(new Request(`https://www.thequery.in/sitemap/${section}.xml`), {
      params: Promise.resolve({ id: `${section}.xml` }),
    });

    assert.equal(res.status, 200, `Expected 200 for section ${section}`);
    assert.equal(res.headers.get("Content-Type"), "application/xml; charset=utf-8");

    const xml = await res.text();
    assert.match(xml, /<urlset xmlns="http:\/\/www\.sitemaps\.org\/schemas\/sitemap\/0\.9">/);
    assert.match(xml, new RegExp(`https://www.thequery.in`));
  }
});

test("segmented sitemaps return 404 for unknown section", async () => {
  const route = loadModule("app/sitemap/[id]/route.ts", {
    "../../sitemap.ts": async () => [],
  }) as {
    GET: (req: Request, props: { params: Promise<{ id: string }> }) => Promise<Response>;
  };

  const res = await route.GET(new Request("https://www.thequery.in/sitemap/unknown.xml"), {
    params: Promise.resolve({ id: "unknown.xml" }),
  });
  assert.equal(res.status, 404);
});

test("segmented sitemaps define static params for all segments", () => {
  const route = loadModule("app/sitemap/[id]/route.ts", {
    "../../sitemap.ts": async () => [],
  }) as {
    generateStaticParams: () => { id: string }[];
  };

  const params = route.generateStaticParams();
  const ids = Array.from(params, (p) => String(p.id));
  assert.deepEqual(ids, [
    "pages.xml",
    "books.xml",
    "guides.xml",
    "comparisons.xml",
    "articles.xml",
    "glossary.xml",
  ]);
});
