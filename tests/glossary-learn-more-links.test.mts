import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";
import Markdown from "react-markdown";
import { buildGlossaryLinks } from "../lib/glossary-link-map.mjs";
import { getGlossaryMentions, prepareGlossaryMarkdown, remarkPlugins, rehypePlugins, rehypeGlossaryLinks } from "../lib/glossary-linking.mjs";

const root = path.resolve(import.meta.dirname, "..");
const terms = [
  { name: "Chunking", slug: "chunking" },
  { name: "Knowledge Graph", slug: "knowledge-graph" },
  { name: "Graph", slug: "graph" },
  { name: "Transformer", slug: "transformer" },
];
const books = [{ slug: "course", title: "Course" }];
const chapter = (slug: string, title: string, body: string) => ({ slug, title, body, parentSlug: "course" });
const guide = (slug: string, content: string) => ({
  slug, title: slug, blocks: [{ type: "markdown", content }],
});
const build = (chapters: ReturnType<typeof chapter>[] = [], guides: ReturnType<typeof guide>[] = [], glossary = terms) =>
  buildGlossaryLinks({ terms: glossary, books, chapters, guides });

function render(content: string) {
  return renderToStaticMarkup(Markdown({
    children: prepareGlossaryMarkdown(content),
    remarkPlugins,
    rehypePlugins: [...rehypePlugins, [rehypeGlossaryLinks, { terms }]],
  }));
}

test("only text that the renderer auto-links contributes outbound links", () => {
  const snippets = [
    "# Chunking", "**Chunking**", "*Chunking*", "\`Chunking\`",
    "[Chunking](/somewhere)", "![Chunking](/image.png)",
    "~~~\nChunking\n~~~", "    Chunking",
    "| Term |\n| --- |\n| Chunking |",
    "<p><strong>Chunking</strong></p>", "$\\mathrm{Chunking}$",
  ];
  for (const content of snippets) {
    assert.deepEqual([...getGlossaryMentions(content, terms)], [], content);
    assert.deepEqual(build([chapter("reference", "Reference", content)]).terms, {}, content);
    assert.doesNotMatch(render(content), /href="\/glossary\//, content);
  }
});

test("longer term matches suppress overlapping names and use word boundaries", () => {
  const content = "Knowledge Graph, KNOWLEDGE GRAPH, and graph. Transformerish.";
  assert.deepEqual([...getGlossaryMentions(content, terms)], [["knowledge-graph", 2], ["graph", 1]]);
  const html = render(content);
  assert.equal((html.match(/href="\/glossary\/knowledge-graph"/g) ?? []).length, 1);
  assert.equal((html.match(/href="\/glossary\/graph"/g) ?? []).length, 1);
  assert.doesNotMatch(html, /href="\/glossary\/transformer"/);
});

test("paragraphs and list items link the first occurrence per term", () => {
  const content = "Chunking, chunking.\n\n- Chunking\n- Graph\n\n> Knowledge Graph";
  const mentions = getGlossaryMentions(content, terms);
  assert.equal(mentions.get("chunking"), 3);
  assert.equal(mentions.get("graph"), 1);
  const html = render(content);
  for (const slug of mentions.keys()) {
    assert.equal(html.split('href="/glossary/' + slug + '"').length - 1, 1);
  }
});

test("every outbound destination has a reciprocal rendered glossary auto-link", () => {
  const chapters = [
    chapter("chunking", "Chunking", "Chunking and graph."),
    chapter("kg", "Knowledge Graph", "Knowledge Graph."),
  ];
  const guides = [guide("database-only-guide", "Transformer and Chunking.")];
  const links = build(chapters, guides);
  const rendered = new Map([
    ...chapters.map((entry) => ["/books/course/" + entry.slug, render(entry.body)] as const),
    ...guides.map((entry) => ["/guides/" + entry.slug, render(entry.blocks[0].content)] as const),
  ]);
  for (const [slug, entries] of Object.entries(links.terms)) {
    for (const entry of entries) {
      assert.ok(rendered.get(entry.href)?.includes('href="/glossary/' + slug + '"'), entry.href + " must link back to " + slug);
    }
  }
  assert.equal(links.sources.guides, 1);
  assert.ok(links.terms.transformer.some((entry) => entry.href === "/guides/database-only-guide"));
});

test("teaching chapters outrank navigation and mention counts break ties", () => {
  const links = build([
    chapter("index", "Key Terminology", "Chunking. ".repeat(20)),
    chapter("practice", "Practice", "Chunking. ".repeat(3)),
    chapter("chunking", "Chunking", "Chunking."),
    chapter("basics", "Basics", "Chunking. ".repeat(2)),
    chapter("intro", "Introduction", "Chunking. ".repeat(30)),
  ]);
  assert.deepEqual(links.terms.chunking.map((entry) => entry.href), [
    "/books/course/chunking", "/books/course/practice", "/books/course/basics", "/books/course/intro",
  ]);
  assert.ok(links.terms.chunking.every((entry) => entry.context === "Course" && entry.mentions > 0));
});

test("content edits, removed routes, and glossary renames change the derived map", () => {
  const entry = guide("new-guide", "Transformer.");
  assert.equal(build([], [entry]).terms.transformer[0].href, "/guides/new-guide");
  assert.deepEqual(build([], [guide("new-guide", "**Transformer**")]).terms, {});
  assert.deepEqual(build([], []).terms, {});
  assert.deepEqual(build([], [entry], [{ name: "Renamed Term", slug: "transformer" }]).terms, {});
  assert.deepEqual(buildGlossaryLinks({ terms, books: [], chapters: [chapter("orphan", "Orphan", "Chunking.")], guides: [] }).terms, {});
});

test("guide links use rendered markdown blocks and ignore body-only mentions", () => {
  const entry = { ...guide("blocks", "Graph."), body: "Transformer." };
  const links = build([], [entry]);
  assert.equal(links.terms.graph[0].context, null);
  assert.equal(links.terms.transformer, undefined);
  assert.equal(links.coverage.totalLinks, 1);
  assert.equal(links.coverage.termsWithLinks, 1);
});

test("admin changes invalidate derived map dependencies and all glossary pages", async () => {
  const [helper, route] = await Promise.all([
    readFile(path.join(root, "lib/glossary-links.ts"), "utf8"),
    readFile(path.join(root, "app/api/admin/content/[type]/route.ts"), "utf8"),
  ]);
  for (const kind of ["glossary", "book", "chapter", "guide"]) {
    assert.ok(helper.includes('"content:' + kind + '"'));
  }
  assert.doesNotMatch(helper, /data\/glossary-links\.json/);
  assert.match(route, /revalidatePath\("\/glossary\/\[term\]", "page"\)/);
});

test("the term page renders internal learn-more links only when links exist", async () => {
  const page = await readFile(path.join(root, "app/glossary/[term]/page.tsx"), "utf8");
  assert.match(page, /term\.learnMore\.length > 0/);
  assert.match(page, /<Link\s+href=\{link\.href\}/);
});
