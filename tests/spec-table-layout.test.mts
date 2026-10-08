import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { prepareGlossaryMarkdown } from "../lib/glossary-linking.mjs";

const root = path.resolve(import.meta.dirname, "..");

async function source(file: string) {
  return readFile(path.join(root, file), "utf8");
}

test("joined spec tables keep one row group so model names stay pinned across sections", async () => {
  const renderer = await source("components/content/ContentBlocksRenderer.tsx");

  assert.match(
    renderer,
    /<tbody>\s*\{tables\.map\(\(table, tableIndex\) => \(\s*<SpecRows\s+key=\{table\.id\}/,
  );
  assert.doesNotMatch(renderer, /tables\.map[\s\S]*?<tbody key=\{table\.id\}>/);
  assert.match(renderer, /title=\{modelA\} className="sticky top-14 z-30/);
  assert.match(renderer, /title=\{modelB\} className="sticky top-14 z-30/);
  assert.match(renderer, /\{table\.title\}\s*<\/th>\s*<td aria-hidden="true"/);
});

test("comparison pricing keeps dollar signs and spaces instead of rendering as math", async () => {
  const renderer = await source("components/content/ContentBlocksRenderer.tsx");
  assert.match(renderer, /prepareGlossaryMarkdown\(compactTokenCounts\(text\)\)/);

  for (const pricing of [
    "$0.10 for prompts up to 100K tokens; $0.50 above 100K",
    "$0.01 for prompts up to 100K tokens; $0.05 above 100K",
    "$0.125 for prompts up to 100K tokens; $0.625 above 100K",
    "$0.50 for prompts up to 100K tokens; $2.50 above 100K",
  ]) {
    const html = renderToStaticMarkup(Markdown({
      children: prepareGlossaryMarkdown(pricing),
      remarkPlugins: [remarkGfm, remarkMath],
      rehypePlugins: [rehypeKatex],
    }));
    assert.equal(html, `<p>${pricing}</p>`);
    assert.doesNotMatch(html, /katex/);
  }
});

test("comparison markdown still renders intentional math and bold prices", () => {
  const render = (text: string) => renderToStaticMarkup(Markdown({
    children: prepareGlossaryMarkdown(text),
    remarkPlugins: [remarkGfm, remarkMath],
    rehypePlugins: [rehypeKatex],
  }));
  assert.match(render("**$0.10**"), /<strong>\$0\.10<\/strong>/);
  assert.match(render("$0.5 \\cdot x$"), /class="katex"/);
});

test("all editorial surfaces use the shared content renderer", async () => {
  const publicSurfaces = [
    "app/articles/[slug]/page.tsx",
    "app/guides/[slug]/page.tsx",
    "app/comparisons/[slug]/page.tsx",
  ];
  const adminSurfaces = [
    "components/admin/EditorialCollection.tsx",
    "components/admin/GlossaryManager.tsx",
    "components/admin/BooksManager.tsx",
  ];

  for (const file of [...publicSurfaces, ...adminSurfaces]) {
    const contents = await source(file);
    assert.match(contents, /ContentBlocksRenderer/, `${file} should use the shared renderer`);
  }
});
