import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

async function source(file: string) {
  return readFile(path.join(root, file), "utf8");
}

test("joined spec tables keep one row group so model names stay pinned across sections", async () => {
  const renderer = await source("components/content/ContentBlocksRenderer.tsx");

  assert.match(
    renderer,
    /<tbody>\s*\{tables\.map\(\(table, tableIndex\) => \(\s*<SpecRows key=\{table\.id\}/s,
  );
  assert.doesNotMatch(renderer, /tables\.map[\s\S]*?<tbody key=\{table\.id\}>/);
  assert.match(renderer, /title=\{modelA\} className="sticky top-14 z-30/);
  assert.match(renderer, /title=\{modelB\} className="sticky top-14 z-30/);
  assert.match(renderer, /\{table\.title\}\s*<\/th>\s*<td aria-hidden="true"/s);
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
