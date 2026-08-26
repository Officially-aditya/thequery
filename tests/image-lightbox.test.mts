import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("all public image renderers use the same-page lightbox", async () => {
  const [lightbox, cover, markdown, rail] = await Promise.all([
    readFile(path.join(root, "components/content/ImageLightbox.tsx"), "utf8"),
    readFile(path.join(root, "components/content/CoverImage.tsx"), "utf8"),
    readFile(path.join(root, "components/MarkdownRenderer.tsx"), "utf8"),
    readFile(path.join(root, "components/article/ArticleImageRail.tsx"), "utf8"),
  ]);

  assert.match(lightbox, /role="dialog"/);
  assert.match(lightbox, /event\.target === event\.currentTarget/);
  assert.match(lightbox, /event\.key === "Escape"/);
  assert.doesNotMatch(lightbox, /cursor-zoom-in/);
  assert.match(cover, /<ImageLightbox/);
  assert.match(markdown, /<ImageLightbox/);
  assert.match(rail, /<ImageLightbox/);
  assert.doesNotMatch(rail, /target="_blank"/);
});
