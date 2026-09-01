import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("benchmark images use the reusable accessible article image rail", async () => {
  const [page, rail] = await Promise.all([
    readFile(path.join(root, "app/articles/[slug]/page.tsx"), "utf8"),
    readFile(path.join(root, "components/article/ArticleImageRail.tsx"), "utf8"),
  ]);

  assert.match(page, /glm-5-3-flash-ox-alpha-free-model-benchmarks/);
  assert.match(page, /glm-53-flash-benchmarks\.jpg/);
  assert.match(page, /claude-fable-51-low-effort-benchmarks/);
  assert.match(page, /claude-fable-51-benchmarks\.webp/);
  assert.match(page, /placement: "right-rail"/);
  assert.match(rail, /<Image/);
  assert.match(rail, /<ImageLightbox/);
  assert.match(rail, /<figcaption/);
  assert.match(rail, /Click the image for the full-size chart/);
  assert.doesNotMatch(rail, /target="_blank"/);
});
