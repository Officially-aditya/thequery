import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { createOpenGraphMetadata, createTwitterMetadata } from "../lib/site.ts";

const root = path.resolve(import.meta.dirname, "..");

test("article cover images are reused for the social preview", async () => {
  const page = await readFile(
    path.join(root, "app/articles/[slug]/page.tsx"),
    "utf8",
  );

  assert.match(page, /image: issue\.coverImageUrl/);
  assert.match(page, /twitter: createTwitterMetadata/);
  assert.equal(page.match(/image: issue\.coverImageUrl/g)?.length, 2);
  assert.match(page, /<CoverImage src=\{issue\.coverImageUrl\}/);
});

test("social metadata uses a cover image when present and defaults when absent", () => {
  const input = {
    title: "Article title",
    description: "Article description",
    url: "https://www.thequery.in/articles/example",
  };

  assert.deepEqual(createOpenGraphMetadata({ ...input, image: "/cover.webp" }).images, ["/cover.webp"]);
  assert.deepEqual(createTwitterMetadata({ ...input, image: "/cover.webp" }).images, ["/cover.webp"]);
  assert.deepEqual(createOpenGraphMetadata(input).images, ["/opengraph-image"]);
  assert.deepEqual(createTwitterMetadata(input).images, ["/twitter-image"]);
});
