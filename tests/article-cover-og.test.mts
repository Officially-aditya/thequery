import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("article cover images are reused for the social preview", async () => {
  const page = await readFile(
    path.join(root, "app/articles/[slug]/page.tsx"),
    "utf8",
  );

  assert.match(page, /image: issue\.coverImageUrl/);
  assert.match(page, /<CoverImage src=\{issue\.coverImageUrl\}/);
});
