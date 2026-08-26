import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("sources list stays aligned with the article reading column", async () => {
  const renderer = await readFile(
    path.join(root, "components/content/ContentBlocksRenderer.tsx"),
    "utf8",
  );

  assert.match(renderer, /mx-auto mt-10 w-full max-w-\[720px\] border-t/);
});
