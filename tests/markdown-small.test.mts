import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

test("small Markdown notes use compact muted styling", async () => {
  const source = await readFile(
    path.join(import.meta.dirname, "../components/MarkdownRenderer.tsx"),
    "utf8",
  );

  assert.match(source, /small: \(\{ children \}\) =>/);
  assert.match(source, /mb-6 block text-xs leading-relaxed text-text-muted/);
});
