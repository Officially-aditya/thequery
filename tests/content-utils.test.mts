import assert from "node:assert/strict";
import test from "node:test";
import { extractSources, normalizeBlocks, slugify } from "../lib/content-utils.ts";

test("extractSources moves a markdown source section into structured source records", () => {
  const result = extractSources(`Opening paragraph.

*Sources:*

- [Primary reporting](https://example.com/reporting) - Example
- [Official documentation](https://example.com/docs)

**Previously on TheQuery:**

- [Related article](/articles/related)`);

  assert.deepEqual(result.sources, [
    { title: "Primary reporting", url: "https://example.com/reporting" },
    { title: "Official documentation", url: "https://example.com/docs" },
  ]);
  assert.doesNotMatch(result.content, /Sources/);
  assert.match(result.content, /Previously on TheQuery/);
});

test("normalizeBlocks keeps supported table and chart data while dropping invalid blocks", () => {
  const blocks = normalizeBlocks([
    { id: "table", type: "comparison_table", columns: ["Model", "Score"], rows: [["A", "62"]] },
    { id: "chart", type: "chart", title: "Scores", data: [{ label: "A", score: 62 }] },
    { id: "broken", type: "chart", title: "", data: [] },
  ]);

  assert.equal(blocks.length, 2);
  assert.equal(blocks[0]?.type, "comparison_table");
  assert.equal(blocks[1]?.type, "chart");
  assert.equal(slugify("RAG + Knowledge Graphs!"), "rag-knowledge-graphs");
});
