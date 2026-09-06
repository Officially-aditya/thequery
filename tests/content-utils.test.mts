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

test("normalizeBlocks keeps spec tables with exactly two models and label-value rows", () => {
  const blocks = normalizeBlocks([
    { id: "spec", type: "spec_table", title: "Pricing", columns: ["Model A", "Model B"], rows: [["Input", "$10", "**$0.25**"], ["", "", ""]] },
    { id: "broken", type: "spec_table", columns: ["Only one"], rows: [["Input", "$10"]] },
  ]);

  assert.equal(blocks.length, 1);
  assert.equal(blocks[0]?.type, "spec_table");
  if (blocks[0]?.type === "spec_table") {
    assert.deepEqual(blocks[0].columns, ["Model A", "Model B"]);
    assert.deepEqual(blocks[0].rows, [["Input", "$10", "**$0.25**"]]);
  }
});
