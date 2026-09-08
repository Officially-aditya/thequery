import test from "node:test";
import assert from "node:assert/strict";
import { benchmarkCsv } from "../lib/research-csv.ts";

test("export preserves zero, false, unknowns and quoted multiline provenance", () => {
  const csv = benchmarkCsv([{ score_numeric: 0, tools: false, source: 'https://example.org/a,b', notes: 'Says "hello"\nnext line' }]);
  assert.ok(csv.includes('"0","","","false"'));
  assert.ok(csv.includes('"https://example.org/a,b","Says ""hello""\nnext line"'));
  assert.ok(csv.endsWith("\r\n"));
});

test("export neutralizes spreadsheet formulas in strings but retains numeric negatives", () => {
  const csv = benchmarkCsv([{ score_numeric: -1, notes: " =HYPERLINK(\"https://example.org\")" }]);
  assert.ok(csv.includes('"-1"'));
  assert.ok(csv.includes('"\' =HYPERLINK('));
});
