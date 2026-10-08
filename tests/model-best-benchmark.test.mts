import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { benchmarkValue, highestReasoningEffort, selectBestBenchmarks, type BenchmarkRow } from "../lib/model-benchmarks.ts";

const migration = await readFile(new URL("../db/migrations/102_add_claude_haiku_5_5_model_catalog.sql", import.meta.url), "utf8");
const haiku: BenchmarkRow[] = JSON.parse(migration.split("$benchmarks$")[1]);

function row(score: number | null, effort: string | null, extra: Partial<BenchmarkRow> = {}): BenchmarkRow {
  return {
    model_slug: "model", category: "coding", benchmark_name: "FrontierCode 1.1 Main",
    benchmark_version: null, score_numeric: score, score_display: `${score}%`,
    tools: null, reasoning_effort: effort, harness: null, evaluator: null, source: null,
    ...extra,
  };
}

test("the six Haiku Terminal-Bench values collapse to the best reported score", () => {
  const terminal = haiku.filter(r => r.benchmark_name === "Terminal-Bench 4.0");
  assert.equal(terminal.length, 6);
  const [best] = selectBestBenchmarks(terminal);
  assert.equal(best.score_numeric, 39.2);
  assert.equal(benchmarkValue(best, highestReasoningEffort(haiku)), "39.2% (Anthropic)");
  assert.equal(selectBestBenchmarks(terminal).length, 1);
});

test("max effort is omitted while a better result at xhigh keeps its effort label", () => {
  const aaTerminal = haiku.filter(r => r.benchmark_name === "Terminal-Bench 4.0" && r.evaluator === "Artificial Analysis");
  const [aaBest] = selectBestBenchmarks(aaTerminal);
  assert.equal(benchmarkValue(aaBest, highestReasoningEffort(haiku)), "32.8% (AA)");

  const sonnet = [row(52.1, "xhigh"), row(46.2, "max")];
  const [best] = selectBestBenchmarks(sonnet);
  assert.equal(benchmarkValue(best, highestReasoningEffort(sonnet)), "52.1% (xhigh)");
});

test("effort labels are relative to the model's highest recorded setting", () => {
  const rows = [row(80, "high"), row(70, "max", { benchmark_name: "Another benchmark" })];
  assert.equal(benchmarkValue(selectBestBenchmarks(rows)[0], highestReasoningEffort(rows)), "80% (high)");
});

test("score ties prefer the higher effort setting regardless of row order", () => {
  for (const rows of [[row(80, "high"), row(80, "max")], [row(80, "max"), row(80, "high")]]) {
    const [best] = selectBestBenchmarks(rows);
    assert.equal(best.reasoning_effort, "max");
    assert.equal(benchmarkValue(best, highestReasoningEffort(rows)), "80%");
  }
});

test("hallucination rates use the lowest score and Omniscience Index uses the highest", () => {
  const selected = selectBestBenchmarks(haiku);
  const hallucination = selected.find(r => r.benchmark_name === "AA-Omniscience Hallucination Rate")!;
  assert.equal(hallucination.score_display, "40.4% (AA)");
  const index = selected.find(r => r.benchmark_name === "AA-Omniscience Index")!;
  assert.equal(index.score_display, "10.7 points (AA)");
  const [legacyRate] = selectBestBenchmarks([row(60, "max", { benchmark_name: "AA-Omniscience" }), row(40, "high", { benchmark_name: "AA-Omniscience" })]);
  assert.equal(legacyRate.score_numeric, 40);
});

test("unknown scores are not treated as zero and selected conditions stay visible", () => {
  const [best] = selectBestBenchmarks([row(null, "max"), row(0, "medium")]);
  assert.equal(best.score_numeric, 0);
  const value = benchmarkValue(row(57.4, null, { benchmark_name: "Humanity's Last Exam", benchmark_version: "v1", tools: true }), "max");
  assert.equal(value, "57.4% (v1; tools)");
});
