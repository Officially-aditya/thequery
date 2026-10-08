import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const migration = await readFile(new URL("../db/migrations/102_add_claude_haiku_5_5_model_catalog.sql", import.meta.url), "utf8");
const [model] = JSON.parse(migration.split("$models$")[1]);
const benchmarks = JSON.parse(migration.split("$benchmarks$")[1]);

const aaRow = (name: string, effort: string) => benchmarks.find((row: {
  benchmark_name: string; reasoning_effort: string; evaluator: string;
}) => row.benchmark_name === name && row.reasoning_effort === effort && row.evaluator === "Artificial Analysis");

test("Haiku 5.5 keeps both prompt-length price tiers and the default effort", () => {
  const [short, long] = model.metadata.pricing_tiers;
  assert.equal(short.max_prompt_tokens, 100000);
  assert.equal(long.min_prompt_tokens, 100001);
  for (const key of ["input_per_million_usd", "output_per_million_usd", "cache_read_per_million_usd", "cache_write_per_million_usd"]) {
    assert.equal(long[key], short[key] * 5);
  }
  assert.equal(short.input_per_million_usd, 0.10);
  assert.equal(short.output_per_million_usd, 0.50);
  assert.equal(model.metadata.default_effort, "medium");
  assert.match(model.comparison_data["Max output"], /128K.*300K/);
});

test("Haiku AA evidence retains units and settings instead of merging unlike scores", () => {
  assert.equal(aaRow("GDPval-AA", "max").score_numeric, 1620.05);
  assert.equal(aaRow("GDPval-AA", "max").score_unit, "Elo");
  assert.equal(aaRow("AA-LCR", "max").score_numeric, 0.826666666666667);
  assert.equal(aaRow("AA-LCR", "max").score_unit, "score");
  assert.ok(Math.abs(aaRow("Terminal-Bench 4.0", "max").score_numeric - 32.8282828282828) < 1e-10);
  assert.ok(Math.abs(aaRow("AA-Omniscience Hallucination Rate", "max").score_numeric - 40.4244170814776) < 1e-10);
  assert.equal(aaRow("AA-Omniscience Index", "max").score_unit, "points");
  assert.equal(aaRow("Terminal-Bench 4.0", "medium").score_display, "15.2% (AA)");
  const launch = benchmarks.find((row: { benchmark_name: string; evaluator: string }) => row.benchmark_name === "Terminal-Bench 4.0" && row.evaluator === "Anthropic");
  assert.equal(launch.score_numeric, 39.2);
  assert.equal(launch.reasoning_effort, null);
  assert.ok(benchmarks.every((row: { model_slug: string; evaluation_date: unknown }) => row.model_slug === "claude-haiku-5-5" && row.evaluation_date === null));
  assert.ok(!benchmarks.some((row: { benchmark_name: string }) => row.benchmark_name === "GPQA Diamond" || row.benchmark_name === "MMMU-Pro"));
});

test("Haiku composite indexes and API measurements stay separate from benchmark rows", () => {
  const profiles = model.metadata.artificial_analysis_effort_profiles;
  assert.deepEqual(profiles.map((p: { effort: string }) => p.effort), ["low", "medium", "high", "xhigh", "max"]);
  assert.equal(model.metadata.independent_intelligence_index_effort, "medium");
  assert.equal(model.metadata.independent_intelligence_index, 34.4646519474657);
  assert.equal(profiles.at(-1).intelligence_index, 43.3950199670746);
  assert.ok(profiles.at(-1).time_to_first_answer_token_seconds > 432);
  assert.ok(profiles[1].time_to_first_answer_token_seconds < 14);
  assert.ok(!benchmarks.some((row: { benchmark_name: string }) => /Intelligence Index|Output speed|Latency/.test(row.benchmark_name)));
  assert.equal(new Set(benchmarks.map((row: { id: string }) => row.id)).size, benchmarks.length);
  assert.equal(migration.split(/;\s*(?:\r?\n|$)/).map(s => s.trim()).filter(Boolean).length, 3);
});
