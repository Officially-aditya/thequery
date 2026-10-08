import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { selectBestBenchmarks, highestReasoningEffort, benchmarkValue, type BenchmarkRow } from "../lib/model-benchmarks.ts";

const migration = await readFile(new URL("../db/migrations/105_add_mistral_large_4.sql", import.meta.url), "utf8");
const [glossary] = JSON.parse(migration.split("$glossary$")[1]);
const [model] = JSON.parse(migration.split("$models$")[1]);
type EvidenceRow = BenchmarkRow & { id: string; score_unit: string };
const scores: EvidenceRow[] = JSON.parse(migration.split("$benchmarks$")[1]);

const score = (name: string, evaluator: string) => scores.find(row => row.benchmark_name === name && row.evaluator === evaluator)!;

test("Mistral Large 4 records current preview access and preserves the spec disagreement", () => {
  assert.equal(model.access, "proprietary");
  assert.equal(model.ga_date, null);
  assert.equal(model.metadata.weights_available, false);
  assert.equal(model.metadata.weight_license, null);
  assert.equal(model.metadata.official_parameters_active, 52_000_000_000);
  assert.equal(model.metadata.artificial_analysis_parameters_active_billions, 49);
  assert.equal(model.metadata.official_context_window_label, "1M");
  assert.equal(model.metadata.artificial_analysis_context_window_tokens, 524288);
  assert.match(glossary.body, /sources disagree/);
  assert.match(glossary.body, /end of October/);
});

test("introductory pricing stays separate from standard rates and reasoning latency", () => {
  for (const key of ["input_per_million_usd", "cache_read_per_million_usd", "output_per_million_usd"]) {
    assert.equal(model.metadata.pricing_intro[key] * 2, model.metadata.pricing_standard[key]);
  }
  assert.equal(model.metadata.pricing_intro.input_per_million_usd, 0.68);
  assert.equal(model.metadata.pricing_intro.output_per_million_usd, 2.09);
  assert.equal(model.metadata.intro_offer_duration, "First two weeks");
  assert.ok(model.metadata.artificial_analysis_time_to_first_answer_token_seconds > 18);
  assert.ok(model.metadata.artificial_analysis_time_to_first_chunk_seconds < 1.5);
});

test("benchmark evidence retains evaluator differences and score scales", () => {
  assert.equal(score("Terminal-Bench 4.0", "Mistral AI").score_numeric, 28.3);
  assert.ok(Math.abs(score("Terminal-Bench 4.0", "Artificial Analysis").score_numeric! - 26.7676767676768) < 1e-10);
  assert.equal(score("Terminal-Bench 4.0", "Vals AI").score_numeric, 22.73);
  const [best] = selectBestBenchmarks(scores.filter(row => row.benchmark_name === "Terminal-Bench 4.0"));
  assert.equal(benchmarkValue(best, highestReasoningEffort(scores)), "28.3% (Mistral)");
  assert.equal(score("AA-Omniscience Index", "Artificial Analysis").score_numeric, -5.3);
  assert.equal(score("AA-Omniscience Index", "Artificial Analysis").score_unit, "points");
  assert.equal(score("AA-LCR", "Artificial Analysis").score_unit, "score");
  assert.equal(score("GDPval-AA", "Artificial Analysis").score_unit, "Elo");
  assert.equal(score("Harvey's Legal Agent Benchmark", "Vals AI").score_numeric, 15.83);
  assert.equal(score("Harvey's Legal Agent Benchmark", "Vals AI").score_display, "15.83% (Vals)");
  assert.equal(score("Harvey's Legal Agent Benchmark", "Vals AI").benchmark_version, "held-out, task pass rate");
  assert.ok(!scores.some(row => /Intelligence Index|Cyber Index|Coding Agent Index|Vals Index|Harvey LAB-AA/.test(row.benchmark_name)));
  assert.ok(scores.every(row => row.model_slug === "mistral-large-4" && row.reasoning_effort === null));
});

test("the glossary metadata and content match the published migrations", async () => {
  const consolidation = await readFile(new URL("../db/migrations/106_consolidate_mistral_large_4_terminal_bench.sql", import.meta.url), "utf8");
  const updatedBody = glossary.body
    .replace(consolidation.split("$rows_before$")[1], consolidation.split("$rows_after$")[1])
    .replace(consolidation.split("$note_before$")[1], consolidation.split("$note_after$")[1]);
  const entries = JSON.parse(await readFile(new URL("../data/glossary.json", import.meta.url), "utf8"));
  const matches = entries.filter((entry: { slug: string }) => entry.slug === "mistral-large-4");
  assert.equal(matches.length, 1);
  const [entry] = matches;
  assert.equal(entry.fullDef, updatedBody);
  assert.equal(entry.fullDef.match(/^\| Terminal-Bench 4\.0 \|/gm).length, 1);
  assert.ok(entry.fullDef.includes("28.3%\\*"));
  assert.ok(entry.fullDef.includes("Mistral reports 28.3%, Artificial Analysis reports 26.8%, and Vals reports 22.73%"));
  assert.deepEqual(entry.references, glossary.sources);
  assert.equal(entry.seoDescription.length, 144);
  assert.equal(entry.lastUpdated, "2026-10-08");
  assert.equal(entry.fullDef.match(/^## /gm).length, 7);
  assert.ok(!entry.fullDef.includes("—"));
  assert.ok(!entry.fullDef.includes("https://www.thequery.in/glossary/"));
  assert.equal(migration.split(/;\s*(?:\r?\n|$)/).map(statement => statement.trim()).filter(Boolean).length, 3);
  assert.equal(new Set(scores.map(row => row.id)).size, scores.length);
});
