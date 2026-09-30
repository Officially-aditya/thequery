import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");
const MIGRATION = "db/migrations/091_add_gpt_6_1_sol_vs_gpt_6_sol_comparison.sql";

test("the GPT-6.1 Sol vs GPT-6 Sol migration is registered", async () => {
  const runner = await source("scripts/migrate.mjs");
  assert.match(runner, /091_add_gpt_6_1_sol_vs_gpt_6_sol_comparison/);
  assert.ok(
    (await source(MIGRATION)).includes("-- Add GPT-6.1 Sol to the model catalog"),
    "migration file should exist next to its registration",
  );
});

test("GPT-6.1 Sol gets a full catalog profile with the spec, pricing and behavior labels", async () => {
  const migration = await source(MIGRATION);

  for (const field of [
    "Developer",
    "Release date",
    "API model ID",
    "Context window",
    "Max output",
    "Knowledge cutoff",
    "Reasoning / effort",
    "Input / 1M tokens",
    "Cached input / 1M",
    "Cache write / 1M",
    "Output / 1M tokens",
    "Batch / flex discount",
    "Long-context surcharge",
    "Text input",
    "Image / vision input",
    "Audio input",
    "Video input",
    "File / document input",
    "Text output",
    "Image output",
    "Audio output",
    "Video output",
    "Tool / function calling",
    "Computer use",
    "API access",
    "Product access",
    "Weights / license",
    "Primary focus",
    "Long-horizon work",
    "Agent orchestration",
    "User collaboration",
    "Efficiency / generation change",
    "Safety / approvals",
  ]) assert.ok(migration.includes(`\"${field}\"`), `${field} should be populated`);

  assert.match(migration, /'gpt-6-1-sol',\s*'GPT-6\.1 Sol',\s*'OpenAI',\s*DATE '2026-09-29'/);
  assert.match(migration, /"API model ID": "gpt-6\.1-sol"/);
  assert.match(migration, /"Context window": "1,050,000 tokens"/);
  assert.match(migration, /"Input \/ 1M tokens": "\$2"/);
  assert.match(migration, /"Cached input \/ 1M": "\$0\.10"/);
  assert.match(migration, /"Output \/ 1M tokens": "\$10"/);
  assert.match(migration, /ON CONFLICT \(slug\) DO UPDATE SET/);
});

test("both launch scorecards are stored as normalized benchmark evidence", async () => {
  const migration = await source(MIGRATION);

  for (const benchmark of [
    "DeepSWE v1.1",
    "Terminal-Bench Science 0.1",
    "Terminal-Bench 4.0",
    "OSWorld 2.0",
    "AutomationBench",
    "AutomationBench-AA",
    "GDP.pdf",
    "HealthBench",
    "GDPval-AA",
    "AA-Briefcase",
    "AA-Omniscience",
  ]) assert.ok(migration.includes(`\"benchmark_name\":\"${benchmark}\"`), `${benchmark} should be normalized`);

  for (const score of [
    "75.2%",
    "71.9%",
    "57.0%",
    "71.4%",
    "36.1%",
    "32.0%",
    "64.2%",
    "56.1%",
    "1575 Elo",
    "1564 Elo",
    "64.9%",
    "54.3%",
    "28.0%",
    "27.6%",
    "43.9%",
    "60.1%",
  ]) assert.ok(migration.includes(`\"score_display\":\"${score}\"`), `${score} should be normalized`);

  assert.match(migration, /ON CONFLICT \(id\) DO UPDATE SET/);
  assert.match(migration, /jsonb_to_recordset\(\$benchmarks\$/);
});

test("DeepSWE keeps both effort readings because the score peaks below max", async () => {
  const migration = await source(MIGRATION);

  assert.match(migration, /\"id\":\"tq-20260929-gpt61sol-deepswe11-high\"[\s\S]*?\"reasoning_effort\":\"high\"/);
  assert.match(migration, /\"id\":\"tq-20260929-gpt61sol-deepswe11-max\"[\s\S]*?\"reasoning_effort\":\"max\"/);
  assert.match(migration, /scoring lower at max than at high/);
});

test("unreported cells stay unseeded rather than inferred", async () => {
  const migration = await source(MIGRATION);

  assert.match(migration, /AutomationBench-AA for GPT-6 Sol \(reported as n\/a by Artificial Analysis\)/);
  assert.ok(
    !migration.includes("\"id\":\"tq-20260929-gpt6sol-automationbench-aa\""),
    "Artificial Analysis reported no GPT-6 Sol value for AutomationBench-AA",
  );
  assert.ok(
    !migration.includes("\"model_slug\":\"gpt-6-1-sol\",\"category\":\"coding\",\"benchmark_name\":\"FrontierCode\""),
    "OpenAI did not report FrontierCode for GPT-6.1 Sol",
  );
  assert.ok(
    !migration.includes("\"model_slug\":\"gpt-6-1-sol\",\"category\":\"agentic_computer_use\",\"benchmark_name\":\"Agents' Last Exam\""),
    "OpenAI did not report Agents' Last Exam for GPT-6.1 Sol",
  );
});

test("the Artificial Analysis index stays out of model_benchmarks", async () => {
  const migration = await source(MIGRATION);

  assert.match(migration, /category-level score/);
  assert.ok(
    !migration.includes("\"benchmark_name\":\"Intelligence Index"),
    "the index is a category score, so it belongs in the authored table only",
  );
  assert.ok(migration.includes("Artificial Analysis Intelligence Index v4.3.2 (max)"));
});

test("the authored comparison follows the canonical block order and metadata contract", async () => {
  const migration = await source(MIGRATION);

  assert.match(migration, /'comparison:gpt-6-1-sol-vs-gpt-6-sol'/);
  assert.match(migration, /'gpt-6-1-sol-vs-gpt-6-sol',\s*'',\s*'comparisons\/gpt-6-1-sol-vs-gpt-6-sol'/);
  assert.match(migration, /'GPT-6\.1 Sol vs GPT-6 Sol'/);
  assert.match(migration, /'modelA', 'gpt-6-1-sol',\s*'modelB', 'gpt-6-sol'/);
  assert.match(migration, /DATE '2026-09-29'/);

  const order = [
    "Specifications",
    "Pricing",
    "Cost per task",
    "Capabilities & access",
    "Model behavior",
    "Coding",
    "Knowledge",
    "Agentic & computer use",
    "Professional",
  ].map((title) => migration.indexOf(`\"title\": \"${title}\"`));

  for (const [index, position] of order.entries()) {
    assert.ok(position >= 0, `section ${index} should be present`);
    if (index > 0) assert.ok(position > order[index - 1], `section ${index} should follow section ${index - 1}`);
  }

  assert.ok(migration.includes("markdown-gpt61sol-6sol-bottom-line"));
  assert.ok(migration.includes("## Bottom line"));
  assert.doesNotMatch(migration, /\"title\": \"Benchmarks\"/);
});

test("the migration's own caveat and migration-cost warnings survive into the page", async () => {
  const migration = await source(MIGRATION);

  assert.match(migration, /transcribed second-hand/);
  assert.match(migration, /mixes effort levels/);
  assert.match(migration, /the newer model is worse on that row/);
  assert.match(migration, /## What breaks when you switch/);
  assert.match(migration, /reasoning is always on/);
  assert.match(migration, /requires the Responses API/);
  assert.match(migration, /regresses against GPT-6 Sol on SciCode/);
});
