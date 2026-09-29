import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");

test("Sonnet 5.5 comparison migration is registered", async () => {
  const runner = await source("scripts/migrate.mjs");
  assert.match(runner, /088_add_claude_sonnet_5_5_comparison/);
});

test("Sonnet 5.5 gets a full catalog profile with the spec, pricing and behavior labels", async () => {
  const migration = await source("db/migrations/088_add_claude_sonnet_5_5_comparison.sql");

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

  assert.match(migration, /'claude-sonnet-5-5',\s*'Claude Sonnet 5\.5',\s*'Anthropic',\s*DATE '2026-09-28'/);
  assert.match(migration, /"API model ID": "claude-sonnet-5-5"/);
  assert.match(migration, /"Context window": "1M tokens"/);
  assert.match(migration, /"Input \/ 1M tokens": "\$2"/);
  assert.match(migration, /"Output \/ 1M tokens": "\$10"/);
  assert.match(migration, /ON CONFLICT \(slug\) DO UPDATE SET/);
});

test("the launch scorecard is stored as normalized benchmark evidence", async () => {
  const migration = await source("db/migrations/088_add_claude_sonnet_5_5_comparison.sql");

  for (const benchmark of [
    "Terminal-Bench 4.0",
    "FrontierCode 1.1 Main",
    "CursorBench",
    "GDPval-AA",
    "AA-Briefcase",
    "Humanity's Last Exam",
    "OSWorld 2.1",
    "Chartography",
  ]) assert.ok(migration.includes(`\"benchmark_name\":\"${benchmark}\"`), `${benchmark} should be normalized`);

  for (const score of [
    "70.6%",
    "52.1%",
    "46.2%",
    "55.5%",
    "1844 Elo",
    "1811 Elo",
    "64.5%",
    "80.1%",
    "61.6%",
    "10.3%",
    "1449 Elo",
    "1359 Elo",
    "54.9%",
    "57.0%",
    "15.6%",
    "1822 Elo",
    "1487 Elo",
    "1483 Elo",
    "53.6%",
  ]) assert.ok(migration.includes(`\"score_display\":\"${score}\"`), `${score} should be normalized`);

  assert.match(migration, /ON CONFLICT \(id\) DO UPDATE SET/);
  assert.match(migration, /jsonb_to_recordset\(\$benchmarks\$/);
});

test("FrontierCode keeps both effort readings and the OSWorld label is canonicalized", async () => {
  const migration = await source("db/migrations/088_add_claude_sonnet_5_5_comparison.sql");

  assert.match(migration, /\"id\":\"tq-20260928-sonnet55-frontiercode11main-xhigh\"[\s\S]*?\"reasoning_effort\":\"xhigh\"/);
  assert.match(migration, /\"id\":\"tq-20260928-sonnet55-frontiercode11main-max\"[\s\S]*?\"reasoning_effort\":\"max\"/);
  assert.match(
    migration,
    /UPDATE model_benchmarks SET\s+benchmark_name = 'OSWorld 2\.1',\s+updated_at = NOW\(\)\s+WHERE id IN \('tq-20260922-opus55-osworld20-partial', 'tq-20260922-fable51-osworld20-partial'\)/,
  );
});

test("vendor caveats and unreported rows are recorded, not invented", async () => {
  const migration = await source("db/migrations/088_add_claude_sonnet_5_5_comparison.sql");

  assert.match(migration, /pre-release Sonnet 5\.5 deployment/);
  assert.match(migration, /degraded image understanding in GPT-6 Sol/);
  assert.match(migration, /GPT-6 Sol did not report this benchmark publicly/);
  assert.ok(
    !migration.includes("\"model_slug\":\"gpt-6-sol\",\"category\":\"coding\",\"benchmark_name\":\"Terminal-Bench 4.0\""),
    "unreported GPT-6 Sol rows must stay unseeded",
  );
});

test("OSWorld 2.1 is a visible agentic comparison label", async () => {
  const [comparison, client] = await Promise.all([
    source("lib/model-comparison.ts"),
    source("components/admin/admin-client.ts"),
  ]);

  assert.match(comparison, /"OSWorld 2\.1"/);
  assert.match(client, /\["OSWorld 2\.1", "", ""\]/);
});
