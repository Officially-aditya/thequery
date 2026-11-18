import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");
const MIGRATION = "db/migrations/135_add_gpt_6_2_astra_vs_sol_comparison.sql";

test("the GPT-6.2 Astra vs GPT-6.2 Sol comparison migration is registered", async () => {
  const runner = await source("scripts/migrate.mjs");
  assert.match(runner, /135_add_gpt_6_2_astra_vs_sol_comparison/);
  assert.ok(
    (await source(MIGRATION)).includes("Publish the authored GPT-6.2 Astra vs GPT-6.2 Sol comparison page"),
    "migration file should exist next to its registration",
  );
});

test("the comparison follows canonical metadata and routing contract", async () => {
  const migration = await source(MIGRATION);

  assert.match(migration, /'comparison:gpt-6-2-astra-vs-gpt-6-2-sol'/);
  assert.match(migration, /'gpt-6-2-astra-vs-gpt-6-2-sol',\s*'',\s*'comparisons\/gpt-6-2-astra-vs-gpt-6-2-sol'/);
  assert.match(migration, /'GPT-6\.2 Astra vs GPT-6\.2 Sol'/);
  assert.match(migration, /'modelA', 'gpt-6-2-astra'/);
  assert.match(migration, /'modelB', 'gpt-6-2-sol'/);
  assert.match(migration, /DATE '2026-11-17'/);
  assert.match(migration, /'published'/);
});

test("spec, pricing, capability, and model behavior tables are populated", async () => {
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
    "Reasoning flexibility",
    "Cost-efficiency at scale",
    "Tool orchestration",
  ]) {
    assert.ok(migration.includes(`\"${field}\"`), `${field} should be in comparison tables`);
  }

  // Pricing verification
  assert.match(migration, /"\$8",\s*"\*\*\$2\*\*"/);
  assert.match(migration, /"\$0\.40",\s*"\*\*\$0\.10\*\*"/);
  assert.match(migration, /"\$40",\s*"\*\*\$10\*\*"/);
});

test("benchmark evidence is reported side-by-side with appropriate effort and caveats", async () => {
  const migration = await source(MIGRATION);

  assert.ok(migration.includes("Terminal-Bench-Science 0.1"));
  assert.match(migration, /\*\*~77%\*\* \(77\.1%\)/);
  assert.match(migration, /~68% \(matches GPT-6 Astra best\)/);

  assert.ok(migration.includes("GPQA Diamond"));
  assert.match(migration, /\*\*97\.0%\*\*/);
  assert.match(migration, /96\.0%/);

  assert.ok(migration.includes("Agents' Last Exam"));
  assert.match(migration, /\*\*77\.6%\*\* \(max\)/);

  assert.ok(migration.includes("FrontierCode 2.0"));
  assert.match(migration, /\*\*49\.8%\*\* \(max\)/);
  assert.match(migration, /49\.3% \(max\)/);
});

test("editorial narrative and decision guidance sections are included", async () => {
  const migration = await source(MIGRATION);

  assert.ok(migration.includes("## Architectural and Operational Differences"));
  assert.ok(migration.includes("The Economics of Cache Reuse"));
  assert.ok(migration.includes("Reasoning Overhead & Determinism"));
  assert.ok(migration.includes("API Integration Surface"));
  assert.ok(migration.includes("## Bottom line"));
  assert.ok(migration.includes("Deploy **GPT-6.2 Astra**"));
  assert.ok(migration.includes("Deploy **GPT-6.2 Sol**"));
});
