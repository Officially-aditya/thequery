import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");

test("Muse comparison depth migration is registered", async () => {
  const runner = await source("scripts/migrate.mjs");
  assert.match(runner, /037_deepen_muse_spark_comparisons/);
});

test("Muse Spark 1.2 and 1.3 receive rich comparison fields", async () => {
  const migration = await source("db/migrations/037_deepen_muse_spark_comparisons.sql");

  for (const field of [
    "Context window",
    "Cached input / 1M",
    "File / document input",
    "Primary focus",
    "Long-horizon work",
    "Agent orchestration",
  ]) assert.ok(migration.includes(`\"${field}\"`), `${field} should be populated`);

  assert.match(migration, /User collaboration/);
  assert.match(migration, /Efficiency \/ generation change/);
  assert.match(migration, /~20% fewer tool calls/);
  assert.match(migration, /~25% fewer tokens/);
  assert.match(migration, /confirms before consequential actions/);
  assert.match(migration, /max for 1\.3 while the 1\.2 comparison column uses xhigh/);
});

test("Muse generation comparison includes long-context, coding and agentic scorecard rows", async () => {
  const migration = await source("db/migrations/037_deepen_muse_spark_comparisons.sql");

  for (const benchmark of [
    "SWE-Atlas Codebase QnA",
    "MRCR v2 256K–512K",
    "MRCR v2 512K–1M",
    "JobBench",
    "DeepSearchQA",
    "Agentic IF Index",
    "OSWorld 2.0",
    "AutomationBench",
  ]) assert.ok(migration.includes(`\"benchmark_name\":\"${benchmark}\"`), `${benchmark} should be normalized`);

  assert.match(migration, /\"muse-spark-1-3\"[\s\S]*?\"MRCR v2 512K–1M\"[\s\S]*?\"98\.1%\"/);
  assert.match(migration, /\"muse-spark-1-2\"[\s\S]*?\"MRCR v2 512K–1M\"[\s\S]*?\"55\.5%\"/);
  assert.match(migration, /\"muse-spark-1-3\"[\s\S]*?\"SWE-Atlas Codebase QnA\"[\s\S]*?\"59\.4%\"/);
});

test("generated comparisons render normalized benchmarks dynamically by category", async () => {
  const [models, comparison] = await Promise.all([
    source("lib/models.ts"),
    source("lib/model-comparison.ts"),
  ]);

  assert.match(models, /category: ModelBenchmarkCategory/);
  assert.match(models, /benchmarks: ModelBenchmarkDisplay\[\]/);
  assert.match(models, /SELECT model_slug, category, benchmark_name/);
  assert.match(comparison, /for \(const benchmark of \[\.\.\.modelA\.benchmarks, \.\.\.modelB\.benchmarks\]\)/);
  assert.match(comparison, /benchmarkSection\[benchmark\.category\]/);
  assert.match(comparison, /dynamic\.get\(section\.title\)/);
});

test("new comparison behavior fields and key Muse benchmarks are visible sections", async () => {
  const comparison = await source("lib/model-comparison.ts");

  assert.match(comparison, /title: "Model behavior"/);
  assert.match(comparison, /"Long-horizon work"/);
  assert.match(comparison, /"Efficiency \/ generation change"/);
  assert.match(comparison, /"SWE-Atlas Codebase QnA"/);
  assert.match(comparison, /"MRCR v2 512K–1M"/);
  assert.match(comparison, /"DeepSearchQA"/);
  assert.match(comparison, /"JobBench"/);
});

test("Muse authored comparisons still preserve non-empty editorial cells", async () => {
  const migration = await source("db/migrations/037_deepen_muse_spark_comparisons.sql");

  assert.match(migration, /WHEN COALESCE\(re\.row_value->>1,''\)='' AND a\.rendered IS NOT NULL/);
  assert.match(migration, /WHEN COALESCE\(re\.row_value->>2,''\)='' AND d\.rendered IS NOT NULL/);
});
