import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");

test("OpenAI and Mythos enrichment migrations are registered in order", async () => {
  const runner = await source("scripts/migrate.mjs");
  const openai = runner.indexOf("034_enrich_openai_remaining");
  const mythos = runner.indexOf("035_enrich_anthropic_mythos");
  assert.ok(openai > 0);
  assert.ok(mythos > openai);
});

test("remaining OpenAI models receive complete catalog profiles", async () => {
  const migration = await source("db/migrations/034_enrich_openai_remaining.sql");
  for (const slug of [
    "gpt-5-3-codex",
    "gpt-5-3-codex-spark",
    "gpt-5-3-instant",
    "gpt-5-4",
    "gpt-5-4-pro",
    "gpt-5-5-pro",
    "gpt-5-5-instant",
  ]) assert.ok(migration.includes(`'${slug}'`), `${slug} should be enriched`);

  assert.match(migration, /'gpt-5-3-codex'[\s\S]*?"Context window":"400K"[\s\S]*?"Max output":"128K"[\s\S]*?"Knowledge cutoff":"Aug 31, 2025"/);
  assert.match(migration, /'gpt-5-4'[\s\S]*?"Context window":"1\.05M"[\s\S]*?"Input \/ 1M tokens":"\$2\.50"[\s\S]*?"Computer use":"Yes — native Responses API computer use"/);
  assert.match(migration, /'gpt-5-4-pro'[\s\S]*?"Input \/ 1M tokens":"\$30"[\s\S]*?"Output \/ 1M tokens":"\$180"/);
  assert.match(migration, /'gpt-5-5-pro'[\s\S]*?"Knowledge cutoff":"Dec 1, 2025"[\s\S]*?"Computer use":"No"/);
  assert.match(migration, /'gpt-5-5-instant'[\s\S]*?"API model ID":"chat-latest \(GPT-5\.5 Instant rolling alias\)"[\s\S]*?"Context window":"400K"/);
  assert.match(migration, /gpt-5\.3-chat-latest[\s\S]*?"catalog_status":"deprecated"/);
});

test("OpenAI enrichment stores only stable public numeric benchmark evidence", async () => {
  const migration = await source("db/migrations/034_enrich_openai_remaining.sql");
  const benchmarkSection = migration.split("WITH b(model_slug")[1]?.split("INSERT INTO model_benchmarks")[0] ?? "";
  const rows = benchmarkSection.match(/^\('gpt-/gm) ?? [];
  assert.equal(rows.length, 27, "expected 27 normalized OpenAI benchmark rows");

  for (const expected of [
    "'SWE-bench Pro','Public',56.8",
    "'Terminal-Bench 2.0','',77.3",
    "'OSWorld-Verified','',74.0",
    "'MCP Atlas','Public',70.6",
    "'BrowseComp','',90.1",
    "'GPQA Diamond','',94.4",
    "'FrontierMath','Tier 4',39.6",
  ]) assert.ok(migration.includes(expected), `${expected} should be seeded`);

  assert.match(migration, /Supersedes the originally reported 64\.7%/);
  assert.doesNotMatch(benchmarkSection, /\('gpt-5-3-codex-spark'/);
  assert.doesNotMatch(benchmarkSection, /\('gpt-5-3-instant'/);
  assert.doesNotMatch(benchmarkSection, /\('gpt-5-5-instant'/);
  assert.doesNotMatch(migration, /GDPval-AA[^\n]*83\.0/);
});

test("Mythos catalog profiles preserve restricted access and current pricing", async () => {
  const migration = await source("db/migrations/035_enrich_anthropic_mythos.sql");
  for (const slug of ["claude-mythos-preview", "claude-mythos-5", "claude-mythos-5-1"])
    assert.ok(migration.includes(`'${slug}'`), `${slug} should be enriched`);

  assert.match(migration, /'claude-mythos-5'[\s\S]*?"Context window":"1M"[\s\S]*?"Knowledge cutoff":"Jan 2026"[\s\S]*?"Cached input \/ 1M":"\$1"/);
  assert.match(migration, /'claude-mythos-5-1'[\s\S]*?"Knowledge cutoff":"Jun 2026"[\s\S]*?"Cached input \/ 1M":"\$0\.25"/);
  assert.match(migration, /'restricted',CASE WHEN slug='claude-mythos-preview'/);
  assert.match(migration, /Historical per-token pricing is intentionally omitted/);
});

test("Mythos Preview gets direct Anthropic capability benchmarks", async () => {
  const migration = await source("db/migrations/035_enrich_anthropic_mythos.sql");
  const direct = migration.split("WITH b(model_slug")[1]?.split("INSERT INTO model_benchmarks")[0] ?? "";
  const rows = direct.match(/^\('claude-mythos-preview'/gm) ?? [];
  assert.equal(rows.length, 9, "expected nine direct Mythos Preview benchmark rows");

  for (const expected of [
    "'SWE-bench Verified','',93.9",
    "'SWE-bench Pro','',77.8",
    "'Terminal-Bench 2.0','',82.0",
    "'Terminal-Bench 2.1','4h timeout',92.1",
    "'GPQA Diamond','',94.6",
    "'Humanity''s Last Exam','',56.8",
    "'Humanity''s Last Exam','',64.7",
    "'BrowseComp','',86.9",
    "'OSWorld-Verified','',79.6",
  ]) assert.ok(migration.includes(expected), `${expected} should be seeded`);
});

test("Mythos 5 variants inherit only ordinary same-model capability evidence", async () => {
  const migration = await source("db/migrations/035_enrich_anthropic_mythos.sql");
  assert.match(migration, /src\.model_slug='claude-fable-5'/);
  assert.match(migration, /src\.model_slug='claude-fable-5-1'/);
  assert.match(migration, /Shared underlying model with Claude Fable 5 per Anthropic/);
  assert.match(migration, /Shared underlying model with Claude Fable 5\.1 per Anthropic/);
  assert.match(migration, /src\.benchmark_name NOT ILIKE 'OSWorld%'/);
  assert.match(migration, /src\.benchmark_name<>'AutomationBench'/);
  assert.doesNotMatch(migration, /SET evaluator='Anthropic'/);
});

test("both enrichments only fill blank authored comparison benchmark cells", async () => {
  for (const file of [
    "db/migrations/034_enrich_openai_remaining.sql",
    "db/migrations/035_enrich_anthropic_mythos.sql",
  ]) {
    const migration = await source(file);
    assert.match(migration, /COALESCE\(re\.row_value->>1,''\)='' AND a\.rendered IS NOT NULL/);
    assert.match(migration, /COALESCE\(re\.row_value->>2,''\)='' AND d\.rendered IS NOT NULL/);
    assert.doesNotMatch(migration, /old_a\.value/);
    assert.doesNotMatch(migration, /old_b\.value/);
  }
});
