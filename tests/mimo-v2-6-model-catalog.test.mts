import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

const migrationPath = path.join(
  root,
  "db/migrations/094_add_mimo_v2_6_model_catalog.sql",
);

const migration = await readFile(migrationPath, "utf8");
const runner = await readFile(path.join(root, "scripts/migrate.mjs"), "utf8");

const slugs = ["mimo-v2-6-pro", "mimo-v2-6-flash", "mimo-v2-6-pro-ultraspeed"];

type EvidenceRow = {
  id: string;
  model_slug: string;
  category: string;
  benchmark_name: string;
  benchmark_version: string | null;
  score_display: string;
  score_unit: string;
  evaluator: string;
  source: string;
};

// The dollar-quoted block already carries its own JSON array brackets, so it
// parses as-is. Reading it back from the migration rather than restating the
// numbers is what makes these tests a drift guard.
const evidence: EvidenceRow[] = JSON.parse(
  migration.slice(
    migration.indexOf("$benchmarks$") + "$benchmarks$".length,
    migration.lastIndexOf("$benchmarks$"),
  ),
);

test("MiMo V2.6 catalog migration is registered and seeds three model rows", () => {
  assert.match(runner, /094_add_mimo_v2_6_model_catalog/);

  for (const slug of slugs) {
    assert.match(migration, new RegExp(`'${slug}'`));
  }

  // No WHERE NOT EXISTS guards. A guard turns the INSERT into a no-op when the
  // row already exists, which silently defeats ON CONFLICT DO UPDATE and left
  // content corrections unapplied on re-run.
  assert.equal(
    (migration.match(/WHERE NOT EXISTS \(SELECT 1 FROM models WHERE slug = 'mimo-/g) ??
      []).length,
    0,
  );
  // Each row upserts, which is what makes a re-run apply corrections rather
  // than skip them.
  assert.equal(
    (migration.match(/ON CONFLICT \(slug\) DO UPDATE SET/g) ?? []).length,
    3,
  );
});

test("every SQL string literal in the migration has balanced quoting", () => {
  // An unescaped apostrophe inside a single-quoted literal silently breaks the
  // statement, and the migrate runner reports it as a syntax error far from the
  // offending text. This walks each statement and checks the quote count.
  const statements = migration
    .split(/;\s*(?:\r?\n|$)/)
    .map((s) => s.trim())
    .filter(Boolean);

  for (const [index, statement] of statements.entries()) {
    const withoutComments = statement
      .split("\n")
      .filter((line) => !line.trimStart().startsWith("--"))
      .join("\n");
    // Dollar-quoted blocks carry literal text, so exclude them first.
    const withoutDollarQuotes = withoutComments.replace(/\$[a-z_]+\$[\s\S]*?\$[a-z_]+\$/g, "");
    const unescaped = withoutDollarQuotes.replace(/''/g, "").split("'").length - 1;
    assert.equal(unescaped % 2, 0, `statement ${index} has unbalanced quoting`);
  }
});

test("MiMo V2.6 catalog migration carries the glossary footnotes forward", () => {
  // 093 published the glossary pages from the same sources, so the catalog rows
  // have to keep the caveats rather than restate the numbers bare.
  assert.match(migration, /Xiaomi's own transcription/);
  assert.match(migration, /in-house benchmarks?/);
  assert.match(migration, /September 21, 2026/);
  assert.match(migration, /prose and its own per-step table disagree/);
  assert.match(migration, /no safety evaluation was published/i);
  assert.match(migration, /MIT license/i);
  assert.match(migration, /MiMo-V2\.6-Distill-Qwen-9B is deliberately not seeded/);

  // The competitor-column rule, and the specific divergence that made it
  // necessary.
  assert.match(migration, /Competitor rows are not inserted from this table/);
  assert.match(migration, /59\.6/);
  assert.match(migration, /57\.9/);
  assert.match(migration, /26\.8/);
  assert.match(migration, /31\.2/);

  // The Intelligence Index reading is a category score, not a benchmark, which
  // is the same call 091 made.
  // The footnote wraps across comment lines, so match the two halves.
  assert.match(migration, /not normalized into model_benchmarks/);
  assert.match(migration, /where no other\s*\n--\s*model has an index row/);
  assert.match(migration, /"independent_intelligence_index": 46/);
  // Artificial Analysis's own measured figures, verified against its model
  // pages rather than transcribed from Xiaomi.
  assert.match(migration, /"independent_intelligence_index": 38/);
  assert.match(migration, /"artificial_analysis_output_tokens_per_second": 41\.1/);
  assert.match(migration, /"artificial_analysis_time_to_first_token_seconds": 4\.24/);
  assert.match(migration, /"artificial_analysis_index_cost_per_task_usd": 0\.13/);
  assert.match(migration, /"artificial_analysis_index_cost_per_task_usd": 0\.06/);
  // Flash's modality set, per Artificial Analysis: text and image only.
  assert.match(migration, /"input_modalities": "text, image"/);
  assert.match(migration, /"Audio input": "No\. Artificial Analysis lists Flash/);
  // UltraSpeed has no Artificial Analysis page at all.
  assert.match(migration, /"Independent evaluation": "None\./);
});

test("the Artificial Analysis source URLs resolve to real model pages", () => {
  assert.match(migration, /artificialanalysis\.ai\/models\/mimo-v2-6-pro"/);
  assert.match(migration, /artificialanalysis\.ai\/models\/mimo-v2-6-flash"/);
  // The old /artificial-analysis-intelligence-index-model/ path 404s.
  assert.doesNotMatch(migration, /artificial-analysis-intelligence-index-model/);
});

test("MiMo V2.6 catalog migration keeps the statement splitter intact", () => {
  const statements = migration
    .split(/;\s*(?:\r?\n|$)/)
    .map((statement) => statement.trim())
    .filter(Boolean);

  assert.equal(statements.length, 4);

  // No semicolon at end of line inside the dollar-quoted blocks, or the runner
  // would cut a statement in half.
  for (const marker of ["$benchmarks$", "$body$"]) {
    const index = migration.indexOf(marker);
    if (index === -1) continue;
    const block = migration.slice(index);
    assert.ok(
      !/;\s*\n/.test(block.slice(0, block.indexOf(marker, 1) + 1)),
      `${marker} block contains a statement-terminating semicolon`,
    );
  }
});

test("MiMo V2.6 catalog migration seeds one benchmark per published cell", () => {
  // 16 for Pro, 15 for Flash. Flash has no GDPval 2.1 figure in Xiaomi's
  // table, so that cell is left unseeded rather than inferred.
  assert.equal(evidence.length, 31);
  assert.equal(evidence.filter((row) => row.model_slug === "mimo-v2-6-pro").length, 16);
  assert.equal(
    evidence.filter((row) => row.model_slug === "mimo-v2-6-flash").length,
    15,
  );

  // Every value is Xiaomi's transcription, so the evaluator is Xiaomi and not
  // the lab whose model a competitor column describes.
  for (const row of evidence) {
    assert.equal(row.evaluator, "Xiaomi");
    assert.equal(row.source, "https://mimo.mi.com/models/en-US/mimo-v2.6-pro");
    assert.match(row.id, /^tq-20260922-mimov26(pro|flash)-/);
    assert.ok(row.score_display.length > 0);
  }

  assert.equal(new Set(evidence.map((row) => row.id)).size, evidence.length);
});

test("MiMo V2.6 benchmark names match the existing catalog so comparisons join", () => {
  const seen = new Map(evidence.map((row) => [row.benchmark_name, row]));

  // Name plus version, so a row cannot silently join a different protocol.
  for (const [name, version] of [
    ["DeepSWE v1.1", null],
    ["Terminal-Bench 4.0", null],
    ["ProgramBench", null],
    ["Toolathlon", "Verified"],
    ["AutomationBench", "1.0.6"],
    ["Agents' Last Exam", null],
    ["OSWorld-Verified", null],
    ["JobBench", null],
    ["GDPval-AA", "v2.1"],
    ["ExploitBench", null],
  ] as const) {
    assert.equal(seen.get(name)?.benchmark_version, version, name);
  }

  // Categories follow the existing rows rather than a fresh taxonomy, so a
  // joined comparison does not split one benchmark across two sections.
  for (const name of [
    "DeepSWE v1.1",
    "Terminal-Bench 4.0",
    "CyberGym",
    "SEC-Bench Pro",
  ]) {
    assert.equal(seen.get(name)?.category, "coding", name);
  }
  for (const name of [
    "Toolathlon",
    "AutomationBench",
    "Agents' Last Exam",
    "OSWorld-Verified",
    "ExploitBench",
    "GDPval-AA",
  ]) {
    assert.equal(seen.get(name)?.category, "agentic_computer_use", name);
  }
  assert.equal(seen.get("JobBench")?.category, "professional");
  assert.equal(seen.get("MiMo Visual Coding")?.category, "multimodal");

  // New to the catalog, and named so the vendor is unambiguous.
  for (const name of ["ProgramBench", "ExploitGym", "MiMo Code Bench", "MiMo Cyber Bench", "MiMo Visual Coding"]) {
    assert.ok(seen.has(name), name);
  }

  // No "(in-house)" suffix leaking into a stored benchmark name.
  for (const row of evidence) {
    assert.ok(!row.benchmark_name.includes("(in-house)"));
  }
});

test("MiMo V2.6 benchmark scores agree with the published glossary table", () => {
  const score = (model: string, name: string) =>
    evidence.find(
      (row) => row.model_slug === model && row.benchmark_name === name,
    )?.score_display;

  // Pro and Flash columns of the MiMo-V2.6-Pro model page table, which is the
  // same table 093 transcribed into the glossary.
  assert.equal(score("mimo-v2-6-pro", "DeepSWE v1.1"), "71.9%");
  assert.equal(score("mimo-v2-6-pro", "Terminal-Bench 4.0"), "34.9%");
  assert.equal(score("mimo-v2-6-pro", "OSWorld-Verified"), "82.0%");
  assert.equal(score("mimo-v2-6-pro", "GDPval-AA"), "1673 Elo");
  assert.equal(score("mimo-v2-6-pro", "CyberGym"), "94.0%");
  assert.equal(score("mimo-v2-6-pro", "SEC-Bench Pro"), "66.3%");
  assert.equal(score("mimo-v2-6-pro", "MiMo Cyber Bench"), "81.7%");
  assert.equal(score("mimo-v2-6-flash", "DeepSWE v1.1"), "67.9%");
  assert.equal(score("mimo-v2-6-flash", "Terminal-Bench 4.0"), "28.8%");
  assert.equal(score("mimo-v2-6-flash", "CyberGym"), "95.1%");
  assert.equal(score("mimo-v2-6-flash", "ExploitBench"), "25.3%");

  // Flash has no GDPval 2.1 cell, so no row is invented for it.
  assert.equal(score("mimo-v2-6-flash", "GDPval-AA"), undefined);
});

test("MiMo V2.6 UltraSpeed is a serving variant with no borrowed scores", () => {
  // Identical weights, so duplicating Pro's evidence under a second slug would
  // make one measurement look like two.
  assert.match(migration, /"benchmarks_inherited_from": "mimo-v2-6-pro"/);
  assert.match(migration, /"serving_variant_of": "mimo-v2-6-pro"/);
  assert.ok(
    !migration.includes('"model_slug":"mimo-v2-6-pro-ultraspeed"'),
    "UltraSpeed must not carry benchmark rows",
  );
  assert.match(migration, /\$0\.036/);
  assert.match(migration, /\$4\.35/);
  assert.match(migration, /\$8\.70/);
});

test("MiMo V2.6 catalog rows record the license and the access classification", () => {
  // Unmodified MIT is OSI-approved, so open_source, matching how the catalog
  // already stores Apache-2.0 models. Kimi's Modified MIT is open_weights
  // because a modified license is not open source.
  for (const slug of slugs) {
    const block = migration.slice(
      migration.indexOf(`'${slug}'`),
      migration.indexOf(`'${slug}'`) + 4000,
    );
    assert.match(block.slice(0, 400), /'open_source'/, slug);
    assert.match(block, /MIT/, slug);
    assert.match(block, /Xiaomi MiMo/, slug);
    assert.match(block, /'MiMo V2\.6'/, slug);
  }
});

test("no catalog row carries more than three sources", () => {
  for (const slug of slugs) {
    const start = migration.indexOf(`'${slug}',`);
    assert.ok(start > 0, slug);
    const sourcesAt = migration.indexOf("'[\n", start) + 3;
    const end = migration.indexOf("]'::jsonb", sourcesAt);
    const block = migration.slice(sourcesAt, end);
    const count = (block.match(/\{"title"/g) ?? []).length;
    assert.equal(count, 3, `${slug} has ${count} sources`);
  }
});
