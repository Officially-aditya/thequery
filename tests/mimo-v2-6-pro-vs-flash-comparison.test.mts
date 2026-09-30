import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

const migrationPath = path.join(
  root,
  "db/migrations/095_add_mimo_v2_6_pro_vs_flash_comparison.sql",
);

const migration = await readFile(migrationPath, "utf8");
const runner = await readFile(path.join(root, "scripts/migrate.mjs"), "utf8");

// The 094 catalog migration is the evidence source for the scores quoted on
// the page, so the two files are cross-checked rather than restated.
const catalog = await readFile(
  path.join(root, "db/migrations/094_add_mimo_v2_6_model_catalog.sql"),
  "utf8",
);

// Read the blocks back out of the migration rather than restating them, so
// these tests fail if the file and the assertions drift apart.
const blocks = JSON.parse(
  migration.slice(
    migration.indexOf("$blocks$") + "$blocks$".length,
    migration.lastIndexOf("$blocks$"),
  ),
);

const table = (title: string) => {
  const found = blocks.find(
    (b: { type: string; title?: string }) =>
      b.type === "spec_table" && b.title === title,
  );
  assert.ok(found, `expected a spec_table titled "${title}"`);
  return found as { columns: string[]; rows: string[][] };
};

const rowFor = (title: string, label: string) => {
  const t = table(title);
  const row = t.rows.find((r) => r[0] === label);
  assert.ok(row, `expected a "${label}" row in "${title}"`);
  return row;
};

test("comparison migration is registered", () => {
  assert.ok(runner.includes("095_add_mimo_v2_6_pro_vs_flash_comparison"));
});

test("comparison compares Pro against Flash, not UltraSpeed", () => {
  assert.ok(migration.includes("'mimo-v2-6-pro-vs-mimo-v2-6-flash'"));
  assert.ok(migration.includes("'modelA', 'mimo-v2-6-pro'"));
  assert.ok(migration.includes("'modelB', 'mimo-v2-6-flash'"));
  // UltraSpeed has no scores of its own, so it must not be a compared model.
  assert.ok(!/model[AB]', 'mimo-v2-6-pro-ultraspeed'/.test(migration));
});

test("every spec table puts Flash first and Pro second", () => {
  const tables = blocks.filter((b: { type: string }) => b.type === "spec_table");
  assert.ok(tables.length >= 8, `expected several spec tables, got ${tables.length}`);
  for (const t of tables) {
    assert.deepEqual(t.columns, ["MiMo V2.6 Flash", "MiMo V2.6 Pro"], t.title);
  }
});

test("specs carry the confirmed context window and distinct parameters", () => {
  const specs = table("Specifications");
  const byLabel = (l: string) => {
    const r = specs.rows.find((x) => x[0] === l);
    assert.ok(r, `missing spec row ${l}`);
    return r;
  };
  assert.equal(byLabel("Context window")[1], "1,000,000 tokens");
  assert.equal(byLabel("Context window")[2], "1,000,000 tokens");
  assert.equal(byLabel("Max output")[1], "128,000 tokens");
  assert.equal(byLabel("Total / active parameters")[1], "309B / 15B");
  assert.equal(byLabel("Total / active parameters")[2], "1.02T / 42B");
  // The stale inference is gone.
  assert.ok(!/architecture''s 1M maximum/.test(migration));
  assert.ok(!/does not state a separate context/.test(migration));
});

test("pricing reflects the documented rates and the cache gap", () => {
  assert.equal(rowFor("Pricing", "Input / 1M tokens (cache miss)")[1], "**$0.14**");
  assert.equal(rowFor("Pricing", "Input / 1M tokens (cache miss)")[2], "$0.435");
  assert.equal(rowFor("Pricing", "Output / 1M tokens")[1], "**$0.28**");
  assert.equal(rowFor("Pricing", "Output / 1M tokens")[2], "$0.87");
  assert.equal(rowFor("Pricing", "Cached input / 1M")[1], "**$0.0028**");
  assert.equal(rowFor("Pricing", "Cached input / 1M")[2], "$0.0036");
  assert.equal(rowFor("Pricing", "Cache miss-to-hit discount")[1], "~50x");
  assert.equal(rowFor("Pricing", "Cache miss-to-hit discount")[2], "~120x");
});

test("UltraSpeed access limits are stated rather than inherited from Pro", () => {
  const row = rowFor("Capabilities & access", "MiMo-V2.6-Pro-UltraSpeed variant");
  assert.equal(row[1], "None");
  const note = row[2];
  assert.match(note, /not production-ready/);
  assert.match(note, /capability column blank/);
  assert.match(note, /no evaluation page/);
  assert.match(note, /no published rate limits/);
  assert.match(note, /no Batch API/);
  assert.match(note, /no Token Plan coverage/);
  // Pro and Flash's own capabilities must be stated plainly, not "unstated".
  assert.equal(rowFor("Capabilities & access", "Tool / function calling")[1],
    "Yes, with streaming, structured output and web search");
});

test("in-house benchmarks are labelled unreproducible wherever they appear", () => {
  for (const t of blocks.filter((b: { type: string }) => b.type === "spec_table")) {
    for (const r of t.rows) {
      if (/in-house/.test(r[0])) {
        assert.match(r[0], /unreproducible/, `${r[0]} is not flagged as unreproducible`);
      }
    }
  }
});

test("the unreported Flash GDPval figure is not filled in from Pro", () => {
  const gdp = rowFor("Knowledge", "GDPval 2.1 (AA, Elo)");
  assert.equal(gdp[1], "Not reported");
  assert.equal(gdp[2], "**1673 Elo**");
  const behavior = table("Model behavior");
  const index = behavior.rows.find((r) =>
    r[0].startsWith("Artificial Analysis Intelligence Index"),
  );
  assert.ok(index);
  // Artificial Analysis scores Flash at 38 and Pro at 46, independently of Xiaomi.
  assert.equal(index![1], "**38**, at $0.06 per index task");
  assert.match(index![2], /46/);
  // The index score belongs in the behavior table only, never as a benchmark
  // section, since no model in the catalog has an index evidence row.
  for (const t of blocks.filter((b: { type: string }) => b.type === "spec_table")) {
    if (t.title === "Model behavior") continue;
    for (const r of t.rows) {
      assert.ok(!/Intelligence Index/.test(r[0]), `${r[0]} leaks the index score`);
    }
  }
});

test("Flash is not described as omnimodal, because Artificial Analysis lists text and image only", () => {
  const caps = table("Capabilities & access");
  for (const key of ["Audio input", "Video input"]) {
    const row = caps.rows.find((r) => r[0] === key);
    assert.ok(row, `${key} row is missing`);
    assert.match(row![1], /^No/, `Flash ${key} should not be claimed`);
    assert.match(row![2], /speech and video/);
  }
  assert.equal(rowFor("Capabilities & access", "Text input")[1], "Yes");
  assert.equal(rowFor("Capabilities & access", "Image / vision input")[1], "Yes, native");
});

test("Flash leading on CyberGym is kept, since it is the measured result", () => {
  assert.equal(rowFor("Cyber capability", "CyberGym")[1], "**95.1%**");
  assert.equal(rowFor("Cyber capability", "CyberGym")[2], "94.0%");
  assert.equal(rowFor("Cyber capability", "ExploitBench")[1], "25.3%");
  assert.equal(rowFor("Cyber capability", "ExploitBench")[2], "**47.9%**");
});

test("benchmark scores match the 094 evidence rows for both models", () => {
  const evidence = JSON.parse(
    catalog.slice(
      catalog.indexOf("$benchmarks$") + "$benchmarks$".length,
      catalog.lastIndexOf("$benchmarks$"),
    ),
  ) as { model_slug: string; benchmark_name: string; score_display: string }[];

  const find = (model: string, name: string) => {
    const row = evidence.find(
      (e) => e.model_slug === model && e.benchmark_name === name,
    );
    assert.ok(row, `no evidence row for ${model} / ${name}`);
    return row.score_display;
  };

  // The authored page and the catalog must tell the same story. score_display
  // already carries the percent sign, so compare against it directly.
  const deepswe = table("Coding").rows.find((r) => r[0] === "DeepSWE v1.1");
  assert.equal(deepswe![1], find("mimo-v2-6-flash", "DeepSWE v1.1"));
  assert.equal(deepswe![2], `**${find("mimo-v2-6-pro", "DeepSWE v1.1")}**`);

  const tb = table("Coding").rows.find((r) => r[0] === "Terminal-Bench 4.0");
  assert.equal(tb![1], find("mimo-v2-6-flash", "Terminal-Bench 4.0"));
  assert.equal(tb![2], `**${find("mimo-v2-6-pro", "Terminal-Bench 4.0")}**`);

  const cyber = table("Cyber capability").rows.find((r) => r[0] === "CyberGym");
  assert.equal(cyber![1], `**${find("mimo-v2-6-flash", "CyberGym")}**`);
  assert.equal(cyber![2], find("mimo-v2-6-pro", "CyberGym"));
});

test("prose carries the caveat footnote and the two closing sections", () => {
  const notes = migration.slice(
    migration.indexOf("$notes$") + "$notes$".length,
    migration.lastIndexOf("$notes$"),
  );
  assert.match(notes, /Xiaomi's own transcription/);
  assert.match(notes, /no safety evaluation published/i);
  const markdown = blocks.filter((b: { type: string }) => b.type === "markdown");
  const headings = markdown
    .map((m: { content: string }) => m.content.split("\n")[0])
    .join(" | ");
  assert.match(headings, /## What changes when you switch/);
  assert.match(headings, /## Bottom line/);
});

test("seo metadata is present and within length bounds", () => {
  const desc = migration.match(/'seoDescription', '([^']+)'/);
  assert.ok(desc, "expected a seoDescription");
  const d = desc![1];
  assert.ok(d.length >= 140 && d.length <= 160, `seoDescription is ${d.length} chars`);
  assert.match(migration, /'seoKeywords', jsonb_build_array/);
});

test("the comparison cites at most three sources", () => {
  const start = migration.indexOf("jsonb_build_array(\n    jsonb_build_object('title', 'Introducing");
  assert.ok(start > 0);
  const end = migration.indexOf("),\n  jsonb_build_object(\n    'modelA'", start);
  const block = migration.slice(start, end);
  const count = (block.match(/jsonb_build_object\('title'/g) ?? []).length;
  assert.equal(count, 3);
});
