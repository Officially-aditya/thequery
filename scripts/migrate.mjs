import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { neon } from "@neondatabase/serverless";
import nextEnv from "@next/env";

const migrations = [
  { id: "001_initial", file: new URL("../db/migrations/001_initial.sql", import.meta.url) },
  { id: "002_add_cover_images", file: new URL("../db/migrations/002_add_cover_images.sql", import.meta.url) },
  { id: "003_update_dense_retrieval", file: new URL("../db/migrations/003_update_dense_retrieval.sql", import.meta.url) },
  { id: "004_update_cuda", file: new URL("../db/migrations/004_update_cuda.sql", import.meta.url) },
  { id: "005_update_benchmark", file: new URL("../db/migrations/005_update_benchmark.sql", import.meta.url) },
  { id: "006_update_lstm_and_neural_network", file: new URL("../db/migrations/006_update_lstm_and_neural_network.sql", import.meta.url) },
  { id: "007_update_gpqa_diamond_and_openclaw", file: new URL("../db/migrations/007_update_gpqa_diamond_and_openclaw.sql", import.meta.url) },
  { id: "008_add_claude_fable_51", file: new URL("../db/migrations/008_add_claude_fable_51.sql", import.meta.url) },
  { id: "009_add_comparisons", file: new URL("../db/migrations/009_add_comparisons.sql", import.meta.url) },
  { id: "010_comparison_template_capabilities", file: new URL("../db/migrations/010_comparison_template_capabilities.sql", import.meta.url) },
  { id: "011_model_catalog", file: new URL("../db/migrations/011_model_catalog.sql", import.meta.url) },
  { id: "012_correct_model_catalog_verification", file: new URL("../db/migrations/012_correct_model_catalog_verification.sql", import.meta.url) },
  { id: "013_expand_model_catalog_modalities", file: new URL("../db/migrations/013_expand_model_catalog_modalities.sql", import.meta.url) },
  { id: "014_canonicalize_existing_comparisons", file: new URL("../db/migrations/014_canonicalize_existing_comparisons.sql", import.meta.url) },
  { id: "015_comprehensive_model_catalog", file: new URL("../db/migrations/015_comprehensive_model_catalog.sql", import.meta.url) },
  { id: "016_enrich_openai_gpt56_gpt55_gpt54_small", file: new URL("../db/migrations/016_enrich_openai_gpt56_gpt55_gpt54_small.sql", import.meta.url) },
  { id: "017_compact_token_counts", file: new URL("../db/migrations/017_compact_token_counts.sql", import.meta.url) },
  { id: "018_enrich_astra_fable", file: new URL("../db/migrations/018_enrich_astra_fable.sql", import.meta.url) },
  { id: "019_enrich_gemini_core", file: new URL("../db/migrations/019_enrich_gemini_core.sql", import.meta.url) },
  { id: "020_enrich_gemini_specialized", file: new URL("../db/migrations/020_enrich_gemini_specialized.sql", import.meta.url) },
  { id: "021_enrich_muse_family", file: new URL("../db/migrations/021_enrich_muse_family.sql", import.meta.url) },
  { id: "022_refresh_enriched_comparisons", file: new URL("../db/migrations/022_refresh_enriched_comparisons.sql", import.meta.url) },
  { id: "023_backfill_agentic_benchmark_labels", file: new URL("../db/migrations/023_backfill_agentic_benchmark_labels.sql", import.meta.url) },
  { id: "024_simplify_benchmark_display", file: new URL("../db/migrations/024_simplify_benchmark_display.sql", import.meta.url) },
  { id: "025_enrich_muse_gemini_benchmarks", file: new URL("../db/migrations/025_enrich_muse_gemini_benchmarks.sql", import.meta.url) },
  { id: "026_seed_frontier_models", file: new URL("../db/migrations/026_seed_frontier_models.sql", import.meta.url) },
  { id: "027_enrich_anthropic_frontier", file: new URL("../db/migrations/027_enrich_anthropic_frontier.sql", import.meta.url) },
];

const { loadEnvConfig } = nextEnv;
loadEnvConfig(process.cwd());

function getSql() {
  if (!process.env.NEW_DATABASE_URL) {
    throw new Error("NEW_DATABASE_URL is required. Add your Neon connection string to .env.");
  }
  return neon(process.env.NEW_DATABASE_URL);
}

function splitStatements(sql) {
  return sql
    .split(/;\s*(?:\r?\n|$)/)
    .map((statement) => statement.trim())
    .filter(Boolean);
}

export async function migrate() {
  const sql = getSql();
  await sql.query(
    "CREATE TABLE IF NOT EXISTS schema_migrations (id TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())",
  );
  for (const migration of migrations) {
    const existing = await sql.query("SELECT id FROM schema_migrations WHERE id = $1", [migration.id]);
    if (existing.length > 0) {
      console.log(`Database migration ${migration.id} is already applied.`);
      continue;
    }

    const sqlSource = await readFile(migration.file, "utf8");
    for (const statement of splitStatements(sqlSource)) {
      await sql.query(statement);
    }
    await sql.query("INSERT INTO schema_migrations (id) VALUES ($1)", [migration.id]);
    console.log(`Applied database migration ${migration.id}.`);
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  migrate().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
