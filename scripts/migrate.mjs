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
