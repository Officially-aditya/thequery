import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { neon } from "@neondatabase/serverless";
import nextEnv from "@next/env";

const migrationId = "001_initial";
const migrationFile = new URL("../db/migrations/001_initial.sql", import.meta.url);

const { loadEnvConfig } = nextEnv;
loadEnvConfig(process.cwd());

function getSql() {
  if (!process.env.DATABASE_URL) {
    throw new Error("DATABASE_URL is required. Add your Neon connection string to .env.");
  }
  return neon(process.env.DATABASE_URL);
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
  const existing = await sql.query("SELECT id FROM schema_migrations WHERE id = $1", [migrationId]);
  if (existing.length > 0) {
    console.log(`Database migration ${migrationId} is already applied.`);
    return;
  }

  const migration = await readFile(migrationFile, "utf8");
  for (const statement of splitStatements(migration)) {
    await sql.query(statement);
  }
  await sql.query("INSERT INTO schema_migrations (id) VALUES ($1)", [migrationId]);
  console.log(`Applied database migration ${migrationId}.`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  migrate().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
