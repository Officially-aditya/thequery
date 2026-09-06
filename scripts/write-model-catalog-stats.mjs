import { writeFile } from "node:fs/promises";
import { neon } from "@neondatabase/serverless";
import nextEnv from "@next/env";

const { loadEnvConfig } = nextEnv;
loadEnvConfig(process.cwd());

if (!process.env.NEW_DATABASE_URL) {
  throw new Error("NEW_DATABASE_URL is required to inspect the model catalog.");
}

const sql = neon(process.env.NEW_DATABASE_URL);
const models = await sql.query("SELECT COUNT(*)::int AS count FROM models WHERE release_date >= DATE '2026-01-01' AND release_date < DATE '2027-01-01'");
const benchmarks = await sql.query("SELECT COUNT(*)::int AS count FROM model_benchmarks");
const migrations = await sql.query("SELECT COUNT(*)::int AS count FROM schema_migrations WHERE id = '015_comprehensive_model_catalog'");
const ernie = await sql.query("SELECT COUNT(*)::int AS count FROM models WHERE slug = 'ernie-5-0'");

const stats = {
  models: Number(models[0]?.count ?? 0),
  benchmarks: Number(benchmarks[0]?.count ?? 0),
  migration015: Number(migrations[0]?.count ?? 0),
  ernie2025: Number(ernie[0]?.count ?? 0),
};

await writeFile("public/_model-catalog-stats.json", `${JSON.stringify(stats)}\n`, "utf8");
console.log("Model catalog stats", stats);
