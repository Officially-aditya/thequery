import { neon } from "@neondatabase/serverless";
import nextEnv from "@next/env";

const { loadEnvConfig } = nextEnv;
loadEnvConfig(process.cwd());

if (!process.env.NEW_DATABASE_URL) {
  throw new Error("NEW_DATABASE_URL is required to verify the model catalog.");
}

const sql = neon(process.env.NEW_DATABASE_URL);

const [migrationRows, modelRows, benchmarkRows, staleRows] = await Promise.all([
  sql.query("SELECT COUNT(*)::int AS count FROM schema_migrations WHERE id = '015_comprehensive_model_catalog'"),
  sql.query("SELECT COUNT(*)::int AS count FROM models WHERE release_date >= DATE '2026-01-01' AND release_date < DATE '2027-01-01'"),
  sql.query("SELECT COUNT(*)::int AS count FROM model_benchmarks"),
  sql.query("SELECT COUNT(*)::int AS count FROM models WHERE slug = 'ernie-5-0'"),
]);

const migrationCount = Number(migrationRows[0]?.count ?? 0);
const modelCount = Number(modelRows[0]?.count ?? 0);
const benchmarkCount = Number(benchmarkRows[0]?.count ?? 0);
const staleCount = Number(staleRows[0]?.count ?? 0);

if (migrationCount !== 1) {
  throw new Error("Model catalog migration 015 is not recorded as applied.");
}
if (modelCount < 95) {
  throw new Error(`Expected at least 95 verified 2026 models, found ${modelCount}.`);
}
if (benchmarkCount < 100) {
  throw new Error(`Expected at least 100 benchmark evidence rows, found ${benchmarkCount}.`);
}
if (staleCount !== 0) {
  throw new Error("ERNIE 5.0 is a 2025 launch and must not remain in the 2026 model catalog.");
}

console.log(`Verified model catalog: ${modelCount} models, ${benchmarkCount} benchmark evidence rows.`);
