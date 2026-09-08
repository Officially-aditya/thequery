import "server-only";
import { getSql } from "./db";

// Explicit public fields: never export internal model metadata or content drafts.
export async function getResearchData() {
  const sql = getSql();
  const [models, benchmarks] = await Promise.all([
    sql.query(`SELECT slug, name, developer, release_date, access, comparison_data, sources, notes, verified_at FROM models ORDER BY slug`),
    sql.query(`SELECT id, model_slug, category, benchmark_name, benchmark_version, score_numeric, score_display, score_unit, tools, reasoning_effort, harness, evaluator, evaluation_date, source, notes FROM model_benchmarks ORDER BY model_slug, benchmark_name, id`),
  ]);
  return {
    schema_version: "1.0",
    retrieved_at: new Date().toISOString(),
    publisher: "TheQuery",
    methodology: "https://www.thequery.in/research#methodology",
    description: "Compiled reported evaluations, not independent tests by TheQuery. Null fields are unknown. Preserve evaluation conditions when comparing results.",
    models: models as Record<string, unknown>[],
    benchmarks: benchmarks as Record<string, unknown>[],
  };
}
