import { NextResponse } from "next/server";
import { getSql } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function GET() {
  const sql = getSql();
  const [models, benchmarks, migrations, ernie] = await Promise.all([
    sql.query("SELECT COUNT(*)::int AS count FROM models WHERE release_date >= DATE '2026-01-01' AND release_date < DATE '2027-01-01'"),
    sql.query("SELECT COUNT(*)::int AS count FROM model_benchmarks"),
    sql.query("SELECT COUNT(*)::int AS count FROM schema_migrations WHERE id = '015_comprehensive_model_catalog'"),
    sql.query("SELECT COUNT(*)::int AS count FROM models WHERE slug = 'ernie-5-0'"),
  ]);
  return NextResponse.json({
    models: Number(models[0]?.count ?? 0),
    benchmarks: Number(benchmarks[0]?.count ?? 0),
    migration015: Number(migrations[0]?.count ?? 0),
    ernie2025: Number(ernie[0]?.count ?? 0),
  });
}
