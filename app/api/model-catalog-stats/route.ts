import { NextResponse } from "next/server";
import { getSql } from "@/lib/db";

export const dynamic = "force-dynamic";

type CountRow = { count: number | string };

export async function GET() {
  const sql = getSql();
  const models = await sql.query("SELECT COUNT(*)::int AS count FROM models WHERE release_date >= DATE '2026-01-01' AND release_date < DATE '2027-01-01'") as CountRow[];
  const benchmarks = await sql.query("SELECT COUNT(*)::int AS count FROM model_benchmarks") as CountRow[];
  const migrations = await sql.query("SELECT COUNT(*)::int AS count FROM schema_migrations WHERE id = '015_comprehensive_model_catalog'") as CountRow[];
  const ernie = await sql.query("SELECT COUNT(*)::int AS count FROM models WHERE slug = 'ernie-5-0'") as CountRow[];

  return NextResponse.json({
    models: Number(models[0]?.count ?? 0),
    benchmarks: Number(benchmarks[0]?.count ?? 0),
    migration015: Number(migrations[0]?.count ?? 0),
    ernie2025: Number(ernie[0]?.count ?? 0),
  });
}
