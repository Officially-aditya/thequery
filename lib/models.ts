import "server-only";

import { getSql } from "./db";
import type { Source } from "./content-types";

export type ModelAccess = "proprietary" | "restricted" | "open_weights" | "open_source";
export type ModelBenchmarkCategory =
  | "coding"
  | "math_reasoning"
  | "knowledge"
  | "agentic_computer_use"
  | "multimodal"
  | "professional"
  | "other";

export interface ModelBenchmarkEvidence {
  id: string;
  category: ModelBenchmarkCategory;
  benchmarkName: string;
  benchmarkVersion: string | null;
  scoreNumeric: number | null;
  scoreDisplay: string;
  scoreUnit: string | null;
  tools: boolean | null;
  reasoningEffort: string | null;
  harness: string | null;
  evaluator: string | null;
  evaluationDate: string | null;
  source: string | null;
  notes: string | null;
}

export interface ModelCatalogEntry {
  slug: string;
  name: string;
  developer: string;
  family: string | null;
  releaseDate: string | null;
  gaDate: string | null;
  access: ModelAccess;
  comparisonData: Record<string, string>;
  benchmarks: ModelBenchmarkEvidence[];
  sources: Source[];
  notes: string | null;
  metadata: Record<string, unknown>;
  verifiedAt: string;
}

interface ModelRow {
  slug: string;
  name: string;
  developer: string;
  family: string | null;
  release_date: string | Date | null;
  ga_date: string | Date | null;
  access: ModelAccess;
  comparison_data: unknown;
  sources: unknown;
  notes: string | null;
  metadata: unknown;
  verified_at: string | Date;
}

interface BenchmarkRow {
  id: string;
  model_slug: string;
  category: ModelBenchmarkCategory;
  benchmark_name: string;
  benchmark_version: string | null;
  score_numeric: number | string | null;
  score_display: string;
  score_unit: string | null;
  tools: boolean | null;
  reasoning_effort: string | null;
  harness: string | null;
  evaluator: string | null;
  evaluation_date: string | Date | null;
  source: string | null;
  notes: string | null;
}

function isoDate(value: string | Date | null): string | null {
  if (!value) return null;
  if (value instanceof Date) return value.toISOString().slice(0, 10);
  return String(value).slice(0, 10);
}

function isoDateTime(value: string | Date): string {
  return value instanceof Date ? value.toISOString() : String(value);
}

function comparisonData(value: unknown): Record<string, string> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return Object.fromEntries(
    Object.entries(value)
      .filter((entry): entry is [string, string] => typeof entry[1] === "string")
      .map(([key, entryValue]) => [key, entryValue.trim()]),
  );
}

function metadata(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function sources(value: unknown): Source[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((entry) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) return [];
    const record = entry as Record<string, unknown>;
    const title = typeof record.title === "string" ? record.title.trim() : "";
    const url = typeof record.url === "string" ? record.url.trim() : "";
    return title && url ? [{ title, url }] : [];
  });
}

function benchmarkFromRow(row: BenchmarkRow): ModelBenchmarkEvidence {
  return {
    id: row.id,
    category: row.category,
    benchmarkName: row.benchmark_name,
    benchmarkVersion: row.benchmark_version,
    scoreNumeric: row.score_numeric === null ? null : Number(row.score_numeric),
    scoreDisplay: row.score_display,
    scoreUnit: row.score_unit,
    tools: row.tools,
    reasoningEffort: row.reasoning_effort,
    harness: row.harness,
    evaluator: row.evaluator,
    evaluationDate: isoDate(row.evaluation_date),
    source: row.source,
    notes: row.notes,
  };
}

function benchmarkCell(evidence: ModelBenchmarkEvidence): string {
  const conditions: string[] = [];
  if (evidence.benchmarkVersion && !evidence.benchmarkName.toLowerCase().includes(evidence.benchmarkVersion.toLowerCase())) {
    conditions.push(evidence.benchmarkVersion);
  }
  if (evidence.tools === true) conditions.push("tools");
  if (evidence.tools === false) conditions.push("no tools");
  if (evidence.reasoningEffort) conditions.push(evidence.reasoningEffort);
  if (evidence.harness) conditions.push(evidence.harness);
  if (evidence.evaluator) conditions.push(evidence.evaluator);
  return conditions.length > 0
    ? `${evidence.scoreDisplay} (${conditions.join("; ")})`
    : evidence.scoreDisplay;
}

function enrichComparisonData(base: Record<string, string>, benchmarks: ModelBenchmarkEvidence[]): Record<string, string> {
  const grouped = new Map<string, string[]>();
  for (const evidence of benchmarks) {
    const values = grouped.get(evidence.benchmarkName) ?? [];
    const rendered = benchmarkCell(evidence);
    if (!values.includes(rendered)) values.push(rendered);
    grouped.set(evidence.benchmarkName, values);
  }

  const enriched = { ...base };
  for (const [benchmarkName, values] of grouped) {
    if (!enriched[benchmarkName] && values.length > 0) enriched[benchmarkName] = values.join(" · ");
  }
  return enriched;
}

function enrichSources(base: Source[], benchmarks: ModelBenchmarkEvidence[]): Source[] {
  const all = [...base];
  for (const evidence of benchmarks) {
    if (!evidence.source) continue;
    all.push({
      title: `${evidence.benchmarkName}${evidence.benchmarkVersion ? ` ${evidence.benchmarkVersion}` : ""} evaluation`,
      url: evidence.source,
    });
  }
  return Array.from(new Map(all.map((source) => [source.url, source])).values());
}

function fromRow(row: ModelRow, benchmarkRows: BenchmarkRow[]): ModelCatalogEntry {
  const benchmarks = benchmarkRows.map(benchmarkFromRow);
  return {
    slug: row.slug,
    name: row.name,
    developer: row.developer,
    family: row.family,
    releaseDate: isoDate(row.release_date),
    gaDate: isoDate(row.ga_date),
    access: row.access,
    comparisonData: enrichComparisonData(comparisonData(row.comparison_data), benchmarks),
    benchmarks,
    sources: enrichSources(sources(row.sources), benchmarks),
    notes: row.notes,
    metadata: metadata(row.metadata),
    verifiedAt: isoDateTime(row.verified_at),
  };
}

const MODEL_COLUMNS = "slug, name, developer, family, release_date, ga_date, access, comparison_data, sources, notes, metadata, verified_at";
const BENCHMARK_COLUMNS = "id, model_slug, category, benchmark_name, benchmark_version, score_numeric, score_display, score_unit, tools, reasoning_effort, harness, evaluator, evaluation_date, source, notes";

export async function getModels(): Promise<ModelCatalogEntry[]> {
  const sql = getSql();
  const [rows, benchmarkRows] = await Promise.all([
    sql.query(
      `SELECT ${MODEL_COLUMNS}
       FROM models
       ORDER BY developer ASC, release_date DESC NULLS LAST, name ASC`,
    ) as Promise<ModelRow[]>,
    sql.query(
      `SELECT ${BENCHMARK_COLUMNS}
       FROM model_benchmarks
       ORDER BY model_slug ASC, benchmark_name ASC, evaluation_date ASC NULLS LAST, id ASC`,
    ) as Promise<BenchmarkRow[]>,
  ]);

  const byModel = new Map<string, BenchmarkRow[]>();
  for (const benchmark of benchmarkRows) {
    const current = byModel.get(benchmark.model_slug) ?? [];
    current.push(benchmark);
    byModel.set(benchmark.model_slug, current);
  }
  return rows.map((row) => fromRow(row, byModel.get(row.slug) ?? []));
}

export async function getModelBySlug(slug: string): Promise<ModelCatalogEntry | null> {
  const sql = getSql();
  const [rows, benchmarkRows] = await Promise.all([
    sql.query(
      `SELECT ${MODEL_COLUMNS}
       FROM models
       WHERE slug = $1
       LIMIT 1`,
      [slug],
    ) as Promise<ModelRow[]>,
    sql.query(
      `SELECT ${BENCHMARK_COLUMNS}
       FROM model_benchmarks
       WHERE model_slug = $1
       ORDER BY benchmark_name ASC, evaluation_date ASC NULLS LAST, id ASC`,
      [slug],
    ) as Promise<BenchmarkRow[]>,
  ]);
  return rows[0] ? fromRow(rows[0], benchmarkRows) : null;
}
