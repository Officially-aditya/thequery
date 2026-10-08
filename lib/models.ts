import "server-only";

import { unstable_cache } from "next/cache";
import { getSql } from "./db";
import type { Source } from "./content-types";
import { benchmarkValue, highestReasoningEffort, selectBestBenchmarks, type BenchmarkRow } from "./model-benchmarks";

export type ModelAccess = "proprietary" | "restricted" | "open_weights" | "open_source";
export type ModelBenchmarkCategory = "coding" | "math_reasoning" | "knowledge" | "agentic_computer_use" | "multimodal" | "professional" | "other";

export interface ModelCatalogOption {
  slug: string;
  name: string;
  developer: string;
  access: ModelAccess;
}

export interface ModelBenchmarkDisplay {
  category: ModelBenchmarkCategory;
  name: string;
  value: string;
}

export interface ModelCatalogEntry extends ModelCatalogOption {
  releaseDate: string | null;
  comparisonData: Record<string, string>;
  benchmarks: ModelBenchmarkDisplay[];
  sources: Source[];
  notes: string | null;
  verifiedAt: string;
}

interface ModelRow {
  slug: string;
  name: string;
  developer: string;
  release_date: string | Date | null;
  access: ModelAccess;
  comparison_data: unknown;
  sources: unknown;
  notes: string | null;
  verified_at: string | Date;
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

function withBenchmarks(base: Record<string, string>, rows: BenchmarkRow[], highestEffort: string | null): Record<string, string> {
  const next = { ...base };
  for (const row of rows) {
    next[row.benchmark_name] = benchmarkValue(row, highestEffort);
  }
  return next;
}

function benchmarkDisplays(rows: BenchmarkRow[], highestEffort: string | null): ModelBenchmarkDisplay[] {
  return rows.map((row) => ({ category: row.category, name: row.benchmark_name, value: benchmarkValue(row, highestEffort) }));
}

function withBenchmarkSources(base: Source[], rows: BenchmarkRow[]): Source[] {
  const additions = rows.flatMap((row): Source[] => row.source
    ? [{ title: `${row.benchmark_name} evaluation`, url: row.source }]
    : []);
  return Array.from(new Map([...base, ...additions].map((source) => [source.url, source])).values());
}

function fromRow(row: ModelRow, benchmarks: BenchmarkRow[] = []): ModelCatalogEntry {
  const best = selectBestBenchmarks(benchmarks);
  const highestEffort = highestReasoningEffort(benchmarks);
  return {
    slug: row.slug,
    name: row.name,
    developer: row.developer,
    releaseDate: isoDate(row.release_date),
    access: row.access,
    comparisonData: withBenchmarks(comparisonData(row.comparison_data), best, highestEffort),
    benchmarks: benchmarkDisplays(best, highestEffort),
    sources: withBenchmarkSources(sources(row.sources), benchmarks),
    notes: row.notes,
    verifiedAt: isoDateTime(row.verified_at),
  };
}

function groupBenchmarks(rows: BenchmarkRow[]): Map<string, BenchmarkRow[]> {
  const grouped = new Map<string, BenchmarkRow[]>();
  for (const row of rows) {
    const current = grouped.get(row.model_slug) ?? [];
    current.push(row);
    grouped.set(row.model_slug, current);
  }
  return grouped;
}

async function queryModelOptions(): Promise<ModelCatalogOption[]> {
  const sql = getSql();
  return await sql.query(
    `SELECT slug, name, developer, access
     FROM models
     ORDER BY developer ASC, release_date DESC NULLS LAST, name ASC`,
  ) as ModelCatalogOption[];
}

export async function getModelOptions(): Promise<ModelCatalogOption[]> {
  return unstable_cache(
    queryModelOptions,
    ["model-catalog-options-v1"],
  )();
}

export async function getModelsBySlugs(slugs: string[]): Promise<ModelCatalogEntry[]> {
  const uniqueSlugs = Array.from(new Set(slugs.map((slug) => slug.trim()).filter(Boolean)));
  if (uniqueSlugs.length === 0) return [];

  const sql = getSql();
  const rows = await sql.query(
    `SELECT slug, name, developer, release_date, access, comparison_data, sources, notes, verified_at
     FROM models
     WHERE slug = ANY($1::text[])`,
    [uniqueSlugs],
  ) as ModelRow[];
  if (rows.length === 0) return [];

  const benchmarkRows = await sql.query(
    `SELECT model_slug, category, benchmark_name, benchmark_version, score_numeric, score_display, tools, reasoning_effort, harness, evaluator, source
     FROM model_benchmarks
     WHERE model_slug = ANY($1::text[])
     ORDER BY model_slug ASC, benchmark_name ASC, evaluation_date ASC NULLS LAST, id ASC`,
    [uniqueSlugs],
  ) as BenchmarkRow[];
  const benchmarks = groupBenchmarks(benchmarkRows);
  const bySlug = new Map(rows.map((row) => [row.slug, fromRow(row, benchmarks.get(row.slug) ?? [])]));
  return uniqueSlugs.flatMap((slug) => {
    const model = bySlug.get(slug);
    return model ? [model] : [];
  });
}

export async function getModels(): Promise<ModelCatalogEntry[]> {
  const options = await getModelOptions();
  return getModelsBySlugs(options.map((model) => model.slug));
}

export async function getModelBySlug(slug: string): Promise<ModelCatalogEntry | null> {
  const [model] = await getModelsBySlugs([slug]);
  return model ?? null;
}
