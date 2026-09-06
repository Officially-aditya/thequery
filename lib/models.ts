import "server-only";

import { getSql } from "./db";
import type { Source } from "./content-types";

export type ModelAccess = "proprietary" | "restricted" | "open_weights" | "open_source";

export interface ModelCatalogEntry {
  slug: string;
  name: string;
  developer: string;
  releaseDate: string | null;
  access: ModelAccess;
  comparisonData: Record<string, string>;
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

function fromRow(row: ModelRow): ModelCatalogEntry {
  return {
    slug: row.slug,
    name: row.name,
    developer: row.developer,
    releaseDate: isoDate(row.release_date),
    access: row.access,
    comparisonData: comparisonData(row.comparison_data),
    sources: sources(row.sources),
    notes: row.notes,
    verifiedAt: isoDateTime(row.verified_at),
  };
}

export async function getModels(): Promise<ModelCatalogEntry[]> {
  const sql = getSql();
  const rows = await sql.query(
    `SELECT slug, name, developer, release_date, access, comparison_data, sources, notes, verified_at
     FROM models
     ORDER BY developer ASC, release_date DESC NULLS LAST, name ASC`,
  ) as ModelRow[];
  return rows.map(fromRow);
}

export async function getModelBySlug(slug: string): Promise<ModelCatalogEntry | null> {
  const sql = getSql();
  const rows = await sql.query(
    `SELECT slug, name, developer, release_date, access, comparison_data, sources, notes, verified_at
     FROM models
     WHERE slug = $1
     LIMIT 1`,
    [slug],
  ) as ModelRow[];
  return rows[0] ? fromRow(rows[0]) : null;
}
