import "server-only";

import { randomUUID } from "crypto";
import { getSql } from "./db";
import { normalizeBlocks, normalizeSources } from "./content-utils";
import type {
  ContentItem,
  ContentKind,
  ContentStatus,
  Source,
  ContentBlock,
} from "./content-types";

interface ContentRow {
  id: string;
  kind: ContentKind;
  slug: string;
  parent_slug: string;
  path: string;
  title: string;
  summary: string;
  body: string;
  blocks: unknown;
  sources: unknown;
  metadata: unknown;
  status: ContentStatus;
  published_at: string | null;
  sort_order: number;
  created_at: string;
  updated_at: string;
}

export interface UpsertContentInput {
  id?: string;
  kind: ContentKind;
  slug: string;
  parentSlug?: string | null;
  title: string;
  summary?: string;
  body?: string;
  blocks?: ContentBlock[];
  sources?: Source[];
  metadata?: Record<string, unknown>;
  status?: ContentStatus;
  publishedAt?: string | null;
  sortOrder?: number;
}

export function contentPath(kind: ContentKind, slug: string, parentSlug?: string | null): string {
  if (kind === "chapter") {
    if (!parentSlug) throw new Error("A chapter requires a parent book slug.");
    return `books/${parentSlug}/${slug}`;
  }
  const collection = kind === "glossary" ? "glossary" : `${kind}s`;
  return `${collection}/${slug}`;
}

function toObject(value: unknown): Record<string, unknown> {
  if (typeof value === "string") {
    try {
      return toObject(JSON.parse(value));
    } catch {
      return {};
    }
  }
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function toContentItem(row: ContentRow): ContentItem {
  return {
    id: row.id,
    kind: row.kind,
    slug: row.slug,
    parentSlug: row.parent_slug || null,
    path: row.path,
    title: row.title,
    summary: row.summary,
    body: row.body,
    blocks: normalizeBlocks(typeof row.blocks === "string" ? JSON.parse(row.blocks) : row.blocks),
    sources: normalizeSources(typeof row.sources === "string" ? JSON.parse(row.sources) : row.sources),
    metadata: toObject(row.metadata),
    status: row.status,
    publishedAt: row.published_at,
    sortOrder: row.sort_order,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

export async function getContentItems(
  kind: ContentKind,
  options: { parentSlug?: string | null; includeDrafts?: boolean } = {},
): Promise<ContentItem[]> {
  const sql = getSql();
  const parentSlug = options.parentSlug ?? "";
  const rows = options.includeDrafts
    ? await sql`
        SELECT * FROM content_items
        WHERE kind = ${kind} AND parent_slug = ${parentSlug}
        ORDER BY sort_order ASC, published_at DESC NULLS LAST, title ASC
      `
    : await sql`
        SELECT * FROM content_items
        WHERE kind = ${kind} AND parent_slug = ${parentSlug} AND status = 'published'
        ORDER BY sort_order ASC, published_at DESC NULLS LAST, title ASC
      `;

  return (rows as ContentRow[]).map(toContentItem);
}

export async function getContentItem(
  kind: ContentKind,
  slug: string,
  parentSlug?: string | null,
  includeDrafts = false,
): Promise<ContentItem | null> {
  const sql = getSql();
  const rows = includeDrafts
    ? await sql`
        SELECT * FROM content_items
        WHERE kind = ${kind} AND slug = ${slug} AND parent_slug = ${parentSlug ?? ""}
        LIMIT 1
      `
    : await sql`
        SELECT * FROM content_items
        WHERE kind = ${kind} AND slug = ${slug} AND parent_slug = ${parentSlug ?? ""} AND status = 'published'
        LIMIT 1
      `;

  const row = (rows as ContentRow[])[0];
  return row ? toContentItem(row) : null;
}

export async function upsertContent(input: UpsertContentInput): Promise<ContentItem> {
  const sql = getSql();
  const parentSlug = input.parentSlug ?? "";
  const path = contentPath(input.kind, input.slug, parentSlug);
  const rows = await sql`
    INSERT INTO content_items (
      id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
      status, published_at, sort_order
    ) VALUES (
      ${input.id ?? randomUUID()}, ${input.kind}, ${input.slug}, ${parentSlug}, ${path},
      ${input.title}, ${input.summary ?? ""}, ${input.body ?? ""},
      ${JSON.stringify(input.blocks ?? [])}::jsonb,
      ${JSON.stringify(input.sources ?? [])}::jsonb,
      ${JSON.stringify(input.metadata ?? {})}::jsonb,
      ${input.status ?? "published"}, ${input.publishedAt ?? null}, ${input.sortOrder ?? 0}
    )
    ON CONFLICT (kind, slug, parent_slug) DO UPDATE SET
      title = EXCLUDED.title,
      summary = EXCLUDED.summary,
      body = EXCLUDED.body,
      blocks = EXCLUDED.blocks,
      sources = EXCLUDED.sources,
      metadata = EXCLUDED.metadata,
      status = EXCLUDED.status,
      published_at = EXCLUDED.published_at,
      sort_order = EXCLUDED.sort_order,
      path = EXCLUDED.path,
      updated_at = NOW()
    RETURNING *
  `;

  return toContentItem((rows as ContentRow[])[0]);
}

export async function deleteContentItem(
  kind: ContentKind,
  slug: string,
  parentSlug?: string | null,
): Promise<void> {
  const sql = getSql();
  const resolvedParentSlug = parentSlug ?? "";

  if (kind === "book") {
    await sql`DELETE FROM content_items WHERE kind = 'chapter' AND parent_slug = ${slug}`;
  }

  await sql`
    DELETE FROM content_items
    WHERE kind = ${kind} AND slug = ${slug} AND parent_slug = ${resolvedParentSlug}
  `;
}

export async function getContentCounts(): Promise<Record<ContentKind, number>> {
  const sql = getSql();
  const rows = await sql`
    SELECT kind, COUNT(*)::int AS count
    FROM content_items
    GROUP BY kind
  `;
  const counts: Record<ContentKind, number> = {
    article: 0,
    guide: 0,
    glossary: 0,
    book: 0,
    chapter: 0,
  };
  for (const row of rows as Array<{ kind: ContentKind; count: number }>) {
    counts[row.kind] = Number(row.count);
  }
  return counts;
}
