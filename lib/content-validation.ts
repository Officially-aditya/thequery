import { markdownBlock, normalizeBlocks, normalizeSources, slugify } from "./content-utils";
import type { ContentKind, ContentStatus, Source } from "./content-types";
import type { UpsertContentInput } from "./content";

const contentKinds: ContentKind[] = ["article", "guide", "glossary", "book", "chapter"];

function object(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function text(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function textList(value: unknown): string[] {
  return Array.isArray(value)
    ? [...new Set(value.map(text).filter(Boolean))]
    : [];
}

function validDate(value: string): string | null {
  return /^\d{4}-\d{2}-\d{2}$/.test(value) ? value : null;
}

function metadataFor(kind: ContentKind, raw: Record<string, unknown>): Record<string, unknown> {
  if (kind === "article") {
    return { manualGlossaryLinks: Boolean(raw.manualGlossaryLinks) };
  }
  if (kind === "glossary") {
    return {
      category: text(raw.category) || "Foundations",
      relatedTerms: textList(raw.relatedTerms),
      ...(text(raw.analogy) ? { analogy: text(raw.analogy) } : {}),
      ...(text(raw.seoDescription) ? { seoDescription: text(raw.seoDescription) } : {}),
      ...(textList(raw.seoKeywords).length ? { seoKeywords: textList(raw.seoKeywords) } : {}),
    };
  }
  if (kind === "book") {
    return {
      author: text(raw.author) || "TheQuery",
      ...(validDate(text(raw.lastModified)) ? { lastModified: text(raw.lastModified) } : {}),
    };
  }
  if (kind === "chapter") {
    return validDate(text(raw.lastModified)) ? { lastModified: text(raw.lastModified) } : {};
  }
  return {};
}

export function isContentKind(value: string): value is ContentKind {
  return contentKinds.includes(value as ContentKind);
}

export type ParsedContentInput =
  | { data: UpsertContentInput; errors: [] }
  | { data: null; errors: string[] };

export function parseContentInput(kind: ContentKind, value: unknown): ParsedContentInput {
  const input = object(value);
  const title = text(input.title);
  const slug = slugify(text(input.slug) || title);
  const parentSlug = slugify(text(input.parentSlug));
  const status: ContentStatus = input.status === "draft" ? "draft" : "published";
  const suppliedBody = typeof input.body === "string" ? input.body : "";
  const blocks = normalizeBlocks(input.blocks);
  const normalizedBlocks = blocks.length ? blocks : suppliedBody.trim() ? [markdownBlock(suppliedBody)] : [];
  const body = normalizedBlocks
    .filter((block) => block.type === "markdown")
    .map((block) => block.content)
    .join("\n\n") || suppliedBody;
  const sources: Source[] = normalizeSources(input.sources);
  const errors: string[] = [];

  if (!title) errors.push("A title is required.");
  if (!slug) errors.push("A URL slug is required.");
  if (kind === "chapter" && !parentSlug) errors.push("Choose the book this chapter belongs to.");
  if (status === "published" && ["article", "guide", "glossary", "chapter"].includes(kind) && normalizedBlocks.length === 0) {
    errors.push("Published content needs at least one content block.");
  }

  if (errors.length) return { data: null, errors };

  const publishedAt = validDate(text(input.publishedAt));
  const sortOrderValue = Number(input.sortOrder);
  return {
    data: {
      id: text(input.id) || undefined,
      kind,
      slug,
      parentSlug: kind === "chapter" ? parentSlug : null,
      title,
      summary: typeof input.summary === "string" ? input.summary.trim() : "",
      body,
      blocks: normalizedBlocks,
      sources,
      metadata: metadataFor(kind, object(input.metadata)),
      status,
      publishedAt,
      sortOrder: Number.isFinite(sortOrderValue) ? Math.trunc(sortOrderValue) : 0,
    },
    errors: [],
  };
}
