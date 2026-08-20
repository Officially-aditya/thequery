import type { ContentBlock, ContentItem, ContentStatus, Source } from "@/lib/content-types";

export interface EditableContent {
  id?: string;
  title: string;
  slug: string;
  parentSlug?: string;
  summary: string;
  body: string;
  blocks: ContentBlock[];
  sources: Source[];
  metadata: Record<string, unknown>;
  status: ContentStatus;
  publishedAt: string;
  sortOrder: number;
}

export class ApiError extends Error {
  constructor(message: string, public status: number, public details: string[] = []) {
    super(message);
  }
}

export async function apiRequest<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  const payload: unknown = await response.json().catch(() => ({}));
  if (!response.ok) {
    const record = payload && typeof payload === "object" ? payload as Record<string, unknown> : {};
    const details = Array.isArray(record.errors) ? record.errors.filter((error): error is string => typeof error === "string") : [];
    throw new ApiError(typeof record.error === "string" ? record.error : details[0] ?? "Request failed.", response.status, details);
  }
  return payload as T;
}

export function today(): string {
  return new Date().toISOString().slice(0, 10);
}

export function markdownBlock(content = ""): ContentBlock {
  return { id: `markdown-${Date.now()}`, type: "markdown", content };
}

export function newContent(metadata: Record<string, unknown> = {}): EditableContent {
  return {
    title: "",
    slug: "",
    summary: "",
    body: "",
    blocks: [markdownBlock()],
    sources: [],
    metadata,
    status: "draft",
    publishedAt: today(),
    sortOrder: 0,
  };
}

export function toEditableContent(item: ContentItem): EditableContent {
  return {
    id: item.id,
    title: item.title,
    slug: item.slug,
    ...(item.parentSlug ? { parentSlug: item.parentSlug } : {}),
    summary: item.summary,
    body: item.body,
    blocks: item.blocks,
    sources: item.sources,
    metadata: item.metadata,
    status: item.status,
    publishedAt: item.publishedAt ?? "",
    sortOrder: item.sortOrder,
  };
}

export function publicHref(kind: "article" | "guide" | "glossary" | "book", slug: string): string {
  const prefix = kind === "glossary" ? "/glossary" : `/${kind}s`;
  return `${prefix}/${slug}`;
}
