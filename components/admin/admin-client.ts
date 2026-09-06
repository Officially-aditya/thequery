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
  coverImageUrl: string;
  coverImageAlt: string;
  status: ContentStatus;
  publishedAt: string;
  sortOrder: number;
}

export interface ContentListItem {
  id: string;
  kind: ContentItem["kind"];
  slug: string;
  parentSlug: string | null;
  path: string;
  title: string;
  summary: string;
  metadata: Record<string, unknown>;
  coverImageUrl: string | null;
  coverImageAlt: string | null;
  status: ContentStatus;
  publishedAt: string | null;
  sortOrder: number;
  createdAt: string;
  updatedAt: string;
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

export function comparisonTemplateBlocks(): ContentBlock[] {
  const id = Date.now();
  const columns = ["Model A", "Model B"];
  return [
    {
      id: `spec-${id}-specifications`,
      type: "spec_table",
      title: "Specifications",
      columns,
      rows: [
        ["Developer", "", ""],
        ["Release date", "", ""],
        ["API model ID", "", ""],
        ["Context window", "", ""],
        ["Max output", "", ""],
        ["Knowledge cutoff", "", ""],
        ["Reasoning / effort", "", ""],
      ],
    },
    {
      id: `spec-${id}-pricing`,
      type: "spec_table",
      title: "Pricing",
      columns,
      rows: [
        ["Input / 1M tokens", "", ""],
        ["Cached input / 1M", "", ""],
        ["Cache write / 1M", "", ""],
        ["Output / 1M tokens", "", ""],
        ["Batch / flex discount", "", ""],
        ["Long-context surcharge", "", ""],
      ],
    },
    {
      id: `spec-${id}-capabilities-access`,
      type: "spec_table",
      title: "Capabilities & access",
      columns,
      rows: [
        ["Text input", "", ""],
        ["Image / vision input", "", ""],
        ["Audio input", "", ""],
        ["Video input", "", ""],
        ["Text output", "", ""],
        ["Audio output", "", ""],
        ["Tool / function calling", "", ""],
        ["Computer use", "", ""],
        ["API access", "", ""],
        ["Product access", "", ""],
        ["Weights / license", "", ""],
      ],
    },
    {
      id: `spec-${id}-coding`,
      type: "spec_table",
      title: "Coding",
      columns,
      rows: [
        ["SWE-bench Verified", "", ""],
        ["SWE-bench Pro", "", ""],
        ["Terminal-Bench", "", ""],
        ["LiveCodeBench", "", ""],
      ],
    },
    {
      id: `spec-${id}-math-reasoning`,
      type: "spec_table",
      title: "Math & reasoning",
      columns,
      rows: [
        ["AIME", "", ""],
        ["HMMT", "", ""],
        ["ARC-AGI", "", ""],
      ],
    },
    {
      id: `spec-${id}-knowledge`,
      type: "spec_table",
      title: "Knowledge",
      columns,
      rows: [
        ["GPQA Diamond", "", ""],
        ["Humanity's Last Exam", "", ""],
        ["MMLU-Pro", "", ""],
      ],
    },
    {
      id: `spec-${id}-agentic`,
      type: "spec_table",
      title: "Agentic & computer use",
      columns,
      rows: [
        ["OSWorld", "", ""],
        ["BrowseComp", "", ""],
        ["MCP / tool-use benchmark", "", ""],
      ],
    },
    {
      id: `markdown-${id}-verdict`,
      type: "markdown",
      content: "## Bottom line\n\n",
    },
  ];
}

export function newContent(metadata: Record<string, unknown> = {}, blocks?: ContentBlock[]): EditableContent {
  return {
    title: "",
    slug: "",
    summary: "",
    body: "",
    blocks: blocks ?? [markdownBlock()],
    sources: [],
    metadata,
    coverImageUrl: "",
    coverImageAlt: "",
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
    coverImageUrl: item.coverImageUrl ?? "",
    coverImageAlt: item.coverImageAlt ?? "",
    status: item.status,
    publishedAt: item.publishedAt ?? "",
    sortOrder: item.sortOrder,
  };
}

export function toContentListItem(item: ContentItem): ContentListItem {
  return {
    id: item.id,
    kind: item.kind,
    slug: item.slug,
    parentSlug: item.parentSlug,
    path: item.path,
    title: item.title,
    summary: item.summary,
    metadata: item.metadata,
    coverImageUrl: item.coverImageUrl,
    coverImageAlt: item.coverImageAlt,
    status: item.status,
    publishedAt: item.publishedAt,
    sortOrder: item.sortOrder,
    createdAt: item.createdAt,
    updatedAt: item.updatedAt,
  };
}

export function publicHref(kind: "article" | "guide" | "comparison" | "glossary" | "book", slug: string): string {
  const prefix = kind === "glossary" ? "/glossary" : `/${kind}s`;
  return `${prefix}/${slug}`;
}
