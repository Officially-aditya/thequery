export type ContentKind = "article" | "guide" | "glossary" | "book" | "chapter";
export type ContentStatus = "draft" | "published";

export interface Source {
  title: string;
  url: string;
}

export interface MarkdownBlock {
  id: string;
  type: "markdown";
  content: string;
}

export interface ComparisonTableBlock {
  id: string;
  type: "comparison_table";
  title?: string;
  caption?: string;
  columns: string[];
  rows: string[][];
  sourceNote?: string;
}

export interface ChartBlock {
  id: string;
  type: "chart";
  title: string;
  description?: string;
  chartType?: "bar" | "line";
  data: Array<Record<string, string | number | null>>;
  sourceNote?: string;
}

export type ContentBlock = MarkdownBlock | ComparisonTableBlock | ChartBlock;

export interface ContentItem<TMetadata extends Record<string, unknown> = Record<string, unknown>> {
  id: string;
  kind: ContentKind;
  slug: string;
  parentSlug: string | null;
  path: string;
  title: string;
  summary: string;
  body: string;
  blocks: ContentBlock[];
  sources: Source[];
  metadata: TMetadata;
  status: ContentStatus;
  publishedAt: string | null;
  sortOrder: number;
  createdAt: string;
  updatedAt: string;
}

export interface ArticleMetadata extends Record<string, unknown> {
  manualGlossaryLinks?: boolean;
}

export interface GlossaryMetadata extends Record<string, unknown> {
  category: string;
  relatedTerms: string[];
  analogy?: string;
  seoDescription?: string;
  seoKeywords?: string[];
}

export interface BookMetadata extends Record<string, unknown> {
  author: string;
  lastModified?: string;
}

export interface ChapterMetadata extends Record<string, unknown> {
  lastModified?: string;
}
