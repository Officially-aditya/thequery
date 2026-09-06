import "server-only";

import { getContentItem, getContentSummaries, contentDisplayDate, type ContentSummary } from "./content";
import type { ContentBlock, Source } from "./content-types";

export interface Comparison {
  title: string;
  slug: string;
  date: string;
  summary: string;
  content: string;
  blocks: ContentBlock[];
  sources: Source[];
  coverImageUrl?: string;
  coverImageAlt?: string;
}

export interface ComparisonSummary {
  title: string;
  slug: string;
  date: string;
  summary: string;
  coverImageUrl?: string;
  coverImageAlt?: string;
}

function asComparison(item: Awaited<ReturnType<typeof getContentItem>> extends infer T ? Exclude<T, null> : never): Comparison {
  return {
    title: item.title,
    slug: item.slug,
    date: contentDisplayDate(item.publishedAt, item.updatedAt),
    summary: item.summary,
    content: item.body,
    blocks: item.blocks,
    sources: item.sources,
    ...(item.coverImageUrl ? { coverImageUrl: item.coverImageUrl } : {}),
    ...(item.coverImageAlt ? { coverImageAlt: item.coverImageAlt } : {}),
  };
}

function asComparisonSummary(item: ContentSummary): ComparisonSummary {
  return {
    title: item.title,
    slug: item.slug,
    date: contentDisplayDate(item.publishedAt, item.updatedAt),
    summary: item.summary,
    ...(item.coverImageUrl ? { coverImageUrl: item.coverImageUrl } : {}),
    ...(item.coverImageAlt ? { coverImageAlt: item.coverImageAlt } : {}),
  };
}

export async function getAllComparisons(): Promise<ComparisonSummary[]> {
  const items = await getContentSummaries("comparison");
  return items
    .map(asComparisonSummary)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

export async function getComparisonBySlug(slug: string): Promise<Comparison | null> {
  const item = await getContentItem("comparison", slug);
  return item ? asComparison(item) : null;
}
