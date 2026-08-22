import "server-only";

import { getContentItem, getContentSummaries, type ContentSummary } from "./content";
import type { ContentBlock, Source } from "./content-types";

export interface Guide {
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

export interface GuideSummary {
  title: string;
  slug: string;
  date: string;
  summary: string;
  coverImageUrl?: string;
  coverImageAlt?: string;
}

function asGuide(item: Awaited<ReturnType<typeof getContentItem>> extends infer T ? Exclude<T, null> : never): Guide {
  return {
    title: item.title,
    slug: item.slug,
    date: item.publishedAt ?? item.updatedAt.slice(0, 10),
    summary: item.summary,
    content: item.body,
    blocks: item.blocks,
    sources: item.sources,
    ...(item.coverImageUrl ? { coverImageUrl: item.coverImageUrl } : {}),
    ...(item.coverImageAlt ? { coverImageAlt: item.coverImageAlt } : {}),
  };
}

function asGuideSummary(item: ContentSummary): GuideSummary {
  return {
    title: item.title,
    slug: item.slug,
    date: item.publishedAt ?? item.updatedAt.slice(0, 10),
    summary: item.summary,
    ...(item.coverImageUrl ? { coverImageUrl: item.coverImageUrl } : {}),
    ...(item.coverImageAlt ? { coverImageAlt: item.coverImageAlt } : {}),
  };
}

export async function getAllGuides(): Promise<GuideSummary[]> {
  const items = await getContentSummaries("guide");
  return items
    .map(asGuideSummary)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

export async function getGuideBySlug(slug: string): Promise<Guide | null> {
  const item = await getContentItem("guide", slug);
  return item ? asGuide(item) : null;
}
