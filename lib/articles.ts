import "server-only";

import { getContentItem, getContentSummaries, type ContentSummary } from "./content";
import type { ContentBlock, Source } from "./content-types";

export interface Article {
  title: string;
  slug: string;
  date: string;
  summary: string;
  content: string;
  blocks: ContentBlock[];
  sources: Source[];
  coverImageUrl?: string;
  coverImageAlt?: string;
  manualGlossaryLinks?: boolean;
}

export interface ArticleSummary {
  title: string;
  slug: string;
  date: string;
  summary: string;
  coverImageUrl?: string;
  coverImageAlt?: string;
}

function asArticle(item: Awaited<ReturnType<typeof getContentItem>> extends infer T ? Exclude<T, null> : never): Article {
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
    manualGlossaryLinks: item.metadata.manualGlossaryLinks === true,
  };
}

function asArticleSummary(item: ContentSummary): ArticleSummary {
  return {
    title: item.title,
    slug: item.slug,
    date: item.publishedAt ?? item.updatedAt.slice(0, 10),
    summary: item.summary,
    ...(item.coverImageUrl ? { coverImageUrl: item.coverImageUrl } : {}),
    ...(item.coverImageAlt ? { coverImageAlt: item.coverImageAlt } : {}),
  };
}

export async function getAllIssues(): Promise<ArticleSummary[]> {
  const items = await getContentSummaries("article");
  return items
    .map(asArticleSummary)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

export async function getIssueBySlug(slug: string): Promise<Article | null> {
  const item = await getContentItem("article", slug);
  return item ? asArticle(item) : null;
}
