import "server-only";

import { getContentItem, getContentItems } from "./content";
import type { ContentBlock, Source } from "./content-types";

export interface Guide {
  title: string;
  slug: string;
  date: string;
  summary: string;
  content: string;
  blocks: ContentBlock[];
  sources: Source[];
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
  };
}

export async function getAllGuides(): Promise<Guide[]> {
  const items = await getContentItems("guide");
  return items
    .map(asGuide)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

export async function getGuideBySlug(slug: string): Promise<Guide | null> {
  const item = await getContentItem("guide", slug);
  return item ? asGuide(item) : null;
}
