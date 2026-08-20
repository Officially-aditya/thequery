import "server-only";

import { getContentItem, getContentItems } from "./content";
import type { Source } from "./content-types";

export interface GlossaryTerm {
  name: string;
  slug: string;
  shortDef: string;
  fullDef: string;
  category: string;
  relatedTerms: string[];
  coverImageUrl?: string;
  coverImageAlt?: string;
  analogy?: string;
  references?: Source[];
  seoDescription?: string;
  seoKeywords?: string[];
  lastUpdated: string;
}

function textList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function asTerm(item: Awaited<ReturnType<typeof getContentItem>> extends infer T ? Exclude<T, null> : never): GlossaryTerm {
  const metadata = item.metadata;
  return {
    name: item.title,
    slug: item.slug,
    shortDef: item.summary,
    fullDef: item.body,
    category: typeof metadata.category === "string" ? metadata.category : "Foundations",
    relatedTerms: textList(metadata.relatedTerms),
    ...(item.coverImageUrl ? { coverImageUrl: item.coverImageUrl } : {}),
    ...(item.coverImageAlt ? { coverImageAlt: item.coverImageAlt } : {}),
    ...(typeof metadata.analogy === "string" ? { analogy: metadata.analogy } : {}),
    ...(item.sources.length ? { references: item.sources } : {}),
    ...(typeof metadata.seoDescription === "string" ? { seoDescription: metadata.seoDescription } : {}),
    ...(textList(metadata.seoKeywords).length ? { seoKeywords: textList(metadata.seoKeywords) } : {}),
    lastUpdated: item.publishedAt ?? item.updatedAt.slice(0, 10),
  };
}

export async function getAllTerms(): Promise<GlossaryTerm[]> {
  const items = await getContentItems("glossary");
  return items.map(asTerm).sort((a, b) => a.name.localeCompare(b.name));
}

export async function getTermBySlug(slug: string): Promise<GlossaryTerm | null> {
  const item = await getContentItem("glossary", slug);
  return item ? asTerm(item) : null;
}

export async function getTermsByCategory(): Promise<Record<string, GlossaryTerm[]>> {
  const terms = await getAllTerms();
  return terms.reduce<Record<string, GlossaryTerm[]>>((grouped, term) => {
    (grouped[term.category] ??= []).push(term);
    return grouped;
  }, {});
}
