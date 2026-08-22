import "server-only";

import {
  getContentIndex,
  getContentItem,
  getContentItems,
  getContentSummaries,
  type ContentSummary,
} from "./content";
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

export interface GlossaryTermSummary {
  name: string;
  slug: string;
  shortDef: string;
  category: string;
  coverImageUrl?: string;
  coverImageAlt?: string;
  lastUpdated: string;
}

export interface GlossaryIndexItem {
  name: string;
  slug: string;
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

function asTermSummary(item: ContentSummary): GlossaryTermSummary {
  return {
    name: item.title,
    slug: item.slug,
    shortDef: item.summary,
    category: typeof item.metadata.category === "string" ? item.metadata.category : "Foundations",
    ...(item.coverImageUrl ? { coverImageUrl: item.coverImageUrl } : {}),
    ...(item.coverImageAlt ? { coverImageAlt: item.coverImageAlt } : {}),
    lastUpdated: item.publishedAt ?? item.updatedAt.slice(0, 10),
  };
}

export async function getAllTerms(): Promise<GlossaryTerm[]> {
  const items = await getContentItems("glossary");
  return items.map(asTerm).sort((a, b) => a.name.localeCompare(b.name));
}

export async function getAllTermSummaries(): Promise<GlossaryTermSummary[]> {
  const items = await getContentSummaries("glossary");
  return items.map(asTermSummary).sort((a, b) => a.name.localeCompare(b.name));
}

export async function getGlossaryIndex(): Promise<GlossaryIndexItem[]> {
  const items = await getContentIndex("glossary");
  return items.map((item) => ({ name: item.title, slug: item.slug }));
}

export async function getGlossaryCount(): Promise<number> {
  return (await getGlossaryIndex()).length;
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
