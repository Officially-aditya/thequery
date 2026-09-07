import "server-only";

import { unstable_cache } from "next/cache";
import { getContentItem, getContentSummaries, contentDisplayDate, type ContentSummary } from "./content";
import type { ContentBlock, Source, SpecTableBlock } from "./content-types";
import { getSql } from "./db";

export interface Comparison {
  title: string;
  slug: string;
  date: string;
  summary: string;
  content: string;
  blocks: ContentBlock[];
  sources: Source[];
  modelA?: string;
  modelB?: string;
  coverImageUrl?: string;
  coverImageAlt?: string;
}

export interface ComparisonSummary {
  title: string;
  slug: string;
  date: string;
  summary: string;
  modelA?: string;
  modelB?: string;
  coverImageUrl?: string;
  coverImageAlt?: string;
}

export interface ComparisonPair {
  modelA: string;
  modelB: string;
  slug: string;
}

function metadataRecord(value: unknown): Record<string, unknown> {
  if (typeof value === "string") {
    try {
      return metadataRecord(JSON.parse(value));
    } catch {
      return {};
    }
  }
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function metadataModelSlug(metadata: Record<string, unknown>, key: "modelA" | "modelB"): string | undefined {
  const value = metadata[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function withFlagshipCapabilities(title: string, blocks: ContentBlock[]): ContentBlock[] {
  const normalizedTitle = title.toLowerCase();
  if (!normalizedTitle.includes("fable 5.1") || !normalizedTitle.includes("astra")) return blocks;

  const alreadyHasCapabilities = blocks.some(
    (block) => block.type === "spec_table" && ["capabilities & access", "capabilities and access"].includes((block.title ?? "").toLowerCase()),
  );
  if (alreadyHasCapabilities) return blocks;

  const pricingIndex = blocks.findIndex(
    (block) => block.type === "spec_table" && (block.title ?? "").toLowerCase() === "pricing",
  );
  if (pricingIndex < 0) return blocks;

  const firstSpec = blocks.find((block): block is SpecTableBlock => block.type === "spec_table");
  const columns = firstSpec?.columns.length === 2 ? firstSpec.columns : ["Claude Fable 5.1", "GPT-6 Astra"];
  const capabilities: SpecTableBlock = {
    id: "spec-capabilities-access-fable-astra",
    type: "spec_table",
    title: "Capabilities & access",
    columns,
    rows: [
      ["Text input", "**Yes**", "**Yes**"],
      ["Image / vision input", "**Yes**", "**Yes**"],
      ["Audio input", "No native audio input", "No"],
      ["Video input", "No", "No"],
      ["Text output", "**Yes**", "**Yes**"],
      ["Audio output", "No native audio output", "No"],
      ["Tool / function calling", "**Yes**", "**Yes**"],
      ["Computer use", "**Yes**", "**Yes**"],
      ["API access", "Claude API; AWS Bedrock; Google Cloud; Microsoft Foundry", "OpenAI API; Microsoft Azure; AWS Bedrock"],
      ["Product access", "Claude Pro, Max, Team, Enterprise; Claude Code", "ChatGPT Plus, Pro, Business, Enterprise"],
      ["Weights / license", "Proprietary", "Proprietary"],
    ],
  };

  return [
    ...blocks.slice(0, pricingIndex + 1),
    capabilities,
    ...blocks.slice(pricingIndex + 1),
  ];
}

function asComparison(item: Awaited<ReturnType<typeof getContentItem>> extends infer T ? Exclude<T, null> : never): Comparison {
  const modelA = metadataModelSlug(item.metadata, "modelA");
  const modelB = metadataModelSlug(item.metadata, "modelB");
  return {
    title: item.title,
    slug: item.slug,
    date: contentDisplayDate(item.publishedAt, item.updatedAt),
    summary: item.summary,
    content: item.body,
    blocks: withFlagshipCapabilities(item.title, item.blocks),
    sources: item.sources,
    ...(modelA ? { modelA } : {}),
    ...(modelB ? { modelB } : {}),
    ...(item.coverImageUrl ? { coverImageUrl: item.coverImageUrl } : {}),
    ...(item.coverImageAlt ? { coverImageAlt: item.coverImageAlt } : {}),
  };
}

function asComparisonSummary(item: ContentSummary): ComparisonSummary {
  const modelA = metadataModelSlug(item.metadata, "modelA");
  const modelB = metadataModelSlug(item.metadata, "modelB");
  return {
    title: item.title,
    slug: item.slug,
    date: contentDisplayDate(item.publishedAt, item.updatedAt),
    summary: item.summary,
    ...(modelA ? { modelA } : {}),
    ...(modelB ? { modelB } : {}),
    ...(item.coverImageUrl ? { coverImageUrl: item.coverImageUrl } : {}),
    ...(item.coverImageAlt ? { coverImageAlt: item.coverImageAlt } : {}),
  };
}

async function queryComparisonPairs(): Promise<ComparisonPair[]> {
  const sql = getSql();
  const rows = await sql.query(
    `SELECT slug, metadata
     FROM content_items
     WHERE kind = 'comparison' AND status = 'published'`,
  ) as Array<{ slug: string; metadata: unknown }>;

  return rows.flatMap((row) => {
    const metadata = metadataRecord(row.metadata);
    const modelA = metadataModelSlug(metadata, "modelA");
    const modelB = metadataModelSlug(metadata, "modelB");
    return modelA && modelB ? [{ modelA, modelB, slug: row.slug }] : [];
  });
}

export async function getComparisonPairs(): Promise<ComparisonPair[]> {
  return unstable_cache(
    queryComparisonPairs,
    ["comparison-pairs-v1"],
    { revalidate: 300, tags: ["content:comparison"] },
  )();
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
