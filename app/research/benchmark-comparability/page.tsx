import { readFile } from "node:fs/promises";
import { join } from "node:path";
import Link from "next/link";
import type { Metadata } from "next";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import { SITE_URL, authorJsonLd, ORGANIZATION_ID, createOpenGraphMetadata } from "@/lib/site";

export const revalidate = false;

const title = "AI Benchmark Scores Change. The Model Often Doesn't.";
const description = "An audit of 837 reported benchmark observations, with source checks showing why tools, harnesses, and graders matter to AI model comparisons.";
const url = `${SITE_URL}/research/benchmark-comparability`;

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: url },
  openGraph: createOpenGraphMetadata({ title, description, url, type: "article" }),
};

export default async function BenchmarkComparabilityPage() {
  const content = await readFile(join(process.cwd(), "app/research/benchmark-comparability/report.md"), "utf8");
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "TechArticle",
    headline: title,
    description,
    datePublished: "2026-10-04",
    dateModified: "2026-10-04",
    url,
    author: authorJsonLd,
    publisher: { "@type": "Organization", "@id": ORGANIZATION_ID, name: "TheQuery" },
    inLanguage: "en",
  };

  return <article className="max-w-[720px] mx-auto px-4 py-12">
    <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
    <Link href="/research" className="text-sm text-text-muted hover:text-text-secondary mb-6 inline-block">&larr; Research and data</Link>
    <MarkdownRenderer content={content} />
  </article>;
}
