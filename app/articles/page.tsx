import { Suspense } from "react";
import { getAllIssues } from "@/lib/articles";
import ArticlePagination from "@/components/articles/ArticlePagination";
import { createOpenGraphMetadata } from "@/lib/site";
import type { Metadata } from "next";

const SITE_URL = "https://www.thequery.in";

export const revalidate = false;

export const metadata: Metadata = {
  title: "Articles",
  description: "A curated weekly summary of the most important AI developments, research, and news.",
  alternates: { canonical: `${SITE_URL}/articles` },
  openGraph: createOpenGraphMetadata({
    title: "Articles - TheQuery",
    description: "A weekly roundup of what actually matters in AI - no hype, just signal.",
    url: `${SITE_URL}/articles`,
  }),
};

export default async function ArticlesPage() {
  const issues = await getAllIssues();

  return (
    <div className="max-w-[960px] mx-auto px-4 py-12">
      <h1 className="font-serif text-3xl font-bold text-text-primary mb-2">Articles</h1>
      <p className="text-text-secondary mb-8">
        A curated summary of the most important AI developments each week.
      </p>
      <Suspense fallback={<p className="text-sm text-text-muted text-center py-12">Loading articles…</p>}>
        <ArticlePagination issues={issues} />
      </Suspense>
    </div>
  );
}
