import Link from "next/link";
import { getAllIssues } from "@/lib/digest";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Weekly AI Digest",
  description: "A curated weekly summary of the most important AI developments, research, and news.",
  openGraph: {
    title: "Weekly AI Digest - TheQuery",
    description: "A weekly roundup of what actually matters in AI - no hype, just signal.",
  },
};

export default function DigestPage() {
  const issues = getAllIssues();

  return (
    <div className="max-w-[960px] mx-auto px-4 py-12">
      <h1 className="font-serif text-3xl font-bold text-text-primary mb-2">Weekly Digest</h1>
      <p className="text-text-secondary mb-8">
        A curated summary of the most important AI developments each week.
      </p>

      {issues.length === 0 ? (
        <p className="text-sm text-text-muted text-center py-12">No issues yet. Check back soon!</p>
      ) : (
        <div className="space-y-4">
          {issues.map((issue) => (
            <Link
              key={issue.slug}
              href={`/digest/${issue.slug}`}
              className="block p-5 border border-border rounded-lg hover:border-accent transition-colors group"
            >
              <h2 className="font-serif text-lg font-semibold text-text-primary group-hover:text-accent transition-colors mb-2">
                {issue.title}
              </h2>
              <p className="text-sm text-text-secondary leading-relaxed">
                {issue.summary}
              </p>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
