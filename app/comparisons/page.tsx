import Link from "next/link";
import { getAllComparisons } from "@/lib/comparisons";
import { createOpenGraphMetadata, SITE_URL } from "@/lib/site";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Comparisons",
  description: "Side-by-side comparisons of AI models, tools, and approaches - so you can choose with evidence, not hype.",
  openGraph: createOpenGraphMetadata({
    title: "Comparisons - TheQuery",
    description: "Side-by-side comparisons of AI models, tools, and approaches - so you can choose with evidence, not hype.",
    url: `${SITE_URL}/comparisons`,
  }),
};

export const revalidate = false;

export default async function ComparisonsPage() {
  const comparisons = await getAllComparisons();

  return (
    <div className="max-w-[960px] mx-auto px-4 py-12">
      <h1 className="font-serif text-3xl font-bold text-text-primary mb-2">Comparisons</h1>
      <p className="text-text-secondary mb-8">
        Side-by-side comparisons of AI models, tools, and approaches - evidence over hype.
      </p>

      {comparisons.length === 0 ? (
        <p className="text-sm text-text-muted text-center py-12">No comparisons yet. Check back soon!</p>
      ) : (
        <div className="space-y-4">
          {comparisons.map((comparison) => (
            <Link
              key={comparison.slug}
              href={`/comparisons/${comparison.slug}`}
              className="block p-5 border border-border rounded-lg hover:border-accent transition-colors group"
            >
              <h2 className="font-serif text-lg font-semibold text-text-primary group-hover:text-accent transition-colors mb-1">
                {comparison.title}
              </h2>
              <p className="text-xs text-text-muted mb-2">
                {new Date(comparison.date).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
              </p>
              <p className="text-sm text-text-secondary leading-relaxed">
                {comparison.summary}
              </p>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
