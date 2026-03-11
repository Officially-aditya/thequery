import Link from "next/link";
import { getTodaysWord } from "@/lib/ai-word-of-the-day";
import { getAllTerms } from "@/lib/glossary";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import type { Metadata } from "next";

export const revalidate = 3600;

export async function generateMetadata(): Promise<Metadata> {
  const wotd = getTodaysWord();
  if (!wotd) return { title: "AI Word of the Day" };
  return {
    title: `${wotd.term.name} — AI Word of the Day`,
    description: wotd.term.shortDef,
    openGraph: {
      title: `${wotd.term.name} — AI Word of the Day`,
      description: wotd.term.shortDef,
    },
  };
}

export default function WordOfTheDayPage() {
  const wotd = getTodaysWord();
  if (!wotd) return <div className="max-w-[960px] mx-auto px-4 py-12">No AI word of the day available.</div>;

  const { term, humor, examples } = wotd;
  const allTerms = getAllTerms();
  const related = term.relatedTerms
    .map((s) => allTerms.find((t) => t.slug === s))
    .filter(Boolean);

  return (
    <div className="max-w-[960px] mx-auto px-4 py-12">
      <Link href="/" className="text-sm text-text-muted hover:text-text-secondary transition-colors mb-6 inline-block">
        &larr; Home
      </Link>

      <div className="mb-2">
        <span className="text-xs font-medium uppercase tracking-wider text-text-muted">AI Word of the Day</span>
      </div>

      <div className="flex items-start justify-between mb-4">
        <h1 className="font-serif text-3xl font-bold text-text-primary">
          {term.name}
        </h1>
        <span className="text-xs px-3 py-1 rounded-full bg-tag-bg text-tag-text mt-2">
          {term.category}
        </span>
      </div>

      <p className="text-lg text-text-secondary mb-6 leading-relaxed">
        {term.shortDef}
      </p>

      <div className="mb-8">
        <MarkdownRenderer content={term.fullDef} disableMath />
      </div>

      {related.length > 0 && (
        <div className="border-t border-border pt-6 mb-8">
          <h2 className="font-serif text-sm font-semibold text-text-muted mb-3">Related Terms</h2>
          <div className="flex flex-wrap gap-2">
            {related.map((r) => r && (
              <Link
                key={r.slug}
                href={`/glossary/${r.slug}`}
                className="text-sm px-3 py-1 border border-border rounded-full text-text-secondary hover:border-accent hover:text-accent transition-colors"
              >
                {r.name}
              </Link>
            ))}
          </div>
        </div>
      )}

      {examples.length > 0 && (
        <div className="border-t border-border pt-6 mb-8">
          <h2 className="font-serif text-sm font-semibold text-text-muted mb-3">Usage in Practice</h2>
          <ul className="space-y-3">
            {examples.map((ex, i) => (
              <li key={i} className="text-text-secondary leading-relaxed pl-4 border-l-2 border-border">
                {ex}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="border-t border-border pt-6">
        <h2 className="font-serif text-sm font-semibold text-text-muted mb-3">The Honest Take</h2>
        <p className="text-text-secondary leading-relaxed italic">
          {humor}
        </p>
      </div>

      <div className="mt-8 text-center">
        <Link
          href={`/glossary/${term.slug}`}
          className="text-sm text-accent hover:text-accent-hover transition-colors"
        >
          View full glossary entry &rarr;
        </Link>
      </div>
    </div>
  );
}
