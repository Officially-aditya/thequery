import Link from "next/link";
import { getAllTerms, getTermBySlug } from "@/lib/glossary";
import { notFound } from "next/navigation";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ term: string }>;
}

export async function generateStaticParams() {
  return getAllTerms().map((t) => ({ term: t.slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { term: slug } = await params;
  const term = getTermBySlug(slug);
  if (!term) return {};
  return {
    title: `${term.name} - AI Glossary`,
    description: term.shortDef,
    openGraph: { title: `${term.name} - AI Glossary`, description: term.shortDef },
  };
}

export default async function TermPage({ params }: Props) {
  const { term: slug } = await params;
  const term = getTermBySlug(slug);
  if (!term) notFound();

  const allTerms = getAllTerms();
  const related = term.relatedTerms
    .map((s) => allTerms.find((t) => t.slug === s))
    .filter(Boolean);

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "DefinedTerm",
    name: term.name,
    description: term.shortDef,
    inDefinedTermSet: {
      "@type": "DefinedTermSet",
      name: "TheQuery AI Glossary",
    },
  };

  return (
    <div className="max-w-[960px] mx-auto px-4 py-12">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      <Link href="/glossary" className="text-sm text-text-muted hover:text-text-secondary transition-colors mb-6 inline-block">
        &larr; Glossary
      </Link>

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

      {term.references && term.references.length > 0 && (
        <div className="border-t border-border pt-6 mb-6">
          <h2 className="font-serif text-sm font-semibold text-text-muted mb-3">References &amp; Resources</h2>
          <ul className="space-y-2">
            {term.references.map((ref, i) => (
              <li key={i}>
                <a
                  href={ref.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-accent hover:text-accent-hover transition-colors inline-flex items-center gap-1"
                >
                  {ref.title}
                  <svg className="w-3 h-3 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" /></svg>
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}

      {related.length > 0 && (
        <div className="border-t border-border pt-6">
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

      <p className="text-xs text-text-muted mt-6">
        Last updated: {new Date(term.lastUpdated).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
      </p>
    </div>
  );
}
