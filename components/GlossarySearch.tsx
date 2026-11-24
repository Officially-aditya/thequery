"use client";

import { memo, useDeferredValue, useMemo, useState } from "react";
import Link from "next/link";

interface Term {
  name: string;
  slug: string;
  shortDef: string;
  category: string;
}

interface GlossaryFilters {
  query: string;
  category: string;
  letter: string | null;
}

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

function normalize(value: string) {
  return value
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

export default function GlossarySearch({ terms }: { terms: Term[] }) {
  const [filters, setFilters] = useState<GlossaryFilters>({
    query: "",
    category: "All",
    letter: "A",
  });
  const { query, category, letter } = filters;

  const { categories, termsByLetter } = useMemo(() => {
    const categoryNames = new Set<string>();
    const termsByLetter = new Map<string, Term[]>();

    for (const term of terms) {
      categoryNames.add(term.category);
      const initial = term.name.trim().toUpperCase().charAt(0);
      const group = termsByLetter.get(initial);
      if (group) group.push(term);
      else termsByLetter.set(initial, [term]);
    }

    return {
      categories: ["All", ...Array.from(categoryNames).sort((a, b) => a.localeCompare(b))],
      termsByLetter,
    };
  }, [terms]);

  const deferredFilters = useDeferredValue(filters);
  const { category: deferredCategory, letter: deferredLetter, query: deferredQuery } = deferredFilters;
  const normalizedQuery = useMemo(() => normalize(deferredQuery), [deferredQuery]);
  const hasQuery = normalizedQuery.length > 0;

  // Build search keys on the first nonempty query and reuse them while typing.
  const searchNames = useMemo(
    () => hasQuery ? new Map(terms.map((term) => [term, normalize(term.name)])) : null,
    [hasQuery, terms],
  );

  const filtered = useMemo(() => {
    const candidates = deferredLetter ? termsByLetter.get(deferredLetter) ?? [] : terms;
    if (deferredCategory === "All" && !searchNames) return candidates;

    return candidates.filter((term) =>
      (deferredCategory === "All" || term.category === deferredCategory)
      && (!searchNames || searchNames.get(term)?.includes(normalizedQuery))
    );
  }, [deferredCategory, deferredLetter, normalizedQuery, searchNames, terms, termsByLetter]);

  return (
    <div>
      <input
        type="search"
        placeholder="Search terms..."
        value={query}
        onChange={(e) => {
          const query = e.target.value;
          setFilters((current) => ({ ...current, query, letter: null }));
        }}
        className="w-full px-4 py-2 mb-4 border border-border rounded-lg bg-bg-primary text-text-primary placeholder-text-muted focus:outline-none focus:border-accent text-sm"
        aria-label="Search glossary terms"
      />

      <div className="flex flex-wrap gap-2 mb-4">
        {categories.map((cat) => (
          <button
            key={cat}
            type="button"
            onClick={() => setFilters((current) =>
              current.category === cat ? current : { ...current, category: cat }
            )}
            className={`text-xs px-3 py-1 rounded-full border transition-colors ${
              category === cat
                ? "border-accent text-accent bg-accent/10"
                : "border-border text-text-secondary hover:border-text-muted"
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap gap-1 mb-6">
        {ALPHABET.map((l) => (
          <button
            key={l}
            type="button"
            aria-pressed={letter === l}
            onClick={() => setFilters((current) => ({
              ...current,
              letter: current.letter === l ? null : l,
            }))}
            className={`text-xs w-7 h-7 rounded flex items-center justify-center transition-colors ${
              letter === l
                ? "bg-accent text-white"
                : "text-text-muted hover:text-text-secondary hover:bg-bg-secondary"
            }`}
          >
            {l}
          </button>
        ))}
      </div>

      <div aria-busy={filters !== deferredFilters}>
        <GlossaryResults terms={filtered} />
      </div>
    </div>
  );
}

const GlossaryResults = memo(function GlossaryResults({ terms }: { terms: Term[] }) {
  return (
    <>
      <div className="grid gap-3 sm:grid-cols-2">
        {terms.map((term) => (
          <Link
            key={term.slug}
            href={`/glossary/${term.slug}`}
            prefetch={false}
            className="block p-4 border border-border rounded-lg hover:border-accent transition-colors group"
          >
            <div className="flex items-start justify-between mb-1">
              <h3 className="text-sm font-semibold text-text-primary group-hover:text-accent transition-colors">
                {term.name}
              </h3>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-tag-bg text-tag-text shrink-0 ml-2">
                {term.category}
              </span>
            </div>
            <p className="text-xs text-text-secondary leading-relaxed">
              {term.shortDef}
            </p>
          </Link>
        ))}
      </div>

      {terms.length === 0 && (
        <p className="text-sm text-text-muted text-center py-8">
          No terms found matching your search.
        </p>
      )}
    </>
  );
});
