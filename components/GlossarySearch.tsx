"use client";

import { useMemo, useState } from "react";
import Link from "next/link";

interface Term {
  name: string;
  slug: string;
  shortDef: string;
  category: string;
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
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("All");
  const [letter, setLetter] = useState<string | null>(null);

  const categories = useMemo(
    () => ["All", ...Array.from(new Set(terms.map((term) => term.category))).sort((a, b) => a.localeCompare(b))],
    [terms]
  );

  const filtered = useMemo(() => {
    const normalizedQuery = normalize(query);

    return terms.filter((term) => {
      const matchesQuery = !normalizedQuery || normalize(term.name).includes(normalizedQuery);
      const matchesCategory = category === "All" || term.category === category;
      const matchesLetter = !letter || term.name.trim().toUpperCase().startsWith(letter);

      return matchesQuery && matchesCategory && matchesLetter;
    });
  }, [terms, query, category, letter]);

  return (
    <div>
      <input
        type="search"
        placeholder="Search terms..."
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setLetter(null);
        }}
        className="w-full px-4 py-2 mb-4 border border-border rounded-lg bg-bg-primary text-text-primary placeholder-text-muted focus:outline-none focus:border-accent text-sm"
        aria-label="Search glossary terms"
      />

      <div className="flex flex-wrap gap-2 mb-4">
        {categories.map((cat) => (
          <button
            key={cat}
            type="button"
            onClick={() => setCategory(cat)}
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
            onClick={() => setLetter(letter === l ? null : l)}
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

      <div className="grid gap-3 sm:grid-cols-2">
        {filtered.map((term) => (
          <Link
            key={term.slug}
            href={`/glossary/${term.slug}`}
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

      {filtered.length === 0 && (
        <p className="text-sm text-text-muted text-center py-8">
          No terms found matching your search.
        </p>
      )}
    </div>
  );
}
