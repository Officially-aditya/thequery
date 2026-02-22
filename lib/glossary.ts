import fs from "fs";
import path from "path";

const dataPath = path.join(process.cwd(), "data", "glossary.json");

export interface GlossaryTerm {
  name: string;
  slug: string;
  shortDef: string;
  fullDef: string;
  category: string;
  relatedTerms: string[];
  references?: { title: string; url: string }[];
  lastUpdated: string;
}

export function getAllTerms(): GlossaryTerm[] {
  const raw = fs.readFileSync(dataPath, "utf-8");
  return JSON.parse(raw);
}

export function getTermBySlug(slug: string): GlossaryTerm | null {
  const terms = getAllTerms();
  return terms.find((t) => t.slug === slug) ?? null;
}

export function getTermsByCategory(): Record<string, GlossaryTerm[]> {
  const terms = getAllTerms();
  const grouped: Record<string, GlossaryTerm[]> = {};
  for (const term of terms) {
    if (!grouped[term.category]) grouped[term.category] = [];
    grouped[term.category].push(term);
  }
  return grouped;
}

export function saveAllTerms(terms: GlossaryTerm[]): void {
  fs.writeFileSync(dataPath, JSON.stringify(terms, null, 2), "utf-8");
}
