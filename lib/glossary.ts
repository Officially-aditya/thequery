import fs from "fs";
import path from "path";

const dataPath = path.join(process.cwd(), "data", "glossary.json");
const additionsPath = path.join(process.cwd(), "data", "glossary-additions.json");

export interface GlossaryTerm {
  name: string;
  slug: string;
  shortDef: string;
  fullDef: string;
  category: string;
  relatedTerms: string[];
  analogy?: string;
  references?: { title: string; url: string }[];
  seoDescription?: string;
  seoKeywords?: string[];
  lastUpdated: string;
}

function getAdditions(): GlossaryTerm[] {
  if (!fs.existsSync(additionsPath)) return [];
  const raw = fs.readFileSync(additionsPath, "utf-8");
  return JSON.parse(raw);
}

export function getAllTerms(): GlossaryTerm[] {
  const raw = fs.readFileSync(dataPath, "utf-8");
  const terms: GlossaryTerm[] = JSON.parse(raw);
  const additions = getAdditions();
  const additionsBySlug = new Map(additions.map((term) => [term.slug, term]));

  // Keep the main glossary as the source of truth while allowing substantive
  // additions to live separately. Additions override a duplicate slug so a
  // term can be expanded without creating duplicate glossary pages.
  const merged = terms.map((term) => additionsBySlug.get(term.slug) ?? term);
  const existingSlugs = new Set(terms.map((term) => term.slug));
  return [...merged, ...additions.filter((term) => !existingSlugs.has(term.slug))];
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
  const additionSlugs = new Set(getAdditions().map((term) => term.slug));
  const baseTerms = terms.filter((term) => !additionSlugs.has(term.slug));
  fs.writeFileSync(dataPath, JSON.stringify(baseTerms, null, 2), "utf-8");
}
