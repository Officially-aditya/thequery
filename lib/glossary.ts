import fs from "fs";
import path from "path";

const dataPath = path.join(process.cwd(), "data", "glossary.json");
const additionsPath = path.join(process.cwd(), "data", "glossary-additions.json");
const deepDivesPath = path.join(process.cwd(), "data", "glossary-deep-dives.json");

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

function readTerms(filePath: string): GlossaryTerm[] {
  if (!fs.existsSync(filePath)) return [];
  const raw = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(raw);
}

function getAdditions(): GlossaryTerm[] {
  return readTerms(additionsPath);
}

function getDeepDives(): GlossaryTerm[] {
  return readTerms(deepDivesPath);
}

export function getAllTerms(): GlossaryTerm[] {
  const terms: GlossaryTerm[] = readTerms(dataPath);
  const additions = getAdditions();
  const deepDives = getDeepDives();

  const overrides = new Map<string, GlossaryTerm>();
  for (const term of additions) overrides.set(term.slug, term);
  for (const term of deepDives) overrides.set(term.slug, term);

  const merged = terms.map((term) => overrides.get(term.slug) ?? term);
  const existingSlugs = new Set(terms.map((term) => term.slug));
  const added = additions.filter((term) => !existingSlugs.has(term.slug));
  const existingAfterAdditions = new Set([...existingSlugs, ...added.map((term) => term.slug)]);
  const deepDiveOnly = deepDives.filter((term) => !existingAfterAdditions.has(term.slug));

  return [...merged, ...added, ...deepDiveOnly];
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
  const managedSlugs = new Set([
    ...getAdditions().map((term) => term.slug),
    ...getDeepDives().map((term) => term.slug),
  ]);
  const baseTerms = terms.filter((term) => !managedSlugs.has(term.slug));
  fs.writeFileSync(dataPath, JSON.stringify(baseTerms, null, 2), "utf-8");
}
