import fs from "fs";
import path from "path";
import { getAllTerms, type GlossaryTerm } from "./glossary";

const dataPath = path.join(process.cwd(), "data", "word-of-the-day.json");

export interface WotdEntry {
  slug: string;
  humor: string;
  examples: string[];
}

export interface WordOfTheDay {
  term: GlossaryTerm;
  humor: string;
  examples: string[];
}

function getAllWotdEntries(): WotdEntry[] {
  const raw = fs.readFileSync(dataPath, "utf-8");
  return JSON.parse(raw);
}

function getDayIndex(): number {
  const now = new Date();
  const start = new Date(now.getFullYear(), 0, 0);
  const diff = now.getTime() - start.getTime();
  const dayOfYear = Math.floor(diff / (1000 * 60 * 60 * 24));
  return dayOfYear;
}

export function getTodaysWord(): WordOfTheDay | null {
  const entries = getAllWotdEntries();
  if (entries.length === 0) return null;

  const index = getDayIndex() % entries.length;
  const entry = entries[index];

  const terms = getAllTerms();
  const term = terms.find((t) => t.slug === entry.slug);
  if (!term) return null;

  return { term, humor: entry.humor, examples: entry.examples };
}

export function getWotdBySlug(slug: string): WordOfTheDay | null {
  const entries = getAllWotdEntries();
  const entry = entries.find((e) => e.slug === slug);
  if (!entry) return null;

  const terms = getAllTerms();
  const term = terms.find((t) => t.slug === entry.slug);
  if (!term) return null;

  return { term, humor: entry.humor, examples: entry.examples };
}
