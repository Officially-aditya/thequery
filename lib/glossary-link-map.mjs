import { getGlossaryMentions } from "./glossary-linking.mjs";

/** @typedef {{name: string, slug: string}} Term */
/** @typedef {{slug: string, title: string}} Book */
/** @typedef {{slug: string, title: string, body: string, parentSlug: string | null}} Chapter */
/** @typedef {{slug: string, title: string, blocks: {type: string, content?: string}[]}} Guide */
/** @typedef {{kind: "chapter" | "guide", href: string, label: string, context: string | null, mentions: number}} LearnMoreLink */

const MAX_LINKS_PER_TERM = 4;

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// These chapters enumerate or reference terms rather than teach them, so they
// accumulate mentions without being the right link for a term page.
const NAVIGATIONAL_TITLE = /\b(terminolog\w*|glossar\w*|assessments?|quizzes|resources?|appendix|overview|introduction|conclusion|capstone|index|projects?)\b/i;

function titleScore(target, name) {
  let score = 0;
  if (NAVIGATIONAL_TITLE.test(target.label)) score -= 1;
  // A chapter or guide named after the term is the most direct pointer to it.
  if (new RegExp(`\\b${escapeRegExp(name)}\\b`, "i").test(target.label)) score += 2;
  return score;
}

function compareTargets(a, b, termName) {
  const scoreA = titleScore(a, termName);
  const scoreB = titleScore(b, termName);
  if (scoreA !== scoreB) return scoreB - scoreA;
  if (a.mentions !== b.mentions) return b.mentions - a.mentions;
  // Chapter prose is the canonical teaching surface for a term, so chapters
  // outrank guides once the above signals are equal.
  if ((a.kind === "chapter") !== (b.kind === "chapter")) return a.kind === "chapter" ? -1 : 1;
  return a.label.localeCompare(b.label);
}

/** @param {{terms: Term[], books: Book[], chapters: Chapter[], guides: Guide[]}} content */
export function buildGlossaryLinks({ terms, books, chapters, guides }) {
  const bookTitles = new Map(books.map((book) => [book.slug, book.title]));
  const targets = [
    ...chapters.filter((chapter) => bookTitles.has(chapter.parentSlug)).map((chapter) => ({
      kind: /** @type {const} */ ("chapter"),
      href: `/books/${chapter.parentSlug}/${chapter.slug}`,
      label: chapter.title,
      context: bookTitles.get(chapter.parentSlug),
      markdown: [chapter.body],
    })),
    ...guides.map((guide) => ({
      kind: /** @type {const} */ ("guide"),
      href: `/guides/${guide.slug}`,
      label: guide.title,
      context: null,
      markdown: guide.blocks.filter((block) => block.type === "markdown" && block.content?.trim()).map((block) => block.content),
    })),
  ].map((target) => {
    const mentions = new Map();
    for (const body of target.markdown) {
      for (const [slug, count] of getGlossaryMentions(body, terms)) {
        mentions.set(slug, (mentions.get(slug) ?? 0) + count);
      }
    }
    return { ...target, mentions };
  });

  /** @type {Record<string, LearnMoreLink[]>} */
  const map = {};
  for (const term of terms) {
    const links = targets
      .map((target) => ({ ...target, mentions: target.mentions.get(term.slug) ?? 0 }))
      .filter((target) => target.mentions > 0)
      .sort((a, b) => compareTargets(a, b, term.name))
      .slice(0, MAX_LINKS_PER_TERM)
      .map(({ kind, href, label, context, mentions }) => ({ kind, href, label, context, mentions }));
    if (links.length) map[term.slug] = links;
  }

  return {
    sources: {
      terms: terms.length,
      chapters: targets.filter((target) => target.kind === "chapter").length,
      guides: guides.length,
    },
    coverage: {
      termsWithLinks: Object.keys(map).length,
      totalLinks: Object.values(map).reduce((count, links) => count + links.length, 0),
      unlinkedTerms: terms.length - Object.keys(map).length,
    },
    terms: map,
  };
}
