import "server-only";

import { unstable_cache } from "next/cache";
import { getContentIndex, getContentItems, getContentSummaries } from "./content";
import { buildGlossaryLinks } from "./glossary-link-map.mjs";

export interface GlossaryLearnMoreLink {
  kind: "chapter" | "guide";
  href: string;
  label: string;
  context: string | null;
  mentions: number;
}

const getLinkMap = unstable_cache(
  async () => {
    const [index, books, guides] = await Promise.all([
      getContentIndex("glossary"),
      getContentSummaries("book"),
      getContentItems("guide"),
    ]);
    const chapters = (await Promise.all(
      books.map((book) => getContentItems("chapter", { parentSlug: book.slug })),
    )).flat();
    return buildGlossaryLinks({
      terms: index.map((term) => ({ name: term.title, slug: term.slug })),
      books,
      chapters,
      guides,
    }).terms;
  },
  ["glossary-learn-more-v1"],
  { tags: ["content:glossary", "content:book", "content:chapter", "content:guide"] },
);

export async function getLearnMoreLinks(termSlug: string): Promise<GlossaryLearnMoreLink[]> {
  return (await getLinkMap())[termSlug] ?? [];
}
