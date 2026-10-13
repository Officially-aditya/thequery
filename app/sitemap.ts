import type { MetadataRoute } from "next";
import { getAllBooks } from "@/lib/books";
import { getAllTermSummaries } from "@/lib/glossary";
import { getAllIssues } from "@/lib/articles";
import { getAllGuides } from "@/lib/guides";
import { getAllComparisons } from "@/lib/comparisons";

const BASE_URL = "https://www.thequery.in";

export const revalidate = false;

export async function generateSitemaps() {
  return [
    { id: "pages" },
    { id: "books" },
    { id: "guides" },
    { id: "comparisons" },
    { id: "articles" },
    { id: "glossary" },
  ];
}

export default async function sitemap(props?: { id?: Promise<string> | string }): Promise<MetadataRoute.Sitemap> {
  const targetId = props?.id ? (await props.id)?.toString() : undefined;

  const [allTerms, allIssues, allGuides, allComparisons, allBooks] = await Promise.all([
    !targetId || targetId === "glossary" || targetId === "pages" ? getAllTermSummaries() : Promise.resolve([]),
    !targetId || targetId === "articles" || targetId === "pages" ? getAllIssues() : Promise.resolve([]),
    !targetId || targetId === "guides" || targetId === "pages" ? getAllGuides() : Promise.resolve([]),
    !targetId || targetId === "comparisons" || targetId === "pages" ? getAllComparisons() : Promise.resolve([]),
    !targetId || targetId === "books" || targetId === "pages" ? getAllBooks() : Promise.resolve([]),
  ]);

  const latestTermDate = allTerms.reduce((max, t) => {
    const d = new Date(t.lastUpdated);
    return d > max ? d : max;
  }, new Date(0));

  const latestArticleDate = allIssues.reduce((max, i) => {
    const d = new Date(i.lastModified ?? i.date);
    return d > max ? d : max;
  }, new Date(0));

  const latestGuideDate = allGuides.reduce((max, g) => {
    const d = new Date(g.lastModified ?? g.date);
    return d > max ? d : max;
  }, new Date(0));

  const latestComparisonDate = allComparisons.reduce((max, c) => {
    const d = new Date(c.lastModified ?? c.date);
    return d > max ? d : max;
  }, new Date(0));

  const siteLastModified = new Date(
    Math.max(latestTermDate.getTime(), latestArticleDate.getTime(), latestGuideDate.getTime(), latestComparisonDate.getTime())
  );

  const pageEntries: MetadataRoute.Sitemap = [
    // Featured learning links and the research section were added on this date.
    { url: BASE_URL, lastModified: new Date(Math.max(siteLastModified.getTime(), new Date("2026-10-13").getTime())) },
    { url: `${BASE_URL}/books`, lastModified: siteLastModified },
    { url: `${BASE_URL}/glossary`, lastModified: latestTermDate },
    { url: `${BASE_URL}/articles`, lastModified: latestArticleDate },
    { url: `${BASE_URL}/guides`, lastModified: latestGuideDate },
    { url: `${BASE_URL}/comparisons`, lastModified: latestComparisonDate },
    { url: `${BASE_URL}/ai-word-of-the-day`, lastModified: siteLastModified },
    { url: `${BASE_URL}/about` },
    { url: `${BASE_URL}/research` },
    { url: `${BASE_URL}/research/benchmark-comparability`, lastModified: new Date("2026-10-04") },
    { url: `${BASE_URL}/press` },
    { url: `${BASE_URL}/privacy` },
  ];

  const bookEntries: MetadataRoute.Sitemap = [];
  for (const book of allBooks) {
    const bookLastModified = book.lastModified
      ? new Date(book.lastModified)
      : book.chapters.reduce((max, chapter) => {
          const date = chapter.lastModified ? new Date(chapter.lastModified) : max;
          return date > max ? date : max;
        }, new Date(0));
    bookEntries.push({
      url: `${BASE_URL}/books/${book.slug}`,
      lastModified: bookLastModified.getTime() ? bookLastModified : siteLastModified,
    });
    for (const ch of book.chapters) {
      bookEntries.push({
        url: `${BASE_URL}/books/${book.slug}/${ch.slug}`,
        lastModified: ch.lastModified ? new Date(ch.lastModified) : bookLastModified,
      });
    }
  }

  const guideEntries: MetadataRoute.Sitemap = allGuides.map((guide) => ({
    url: `${BASE_URL}/guides/${guide.slug}`,
    lastModified: new Date(guide.lastModified ?? guide.date),
  }));

  const comparisonEntries: MetadataRoute.Sitemap = allComparisons.map((comparison) => ({
    url: `${BASE_URL}/comparisons/${comparison.slug}`,
    lastModified: new Date(comparison.lastModified ?? comparison.date),
  }));

  const articleEntries: MetadataRoute.Sitemap = allIssues.map((issue) => ({
    url: `${BASE_URL}/articles/${issue.slug}`,
    lastModified: new Date(issue.lastModified ?? issue.date),
  }));

  const glossaryEntries: MetadataRoute.Sitemap = allTerms.map((term) => ({
    url: `${BASE_URL}/glossary/${term.slug}`,
    lastModified: new Date(term.lastUpdated),
  }));

  if (targetId === "pages") return pageEntries;
  if (targetId === "books") return bookEntries;
  if (targetId === "guides") return guideEntries;
  if (targetId === "comparisons") return comparisonEntries;
  if (targetId === "articles") return articleEntries;
  if (targetId === "glossary") return glossaryEntries;

  return [
    ...pageEntries,
    ...bookEntries,
    ...glossaryEntries,
    ...guideEntries,
    ...comparisonEntries,
    ...articleEntries,
  ];
}
