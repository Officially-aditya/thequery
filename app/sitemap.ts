import type { MetadataRoute } from "next";
import { getAllBooks } from "@/lib/books";
import { getAllTerms } from "@/lib/glossary";
import { getAllIssues } from "@/lib/articles";
import { getAllGuides } from "@/lib/guides";

const BASE_URL = "https://www.thequery.in";

export const dynamic = "force-dynamic";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const [allTerms, allIssues, allGuides, allBooks] = await Promise.all([
    getAllTerms(),
    getAllIssues(),
    getAllGuides(),
    getAllBooks(),
  ]);

  // Compute latest dates per section for deterministic index page lastmod
  const latestTermDate = allTerms.reduce((max, t) => {
    const d = new Date(t.lastUpdated);
    return d > max ? d : max;
  }, new Date(0));

  const latestArticleDate = allIssues.reduce((max, i) => {
    const d = new Date(i.date);
    return d > max ? d : max;
  }, new Date(0));

  const latestGuideDate = allGuides.reduce((max, g) => {
    const d = new Date(g.date);
    return d > max ? d : max;
  }, new Date(0));

  const siteLastModified = new Date(
    Math.max(latestTermDate.getTime(), latestArticleDate.getTime(), latestGuideDate.getTime())
  );

  const entries: MetadataRoute.Sitemap = [
    { url: BASE_URL, lastModified: siteLastModified },
    { url: `${BASE_URL}/books`, lastModified: siteLastModified },
    { url: `${BASE_URL}/glossary`, lastModified: latestTermDate },
    { url: `${BASE_URL}/articles`, lastModified: latestArticleDate },
    { url: `${BASE_URL}/guides`, lastModified: latestGuideDate },
    { url: `${BASE_URL}/ai-word-of-the-day`, lastModified: siteLastModified },
    { url: `${BASE_URL}/about`, lastModified: siteLastModified },
    { url: `${BASE_URL}/privacy`, lastModified: siteLastModified },
  ];

  // Books and chapters
  for (const book of allBooks) {
    const bookLastModified = book.lastModified
      ? new Date(book.lastModified)
      : book.chapters.reduce((max, chapter) => {
          const date = chapter.lastModified ? new Date(chapter.lastModified) : max;
          return date > max ? date : max;
        }, new Date(0));
    entries.push({
      url: `${BASE_URL}/books/${book.slug}`,
      lastModified: bookLastModified.getTime() ? bookLastModified : siteLastModified,
    });
    for (const ch of book.chapters) {
      entries.push({
        url: `${BASE_URL}/books/${book.slug}/${ch.slug}`,
        lastModified: ch.lastModified ? new Date(ch.lastModified) : bookLastModified,
      });
    }
  }

  // Glossary terms
  for (const term of allTerms) {
    entries.push({
      url: `${BASE_URL}/glossary/${term.slug}`,
      lastModified: new Date(term.lastUpdated),
    });
  }

  // Guides
  for (const guide of allGuides) {
    entries.push({
      url: `${BASE_URL}/guides/${guide.slug}`,
      lastModified: new Date(guide.date),
    });
  }

  // Articles
  for (const issue of allIssues) {
    entries.push({
      url: `${BASE_URL}/articles/${issue.slug}`,
      lastModified: new Date(issue.date),
    });
  }

  return entries;
}
