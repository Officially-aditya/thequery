import type { MetadataRoute } from "next";
import { getAllBooks } from "@/lib/books";
import { getAllTerms } from "@/lib/glossary";
import { getAllIssues } from "@/lib/articles";
import { getAllGuides } from "@/lib/guides";

const BASE_URL = "https://www.thequery.in";

export default function sitemap(): MetadataRoute.Sitemap {
  const allTerms = getAllTerms();
  const allIssues = getAllIssues();
  const allGuides = getAllGuides();
  const allBooks = getAllBooks();

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
  ];

  // Books and chapters
  for (const book of allBooks) {
    entries.push({
      url: `${BASE_URL}/books/${book.slug}`,
      lastModified: siteLastModified,
    });
    for (const ch of book.chapters) {
      entries.push({
        url: `${BASE_URL}/books/${book.slug}/${ch.slug}`,
        lastModified: siteLastModified,
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
