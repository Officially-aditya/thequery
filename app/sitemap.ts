import type { MetadataRoute } from "next";
import { getAllBooks } from "@/lib/books";
import { getAllTerms } from "@/lib/glossary";
import { getAllIssues } from "@/lib/articles";

const BASE_URL = "https://thequery.in";

export default function sitemap(): MetadataRoute.Sitemap {
  const entries: MetadataRoute.Sitemap = [
    { url: BASE_URL, lastModified: new Date(), changeFrequency: "weekly", priority: 1 },
    { url: `${BASE_URL}/books`, lastModified: new Date(), changeFrequency: "weekly", priority: 0.8 },
    { url: `${BASE_URL}/glossary`, lastModified: new Date(), changeFrequency: "weekly", priority: 0.8 },
    { url: `${BASE_URL}/articles`, lastModified: new Date(), changeFrequency: "weekly", priority: 0.8 },
  ];

  // Books and chapters
  for (const book of getAllBooks()) {
    entries.push({
      url: `${BASE_URL}/books/${book.slug}`,
      changeFrequency: "monthly",
      priority: 0.7,
    });
    for (const ch of book.chapters) {
      entries.push({
        url: `${BASE_URL}/books/${book.slug}/${ch.slug}`,
        changeFrequency: "monthly",
        priority: 0.6,
      });
    }
  }

  // Glossary terms
  for (const term of getAllTerms()) {
    entries.push({
      url: `${BASE_URL}/glossary/${term.slug}`,
      lastModified: new Date(term.lastUpdated),
      changeFrequency: "monthly",
      priority: 0.6,
    });
  }

  // Articles
  for (const issue of getAllIssues()) {
    entries.push({
      url: `${BASE_URL}/articles/${issue.slug}`,
      lastModified: new Date(issue.date),
      changeFrequency: "yearly",
      priority: 0.5,
    });
  }

  return entries;
}
