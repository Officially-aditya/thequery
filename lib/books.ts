import "server-only";

import { getContentItem, getContentItems } from "./content";
import type { ContentBlock, Source } from "./content-types";

export interface ChapterMeta {
  slug: string;
  title: string;
  coverImageUrl?: string;
  coverImageAlt?: string;
  lastModified?: string;
}

export interface BookMeta {
  title: string;
  slug: string;
  description: string;
  author: string;
  coverImageUrl?: string;
  coverImageAlt?: string;
  lastModified?: string;
  chapters: ChapterMeta[];
}

function chapterMeta(item: Awaited<ReturnType<typeof getContentItem>> extends infer T ? Exclude<T, null> : never): ChapterMeta {
  return {
    slug: item.slug,
    title: item.title,
    ...(item.coverImageUrl ? { coverImageUrl: item.coverImageUrl } : {}),
    ...(item.coverImageAlt ? { coverImageAlt: item.coverImageAlt } : {}),
    ...(typeof item.metadata.lastModified === "string" ? { lastModified: item.metadata.lastModified } : {}),
  };
}

async function asBook(item: Awaited<ReturnType<typeof getContentItem>> extends infer T ? Exclude<T, null> : never): Promise<BookMeta> {
  const chapters = await getContentItems("chapter", { parentSlug: item.slug });
  return {
    title: item.title,
    slug: item.slug,
    description: item.summary,
    author: typeof item.metadata.author === "string" ? item.metadata.author : "TheQuery",
    ...(item.coverImageUrl ? { coverImageUrl: item.coverImageUrl } : {}),
    ...(item.coverImageAlt ? { coverImageAlt: item.coverImageAlt } : {}),
    ...(typeof item.metadata.lastModified === "string" ? { lastModified: item.metadata.lastModified } : {}),
    chapters: chapters.map(chapterMeta),
  };
}

export async function getAllBooks(): Promise<BookMeta[]> {
  const books = await getContentItems("book");
  return Promise.all(books.map(asBook));
}

export async function getBookMeta(slug: string): Promise<BookMeta | null> {
  const book = await getContentItem("book", slug);
  return book ? asBook(book) : null;
}

export async function getChapterContent(
  bookSlug: string,
  chapterSlug: string,
): Promise<{ content: string; blocks: ContentBlock[]; sources: Source[]; meta: ChapterMeta; book: BookMeta } | null> {
  const [book, chapter] = await Promise.all([
    getBookMeta(bookSlug),
    getContentItem("chapter", chapterSlug, bookSlug),
  ]);
  if (!book || !chapter) return null;

  return {
    content: chapter.body,
    blocks: chapter.blocks,
    sources: chapter.sources,
    meta: chapterMeta(chapter),
    book,
  };
}

export interface Heading {
  level: number;
  text: string;
  id: string;
}

export function extractHeadings(markdown: string): Heading[] {
  const headings: Heading[] = [];
  const lines = markdown.split("\n");
  let inCodeBlock = false;

  for (const line of lines) {
    if (line.trim().startsWith("```")) {
      inCodeBlock = !inCodeBlock;
      continue;
    }
    if (inCodeBlock) continue;

    const match = line.match(/^(#{1,3})\s+(.+)$/);
    if (match) {
      const level = match[1].length;
      const text = match[2].replace(/[*_`]/g, "").trim();
      const id = text
        .toLowerCase()
        .replace(/[^a-z0-9\s-]/g, "")
        .replace(/\s+/g, "-")
        .replace(/-+/g, "-")
        .replace(/(^-|-$)/g, "");
      headings.push({ level, text, id });
    }
  }

  return headings;
}

export interface Section {
  title: string;
  id: string;
  content: string;
}

export function splitIntoSections(markdown: string): Section[] {
  const lines = markdown.split("\n");
  const sections: Section[] = [];
  let currentLines: string[] = [];
  let currentTitle = "";
  let currentId = "";

  for (const line of lines) {
    const match = line.match(/^## (.+)$/);
    if (match) {
      if (currentLines.length > 0) {
        sections.push({ title: currentTitle, id: currentId, content: currentLines.join("\n").trim() });
      }
      currentTitle = match[1].replace(/[*_`]/g, "").trim();
      currentId = currentTitle
        .toLowerCase()
        .replace(/[^a-z0-9\s-]/g, "")
        .replace(/\s+/g, "-")
        .replace(/-+/g, "-")
        .replace(/(^-|-$)/g, "");
      currentLines = [line];
    } else {
      currentLines.push(line);
    }
  }

  if (currentLines.length > 0) {
    sections.push({ title: currentTitle, id: currentId, content: currentLines.join("\n").trim() });
  }

  const merged: Section[] = [];
  let index = 0;
  while (index < sections.length) {
    const current = { ...sections[index] };
    while (index + 1 < sections.length && current.content.length < 800) {
      index += 1;
      current.content += `\n\n${sections[index].content}`;
    }
    merged.push(current);
    index += 1;
  }

  return merged;
}

export async function getAdjacentChapters(
  bookSlug: string,
  chapterSlug: string,
): Promise<{ prev: ChapterMeta | null; next: ChapterMeta | null }> {
  const book = await getBookMeta(bookSlug);
  if (!book) return { prev: null, next: null };

  const index = book.chapters.findIndex((chapter) => chapter.slug === chapterSlug);
  return {
    prev: index > 0 ? book.chapters[index - 1] : null,
    next: index >= 0 && index < book.chapters.length - 1 ? book.chapters[index + 1] : null,
  };
}
