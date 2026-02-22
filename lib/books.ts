import fs from "fs";
import path from "path";
import matter from "gray-matter";

const booksDir = path.join(process.cwd(), "content", "books");

export interface ChapterMeta {
  slug: string;
  title: string;
  file: string;
}

export interface BookMeta {
  title: string;
  slug: string;
  description: string;
  author: string;
  chapters: ChapterMeta[];
}

export function getAllBooks(): BookMeta[] {
  const slugs = fs.readdirSync(booksDir).filter((f) => {
    return fs.statSync(path.join(booksDir, f)).isDirectory();
  });

  return slugs.map((slug) => getBookMeta(slug)).filter(Boolean) as BookMeta[];
}

export function getBookMeta(slug: string): BookMeta | null {
  const metaPath = path.join(booksDir, slug, "meta.json");
  if (!fs.existsSync(metaPath)) return null;
  const raw = fs.readFileSync(metaPath, "utf-8");
  return JSON.parse(raw);
}

export function getChapterContent(
  bookSlug: string,
  chapterSlug: string
): { content: string; meta: ChapterMeta; book: BookMeta } | null {
  const book = getBookMeta(bookSlug);
  if (!book) return null;

  const chapter = book.chapters.find((c) => c.slug === chapterSlug);
  if (!chapter) return null;

  const filePath = path.join(booksDir, bookSlug, chapter.file);
  if (!fs.existsSync(filePath)) return null;

  const raw = fs.readFileSync(filePath, "utf-8");
  const { content } = matter(raw);

  return { content, meta: chapter, book };
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
      // Save previous section
      if (currentLines.length > 0) {
        sections.push({
          title: currentTitle,
          id: currentId,
          content: currentLines.join("\n").trim(),
        });
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

  // Push last section
  if (currentLines.length > 0) {
    sections.push({
      title: currentTitle,
      id: currentId,
      content: currentLines.join("\n").trim(),
    });
  }

  // Merge small sections with the next one so short content doesn't look odd
  const MIN_LENGTH = 800; // characters
  const merged: Section[] = [];
  let i = 0;
  while (i < sections.length) {
    const current = { ...sections[i] };
    // Keep merging forward while the accumulated content is small
    while (i + 1 < sections.length && current.content.length < MIN_LENGTH) {
      i++;
      current.content += "\n\n" + sections[i].content;
    }
    merged.push(current);
    i++;
  }

  return merged;
}

export function getAdjacentChapters(
  bookSlug: string,
  chapterSlug: string
): { prev: ChapterMeta | null; next: ChapterMeta | null } {
  const book = getBookMeta(bookSlug);
  if (!book) return { prev: null, next: null };

  const idx = book.chapters.findIndex((c) => c.slug === chapterSlug);
  return {
    prev: idx > 0 ? book.chapters[idx - 1] : null,
    next: idx < book.chapters.length - 1 ? book.chapters[idx + 1] : null,
  };
}
