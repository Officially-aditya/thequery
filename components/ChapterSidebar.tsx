"use client";

import Link from "next/link";
import type { ChapterMeta, Section } from "@/lib/books";

interface Props {
  bookSlug: string;
  bookTitle: string;
  chapters: ChapterMeta[];
  currentChapter: string;
  sections: Section[];
  currentSection: number;
  onSectionChange: (idx: number) => void;
}

export default function ChapterSidebar({
  bookSlug,
  bookTitle,
  chapters,
  currentChapter,
  sections,
  currentSection,
  onSectionChange,
}: Props) {
  return (
    <aside className="lg:w-60 shrink-0">
      <div className="lg:sticky lg:top-[72px] lg:max-h-[calc(100vh-88px)] lg:overflow-y-auto">
        <Link
          href={`/books/${bookSlug}`}
          className="text-sm font-serif font-semibold text-text-primary hover:text-accent transition-colors block mb-4"
        >
          {bookTitle}
        </Link>

        {/* Chapter list */}
        <nav className="space-y-0.5 mb-5">
          {chapters.map((ch, idx) => (
            <Link
              key={ch.slug}
              href={`/books/${bookSlug}/${ch.slug}`}
              className={`block text-xs py-1 px-2 rounded transition-colors ${
                ch.slug === currentChapter
                  ? "bg-bg-secondary text-accent font-medium"
                  : "text-text-muted hover:text-text-secondary"
              }`}
            >
              {ch.title}
            </Link>
          ))}
        </nav>

        {/* Subsections for current chapter */}
        {sections.length > 1 && (
          <div className="border-t border-border pt-4">
            <p className="text-[10px] uppercase tracking-wider text-text-muted font-semibold mb-2 px-2">
              On this page
            </p>
            <nav className="space-y-0.5">
              {sections.map((section, idx) => (
                <button
                  key={section.id || idx}
                  onClick={() => {
                    onSectionChange(idx);
                    window.scrollTo({ top: 0, behavior: "smooth" });
                  }}
                  className={`block w-full text-left text-xs py-1 px-2 rounded transition-colors truncate ${
                    currentSection === idx
                      ? "text-accent font-medium bg-bg-secondary"
                      : "text-text-muted hover:text-text-secondary"
                  }`}
                  title={section.title || `Section ${idx + 1}`}
                >
                  {idx + 1}. {section.title || `Introduction`}
                </button>
              ))}
            </nav>
          </div>
        )}
      </div>
    </aside>
  );
}
