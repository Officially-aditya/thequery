"use client";

import { useState } from "react";
import ChapterSidebar from "./ChapterSidebar";
import SectionReader from "./SectionReader";
import type { ChapterMeta, Section } from "@/lib/books";
import type { GlossaryLink } from "./MarkdownRenderer";

interface Props {
  bookSlug: string;
  bookTitle: string;
  chapters: ChapterMeta[];
  currentChapter: string;
  currentIdx: number;
  sections: Section[];
  prevChapter: { slug: string; title: string } | null;
  nextChapter: { slug: string; title: string } | null;
  glossaryTerms?: GlossaryLink[];
}

export default function ChapterView({
  bookSlug,
  bookTitle,
  chapters,
  currentChapter,
  currentIdx,
  sections,
  prevChapter,
  nextChapter,
  glossaryTerms = [],
}: Props) {
  const [currentSection, setCurrentSection] = useState(0);

  return (
    <div className="flex flex-col lg:flex-row gap-8">
      <ChapterSidebar
        bookSlug={bookSlug}
        bookTitle={bookTitle}
        chapters={chapters}
        currentChapter={currentChapter}
        sections={sections}
        currentSection={currentSection}
        onSectionChange={setCurrentSection}
      />

      <article className="flex-1 min-w-0">
        <p className="text-xs text-text-muted mb-1">
          Chapter {currentIdx + 1} of {chapters.length}
        </p>
        <SectionReader
          sections={sections}
          currentSection={currentSection}
          onSectionChange={setCurrentSection}
          nav={{ prevChapter, nextChapter, bookSlug }}
          glossaryTerms={glossaryTerms}
        />
      </article>
    </div>
  );
}
