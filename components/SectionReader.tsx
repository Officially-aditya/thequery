"use client";

import MarkdownRenderer from "./MarkdownRenderer";
import type { GlossaryLink } from "./MarkdownRenderer";

interface Section {
  title: string;
  id: string;
  content: string;
}

interface NavInfo {
  prevChapter: { slug: string; title: string } | null;
  nextChapter: { slug: string; title: string } | null;
  bookSlug: string;
}

function PrevNextBar({
  sections,
  current,
  onNav,
  nav,
}: {
  sections: Section[];
  current: number;
  onNav: (i: number) => void;
  nav: NavInfo;
}) {
  const isFirst = current === 0;
  const isLast = current === sections.length - 1;

  const handlePrev = () => {
    onNav(current - 1);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleNext = () => {
    onNav(current + 1);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="flex justify-between items-center py-4">
      {isFirst && nav.prevChapter ? (
        <a
          href={`/books/${nav.bookSlug}/${nav.prevChapter.slug}`}
          className="text-sm text-accent hover:text-accent-hover transition-colors"
        >
          &larr; {nav.prevChapter.title}
        </a>
      ) : !isFirst ? (
        <button
          onClick={handlePrev}
          className="text-sm text-accent hover:text-accent-hover transition-colors text-left max-w-[40%]"
        >
          &larr; {sections[current - 1].title || "Previous"}
        </button>
      ) : (
        <span />
      )}

      <span className="text-xs text-text-muted shrink-0">
        {current + 1} / {sections.length}
      </span>

      {isLast && nav.nextChapter ? (
        <a
          href={`/books/${nav.bookSlug}/${nav.nextChapter.slug}`}
          className="text-sm text-accent hover:text-accent-hover transition-colors"
        >
          {nav.nextChapter.title} &rarr;
        </a>
      ) : !isLast ? (
        <button
          onClick={handleNext}
          className="text-sm text-accent hover:text-accent-hover transition-colors text-right max-w-[40%]"
        >
          {sections[current + 1].title || "Next"} &rarr;
        </button>
      ) : (
        <span />
      )}
    </div>
  );
}

export default function SectionReader({
  sections,
  currentSection,
  onSectionChange,
  nav,
  glossaryTerms = [],
}: {
  sections: Section[];
  currentSection: number;
  onSectionChange: (i: number) => void;
  nav: NavInfo;
  glossaryTerms?: GlossaryLink[];
}) {
  if (sections.length <= 1) {
    return (
      <div>
        <MarkdownRenderer content={sections[0]?.content || ""} glossaryTerms={glossaryTerms} />
        <div className="mt-8 pt-4 border-t border-border">
          <PrevNextBar sections={sections} current={0} onNav={onSectionChange} nav={nav} />
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-6 border-b border-border">
        <PrevNextBar sections={sections} current={currentSection} onNav={onSectionChange} nav={nav} />
      </div>

      <MarkdownRenderer content={sections[currentSection].content} glossaryTerms={glossaryTerms} />

      <div className="mt-8 pt-4 border-t border-border">
        <PrevNextBar sections={sections} current={currentSection} onNav={onSectionChange} nav={nav} />
      </div>
    </div>
  );
}
