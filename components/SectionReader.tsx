import MarkdownRenderer, { type GlossaryLink } from "./MarkdownRenderer";
import type { Section } from "@/lib/books";

interface NavInfo {
  prevChapter: { slug: string; title: string } | null;
  nextChapter: { slug: string; title: string } | null;
  bookSlug: string;
}

function PrevNextBar({ nav }: { nav: NavInfo }) {
  return (
    <nav aria-label="Adjacent chapters" className="flex justify-between items-center gap-4 py-4">
      {nav.prevChapter ? (
        <a
          href={`/books/${nav.bookSlug}/${nav.prevChapter.slug}`}
          className="text-sm text-accent hover:text-accent-hover transition-colors"
        >
          &larr; {nav.prevChapter.title}
        </a>
      ) : <span />}
      {nav.nextChapter ? (
        <a
          href={`/books/${nav.bookSlug}/${nav.nextChapter.slug}`}
          className="text-sm text-accent hover:text-accent-hover transition-colors text-right"
        >
          {nav.nextChapter.title} &rarr;
        </a>
      ) : <span />}
    </nav>
  );
}

export default function SectionReader({
  sections,
  nav,
  glossaryTerms = [],
}: {
  sections: Section[];
  nav: NavInfo;
  glossaryTerms?: GlossaryLink[];
}) {
  return (
    <div>
      <div className="mb-6 border-b border-border">
        <PrevNextBar nav={nav} />
      </div>
      {sections.map((section, idx) => (
        <section key={idx} id={`section-${idx + 1}`} className="scroll-mt-24 mb-8">
          <MarkdownRenderer content={section.content} glossaryTerms={glossaryTerms} />
        </section>
      ))}
      <div className="mt-8 pt-4 border-t border-border">
        <PrevNextBar nav={nav} />
      </div>
    </div>
  );
}
