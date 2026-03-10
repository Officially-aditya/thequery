import { getAllBooks, getChapterContent, getAdjacentChapters, splitIntoSections } from "@/lib/books";
import { getAllTerms } from "@/lib/glossary";
import { notFound } from "next/navigation";
import ReadingProgress from "@/components/ReadingProgress";
import ChapterView from "@/components/ChapterView";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ slug: string; chapter: string }>;
}

export async function generateStaticParams() {
  const books = getAllBooks();
  const params: { slug: string; chapter: string }[] = [];
  for (const book of books) {
    for (const ch of book.chapters) {
      params.push({ slug: book.slug, chapter: ch.slug });
    }
  }
  return params;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug, chapter } = await params;
  const data = getChapterContent(slug, chapter);
  if (!data) return {};
  return {
    title: `${data.meta.title} - ${data.book.title}`,
    description: `Read "${data.meta.title}" from ${data.book.title} on TheQuery.`,
    openGraph: { title: `${data.meta.title} - ${data.book.title}` },
  };
}

export default async function ChapterPage({ params }: Props) {
  const { slug, chapter } = await params;
  const data = getChapterContent(slug, chapter);
  if (!data) notFound();

  const { prev, next } = getAdjacentChapters(slug, chapter);
  const currentIdx = data.book.chapters.findIndex((c) => c.slug === chapter);
  const sections = splitIntoSections(data.content);
  const glossaryTerms = getAllTerms().map(({ name, slug: s }) => ({ name, slug: s }));

  const jsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Chapter",
        name: data.meta.title,
        url: `https://www.thequery.in/books/${slug}/${chapter}`,
        position: currentIdx + 1,
        isPartOf: {
          "@type": "Book",
          name: data.book.title,
          url: `https://www.thequery.in/books/${slug}`,
        },
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Home", item: "https://www.thequery.in" },
          { "@type": "ListItem", position: 2, name: "Books", item: "https://www.thequery.in/books" },
          { "@type": "ListItem", position: 3, name: data.book.title, item: `https://www.thequery.in/books/${slug}` },
          { "@type": "ListItem", position: 4, name: data.meta.title },
        ],
      },
    ],
  };

  return (
    <>
      <link
        rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"
        crossOrigin="anonymous"
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <ReadingProgress />
      <div className="max-w-[1100px] mx-auto px-4 py-12">
        <ChapterView
          bookSlug={slug}
          bookTitle={data.book.title}
          chapters={data.book.chapters}
          currentChapter={chapter}
          currentIdx={currentIdx}
          sections={sections}
          prevChapter={prev ? { slug: prev.slug, title: prev.title } : null}
          nextChapter={next ? { slug: next.slug, title: next.title } : null}
          glossaryTerms={glossaryTerms}
        />
      </div>
    </>
  );
}
