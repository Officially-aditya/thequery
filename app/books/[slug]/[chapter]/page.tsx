import { getChapterContent, getAdjacentChapters, splitIntoSections } from "@/lib/books";
import { getGlossaryIndex } from "@/lib/glossary";
import { notFound } from "next/navigation";
import Link from "next/link";
import ReadingProgress from "@/components/ReadingProgress";
import ChapterView from "@/components/ChapterView";
import CoverImage from "@/components/content/CoverImage";
import {
  AUTHOR,
  ORGANIZATION_ID,
  ORGANIZATION_LOGO,
  SITE_URL,
  authorJsonLd,
  createOpenGraphMetadata,
} from "@/lib/site";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ slug: string; chapter: string }>;
}

export const dynamic = "force-dynamic";

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug, chapter } = await params;
  const data = await getChapterContent(slug, chapter);
  if (!data) return {};
  return {
    title: `${data.meta.title} - ${data.book.title}`,
    description: `Read "${data.meta.title}" from ${data.book.title} on TheQuery.`,
    openGraph: createOpenGraphMetadata({
      title: `${data.meta.title} - ${data.book.title}`,
      description: `Read "${data.meta.title}" from ${data.book.title} on TheQuery.`,
      url: `${SITE_URL}/books/${data.book.slug}/${data.meta.slug}`,
      type: "article",
      image: data.meta.coverImageUrl,
    }),
  };
}

export default async function ChapterPage({ params }: Props) {
  const { slug, chapter } = await params;
  const data = await getChapterContent(slug, chapter);
  if (!data) notFound();

  const { prev, next } = await getAdjacentChapters(slug, chapter);
  const currentIdx = data.book.chapters.findIndex((c) => c.slug === chapter);
  const sections = splitIntoSections(data.content);
  const glossaryTerms = await getGlossaryIndex();

  const jsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Chapter",
        name: data.meta.title,
        url: `${SITE_URL}/books/${slug}/${chapter}`,
        position: currentIdx + 1,
        author: { ...authorJsonLd, name: data.book.author },
        dateModified: data.meta.lastModified ?? data.book.lastModified,
        publisher: {
          "@type": "Organization",
          "@id": ORGANIZATION_ID,
          name: "TheQuery",
          logo: {
            "@type": "ImageObject",
            url: ORGANIZATION_LOGO,
          },
        },
        isPartOf: {
          "@type": "Book",
          name: data.book.title,
          url: `${SITE_URL}/books/${slug}`,
        },
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Home", item: SITE_URL },
          { "@type": "ListItem", position: 2, name: "Books", item: `${SITE_URL}/books` },
          { "@type": "ListItem", position: 3, name: data.book.title, item: `${SITE_URL}/books/${slug}` },
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
      <div data-reading-frame className="max-w-[1100px] mx-auto px-4 py-12">
        <p className="text-xs text-text-muted mb-6">
          By <Link href={AUTHOR.url} className="text-accent hover:text-accent-hover transition-colors">{data.book.author}</Link>
          {data.meta.lastModified ? (
            <> &middot; Updated {new Date(data.meta.lastModified).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}</>
          ) : null}
        </p>
        <CoverImage src={data.meta.coverImageUrl} alt={data.meta.coverImageAlt} title={data.meta.title} />
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
