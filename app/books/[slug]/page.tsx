import Link from "next/link";
import BookCitation from "@/components/BookCitation";
import { getAllBooks, getBookMeta } from "@/lib/books";
import { notFound } from "next/navigation";
import CoverImage from "@/components/content/CoverImage";
import {
  ORGANIZATION_ID,
  ORGANIZATION_LOGO,
  SITE_URL,
  authorJsonLd,
  createOpenGraphMetadata,
} from "@/lib/site";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ slug: string }>;
}

export const revalidate = false;

export async function generateStaticParams() {
  return (await getAllBooks()).map(({ slug }) => ({ slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const book = await getBookMeta(slug);
  if (!book) return {};
  return {
    title: book.title,
    description: book.description,
    openGraph: createOpenGraphMetadata({
      title: book.title,
      description: book.description,
      url: `${SITE_URL}/books/${book.slug}`,
      type: "book",
      image: book.coverImageUrl,
    }),
  };
}

export default async function BookPage({ params }: Props) {
  const { slug } = await params;
  const book = await getBookMeta(slug);
  if (!book) notFound();

  const bookUrl = `${SITE_URL}/books/${book.slug}`;
  const edition = book.lastModified?.slice(0, 10);
  const citation = `${book.author}. ${book.title}. TheQuery.${edition ? ` Updated ${edition}.` : ""} ${bookUrl}`;
  const bibEscape = (value: string) => value.replace(/[\\{}%&#_$]/g, (char) => `\\${char}`);
  const bibtex = `@book{thequery-${book.slug},\n  author = {${bibEscape(book.author)}},\n  title = {${bibEscape(book.title)}},\n  publisher = {TheQuery},\n  url = {${bookUrl}}${edition ? `,\n  note = {Online edition updated ${edition}}` : ""}\n}`;

  const jsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Book",
        name: book.title,
        description: book.description,
        author: { ...authorJsonLd, name: book.author },
        dateModified: book.lastModified,
        url: `${SITE_URL}/books/${book.slug}`,
        inLanguage: "en",
        publisher: {
          "@type": "Organization",
          "@id": ORGANIZATION_ID,
          name: "TheQuery",
          logo: {
            "@type": "ImageObject",
            url: ORGANIZATION_LOGO,
          },
        },
        isAccessibleForFree: true,
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Home", item: SITE_URL },
          { "@type": "ListItem", position: 2, name: "Books", item: `${SITE_URL}/books` },
          { "@type": "ListItem", position: 3, name: book.title },
        ],
      },
    ],
  };

  return (
    <div className="max-w-[960px] mx-auto px-4 py-12">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <Link href="/books" className="text-sm text-text-muted hover:text-text-secondary transition-colors mb-6 inline-block">
        &larr; All Books
      </Link>

      <h1 className="font-serif text-3xl font-bold text-text-primary mb-2">
        {book.title}
      </h1>
      <CoverImage src={book.coverImageUrl} alt={book.coverImageAlt} title={book.title} />
      <p className="text-sm text-text-muted mb-2">
        By <Link href="/about" className="text-accent hover:text-accent-hover transition-colors">{book.author}</Link>
        {book.lastModified ? (
          <> &middot; Updated {new Date(book.lastModified).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}</>
        ) : null}
      </p>
      <p className="text-text-secondary mb-4 leading-relaxed">
        {book.description}
      </p>
      <p className="mb-4 font-medium text-accent">Free to read. No account required.</p>
      {["ai-from-first-principles", "rag-kg-master-course"].includes(book.slug) ? <p className="mb-4 text-sm"><a className="text-accent underline" href={`/books/downloads/${book.slug}.epub`} download>Download EPUB</a><span className="text-text-muted"> · Offline snapshot: September 9, 2026</span></p> : null}
      <p className="text-sm text-text-muted mb-8">
        This educational book follows TheQuery&apos;s <Link href="/about#editorial-standards" className="text-accent hover:text-accent-hover transition-colors">editorial standards</Link>. Report a factual correction at <a href="mailto:addy@thequery.in" className="text-accent hover:text-accent-hover transition-colors">addy@thequery.in</a>.
      </p>

      <div className="border border-border rounded-lg overflow-hidden">
        <div className="bg-bg-secondary px-4 py-3 border-b border-border">
          <h2 className="font-serif text-sm font-semibold text-text-primary">Table of Contents</h2>
        </div>
        <ul className="divide-y divide-border">
          {book.chapters.map((chapter) => (
            <li key={chapter.slug}>
              <Link
                href={`/books/${book.slug}/${chapter.slug}`}
                className="block px-4 py-3 hover:bg-bg-secondary transition-colors group"
              >
                <span className="text-sm text-text-primary group-hover:text-accent transition-colors">
                  {chapter.title}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      </div>
      <BookCitation citation={citation} bibtex={bibtex} />
      <section className="mt-8 space-y-3 text-sm text-text-secondary">
        <h2 className="font-serif text-xl font-semibold text-text-primary">Edition and reuse</h2>
        <p>{edition ? `Online edition updated ${edition}.` : "Continuously maintained online edition."} Cite the page URL and include your access date when referring to a changing online edition.</p>
        <p>Reading is free. Copyright remains with the author; contact <a className="text-accent underline" href="mailto:addy@thequery.in">addy@thequery.in</a> for republication or translation permission.</p>
        <p>September 9, 2026: added citation and BibTeX tools to this landing page. This is a presentation update, not a new content edition.</p>
        <Link className="text-accent underline" href="/research">More resources for researchers and educators</Link>
      </section>
    </div>
  );
}
