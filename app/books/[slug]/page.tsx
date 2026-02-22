import Link from "next/link";
import { getAllBooks, getBookMeta } from "@/lib/books";
import { notFound } from "next/navigation";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  return getAllBooks().map((book) => ({ slug: book.slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const book = getBookMeta(slug);
  if (!book) return {};
  return {
    title: book.title,
    description: book.description,
    openGraph: { title: book.title, description: book.description },
  };
}

export default async function BookPage({ params }: Props) {
  const { slug } = await params;
  const book = getBookMeta(slug);
  if (!book) notFound();

  return (
    <div className="max-w-[960px] mx-auto px-4 py-12">
      <Link href="/books" className="text-sm text-text-muted hover:text-text-secondary transition-colors mb-6 inline-block">
        &larr; All Books
      </Link>

      <h1 className="font-serif text-3xl font-bold text-text-primary mb-2">
        {book.title}
      </h1>
      <p className="text-sm text-text-muted mb-2">By {book.author}</p>
      <p className="text-text-secondary mb-4 leading-relaxed">
        {book.description}
      </p>
      <a
        href="#"
        className="inline-flex items-center gap-1.5 text-sm font-medium border border-border text-text-secondary rounded-md px-3 py-1.5 hover:border-accent hover:text-accent transition-colors mb-8"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        Download PDF
      </a>

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
    </div>
  );
}
