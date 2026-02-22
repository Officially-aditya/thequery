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
      <p className="text-text-secondary mb-8 leading-relaxed">
        {book.description}
      </p>

      <div className="border border-border rounded-lg overflow-hidden">
        <div className="bg-bg-secondary px-4 py-3 border-b border-border">
          <h2 className="font-serif text-sm font-semibold text-text-primary">Table of Contents</h2>
        </div>
        <ol className="divide-y divide-border">
          {book.chapters.map((chapter, idx) => (
            <li key={chapter.slug}>
              <Link
                href={`/books/${book.slug}/${chapter.slug}`}
                className="flex items-center px-4 py-3 hover:bg-bg-secondary transition-colors group"
              >
                <span className="text-sm text-text-muted w-8">{idx + 1}.</span>
                <span className="text-sm text-text-primary group-hover:text-accent transition-colors">
                  {chapter.title}
                </span>
              </Link>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
