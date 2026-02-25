import Link from "next/link";
import { getAllBooks } from "@/lib/books";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Books",
  description: "In-depth AI and machine learning books, written from first principles. Read online for free.",
  openGraph: {
    title: "Books - TheQuery",
    description: "In-depth AI and machine learning books, written from first principles. Read online for free.",
  },
};

export default function BooksPage() {
  const books = getAllBooks();

  return (
    <div className="max-w-[960px] mx-auto px-4 py-12">
      <h1 className="font-serif text-3xl font-bold text-text-primary mb-2">Books</h1>
      <p className="text-text-secondary mb-8">
        In-depth guides on AI and machine learning, written from first principles.
      </p>

      <div className="space-y-8">
        {books.map((book) => (
          <div key={book.slug} className="border-b border-border pb-8 last:border-b-0">
            <h2 className="font-serif text-xl font-semibold text-text-primary mb-2">
              {book.title}
            </h2>
            <p className="text-sm text-text-muted mb-2">By {book.author} · {book.chapters.length} chapters</p>
            <p className="text-text-secondary leading-relaxed mb-4">
              {book.description}
            </p>
            <div className="flex items-center gap-4">
              <Link
                href={`/books/${book.slug}`}
                className="inline-flex items-center gap-1.5 text-sm font-medium text-accent hover:text-accent-hover transition-colors"
              >
                Explore
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
              </Link>
              <a
                href="#"
                className="inline-flex items-center gap-1.5 text-sm font-medium border border-border text-text-secondary rounded-md px-3 py-1 hover:border-accent hover:text-accent transition-colors"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                Download PDF
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
