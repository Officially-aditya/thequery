import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Page Not Found",
  description: "The page you are looking for does not exist on TheQuery.",
  robots: {
    index: false,
    follow: true,
  },
};

export default function NotFound() {
  return (
    <section
      aria-labelledby="not-found-title"
      className="min-h-[calc(100svh-3.5rem)] flex items-center px-4 py-16"
    >
      <div className="w-full max-w-[960px] mx-auto">
        <div className="grid gap-10 lg:grid-cols-[1fr_0.8fr] lg:items-center">
          <div>
            <p className="font-mono text-sm tracking-[0.18em] text-accent uppercase mb-5">
              &gt;_ 404 / signal lost
            </p>
            <h1
              id="not-found-title"
              className="font-serif text-4xl sm:text-5xl font-bold tracking-tight text-text-primary mb-5"
            >
              This page isn&apos;t in the corpus.
            </h1>
            <p className="max-w-[560px] text-lg leading-relaxed text-text-secondary mb-8">
              The link may be outdated, the slug may have changed, or this idea has not
              made it into TheQuery yet.
            </p>

            <div className="flex flex-wrap gap-3">
              <Link
                href="/"
                className="inline-flex items-center rounded-md bg-accent px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-accent-hover"
              >
                Return home
              </Link>
              <Link
                href="/glossary"
                className="inline-flex items-center rounded-md border border-border px-4 py-2.5 text-sm font-medium text-text-secondary transition-colors hover:border-accent hover:text-accent"
              >
                Browse the glossary
              </Link>
              <Link
                href="/articles"
                className="inline-flex items-center rounded-md border border-border px-4 py-2.5 text-sm font-medium text-text-secondary transition-colors hover:border-accent hover:text-accent"
              >
                Read the articles
              </Link>
            </div>
          </div>

          <div className="overflow-hidden rounded-xl border border-border bg-bg-secondary shadow-sm">
            <div className="flex items-center gap-2 border-b border-border px-4 py-3">
              <span aria-hidden="true" className="h-2.5 w-2.5 rounded-full bg-[#ef6a62]" />
              <span aria-hidden="true" className="h-2.5 w-2.5 rounded-full bg-[#e5b567]" />
              <span aria-hidden="true" className="h-2.5 w-2.5 rounded-full bg-[#65b87a]" />
              <span className="ml-2 font-mono text-xs text-text-muted">thequery.in</span>
            </div>
            <div className="px-5 py-6 font-mono text-sm leading-8">
              <p className="text-text-muted">$ locate requested_page</p>
              <p className="text-accent">searching...</p>
              <p className="text-text-primary">404: no matching document</p>
              <p className="mt-3 text-text-muted">
                try: <Link href="/glossary" className="text-accent hover:text-accent-hover">/glossary</Link>
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
