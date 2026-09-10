"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import type { ArticleSummary } from "@/lib/articles";

const PAGE_SIZE = 10;

export default function ArticlePagination({ issues }: { issues: ArticleSummary[] }) {
  const searchParams = useSearchParams();
  const totalPages = Math.max(1, Math.ceil(issues.length / PAGE_SIZE));
  const rawPage = searchParams.get("page");
  const parsedPage = rawPage && /^[1-9]\d*$/.test(rawPage) ? Number(rawPage) : 1;
  const currentPage = Number.isSafeInteger(parsedPage) && parsedPage <= totalPages ? parsedPage : 1;
  const pageIssues = issues.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);
  const pageHref = (page: number) => (page === 1 ? "/articles" : `/articles?page=${page}`);

  if (issues.length === 0) {
    return <p className="text-sm text-text-muted text-center py-12">No articles yet. Check back soon!</p>;
  }

  return (
    <>
      <div className="space-y-4">
        {pageIssues.map((issue) => (
          <Link
            key={issue.slug}
            href={`/articles/${issue.slug}`}
            className="block p-5 border border-border rounded-lg hover:border-accent transition-colors group"
          >
            <h2 className="font-serif text-lg font-semibold text-text-primary group-hover:text-accent transition-colors mb-1">
              {issue.title}
            </h2>
            <p className="text-xs text-text-muted mb-2">
              {new Date(issue.date).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
            </p>
            <p className="text-sm text-text-secondary leading-relaxed">{issue.summary}</p>
          </Link>
        ))}
      </div>

      {totalPages > 1 && (
        <nav aria-label="Article pages" className="flex items-center justify-center gap-2 mt-10">
          {currentPage > 1 ? (
            <Link href={pageHref(currentPage - 1)} className="px-3 py-2 text-sm text-text-secondary border border-border rounded-md hover:border-accent hover:text-accent transition-colors">Previous</Link>
          ) : (
            <span className="px-3 py-2 text-sm text-text-muted/50 border border-border/50 rounded-md">Previous</span>
          )}

          <div className="flex items-center gap-1" role="list">
            {Array.from({ length: totalPages }, (_, index) => index + 1).map((page) => (
              <Link
                key={page}
                href={pageHref(page)}
                aria-current={page === currentPage ? "page" : undefined}
                className={`min-w-9 px-2 py-2 text-sm text-center rounded-md border transition-colors ${page === currentPage ? "border-accent bg-accent text-white" : "border-border text-text-secondary hover:border-accent hover:text-accent"}`}
              >
                {page}
              </Link>
            ))}
          </div>

          {currentPage < totalPages ? (
            <Link href={pageHref(currentPage + 1)} className="px-3 py-2 text-sm text-text-secondary border border-border rounded-md hover:border-accent hover:text-accent transition-colors">Next</Link>
          ) : (
            <span className="px-3 py-2 text-sm text-text-muted/50 border border-border/50 rounded-md">Next</span>
          )}
        </nav>
      )}
    </>
  );
}
