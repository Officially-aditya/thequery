import Link from "next/link";
import { getAllIssues } from "@/lib/articles";
import { notFound } from "next/navigation";
import type { Metadata } from "next";

const SITE_URL = "https://www.thequery.in";
const PAGE_SIZE = 10;

type ArticlesPageProps = {
  searchParams: Promise<{ page?: string }>;
};

function getValidPage(pageParam: string | undefined, totalPages: number): number {
  if (pageParam === undefined) return 1;
  if (!/^[1-9]\d*$/.test(pageParam)) notFound();

  const page = Number(pageParam);
  if (!Number.isSafeInteger(page) || page > totalPages) notFound();
  return page;
}

export async function generateMetadata({ searchParams }: ArticlesPageProps): Promise<Metadata> {
  const totalPages = Math.max(1, Math.ceil(getAllIssues().length / PAGE_SIZE));
  const { page: pageParam } = await searchParams;
  const currentPage = getValidPage(pageParam, totalPages);
  const canonical = currentPage === 1
    ? `${SITE_URL}/articles`
    : `${SITE_URL}/articles?page=${currentPage}`;

  return {
    title: "Articles",
    description: "A curated weekly summary of the most important AI developments, research, and news.",
    alternates: { canonical },
    openGraph: {
      title: "Articles - TheQuery",
      description: "A weekly roundup of what actually matters in AI - no hype, just signal.",
      url: canonical,
      images: ["/opengraph-image"],
    },
  };
}

export default async function ArticlesPage({ searchParams }: ArticlesPageProps) {
  const issues = getAllIssues();
  const { page: pageParam } = await searchParams;
  const totalPages = Math.max(1, Math.ceil(issues.length / PAGE_SIZE));
  const currentPage = getValidPage(pageParam, totalPages);
  const pageIssues = issues.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  const pageHref = (page: number) => (page === 1 ? "/articles" : `/articles?page=${page}`);

  return (
    <div className="max-w-[960px] mx-auto px-4 py-12">
      <h1 className="font-serif text-3xl font-bold text-text-primary mb-2">Articles</h1>
      <p className="text-text-secondary mb-8">
        A curated summary of the most important AI developments each week.
      </p>

      {issues.length === 0 ? (
        <p className="text-sm text-text-muted text-center py-12">No articles yet. Check back soon!</p>
      ) : (
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
              <p className="text-sm text-text-secondary leading-relaxed">
                {issue.summary}
              </p>
            </Link>
          ))}
        </div>
      )}

      {totalPages > 1 && (
        <nav aria-label="Article pages" className="flex items-center justify-center gap-2 mt-10">
          {currentPage > 1 ? (
            <Link
              href={pageHref(currentPage - 1)}
              className="px-3 py-2 text-sm text-text-secondary border border-border rounded-md hover:border-accent hover:text-accent transition-colors"
            >
              Previous
            </Link>
          ) : (
            <span className="px-3 py-2 text-sm text-text-muted/50 border border-border/50 rounded-md">
              Previous
            </span>
          )}

          <div className="flex items-center gap-1" role="list">
            {Array.from({ length: totalPages }, (_, index) => index + 1).map((page) => (
              <Link
                key={page}
                href={pageHref(page)}
                aria-current={page === currentPage ? "page" : undefined}
                className={`min-w-9 px-2 py-2 text-sm text-center rounded-md border transition-colors ${
                  page === currentPage
                    ? "border-accent bg-accent text-white"
                    : "border-border text-text-secondary hover:border-accent hover:text-accent"
                }`}
              >
                {page}
              </Link>
            ))}
          </div>

          {currentPage < totalPages ? (
            <Link
              href={pageHref(currentPage + 1)}
              className="px-3 py-2 text-sm text-text-secondary border border-border rounded-md hover:border-accent hover:text-accent transition-colors"
            >
              Next
            </Link>
          ) : (
            <span className="px-3 py-2 text-sm text-text-muted/50 border border-border/50 rounded-md">
              Next
            </span>
          )}
        </nav>
      )}
    </div>
  );
}
