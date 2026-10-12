import { redirect } from "next/navigation";
import { getAllIssues } from "@/lib/articles";
import ArticlePagination, { PAGE_SIZE } from "@/components/articles/ArticlePagination";
import { createOpenGraphMetadata, SITE_URL } from "@/lib/site";
import type { Metadata } from "next";

interface Props {
  searchParams: Promise<{ page?: string | string[] }>;
}

export const revalidate = false;

async function getArchivePage(searchParams: Props["searchParams"]) {
  const [issues, params] = await Promise.all([getAllIssues(), searchParams]);
  const totalPages = Math.max(1, Math.ceil(issues.length / PAGE_SIZE));
  const rawPage = params.page;
  const parsedPage = typeof rawPage === "string" && /^[1-9]\d*$/.test(rawPage) ? Number(rawPage) : 1;
  const currentPage = Number.isSafeInteger(parsedPage) && parsedPage <= totalPages ? parsedPage : 1;

  if (rawPage !== undefined && currentPage === 1) redirect("/articles");

  return {
    issues,
    currentPage,
    href: currentPage === 1 ? "/articles" : `/articles?page=${currentPage}`,
  };
}

export async function generateMetadata({ searchParams }: Props): Promise<Metadata> {
  const { currentPage, href } = await getArchivePage(searchParams);
  const title = `AI News & Analysis${currentPage > 1 ? ` - Page ${currentPage}` : ""}`;
  const description = "Read AI news and analysis of new models, research papers, benchmarks, and developer tools, with sources and clear technical context from TheQuery.";
  const canonical = `${SITE_URL}${href}`;

  return {
    title,
    description,
    alternates: { canonical },
    openGraph: createOpenGraphMetadata({
      title: `${title} - TheQuery`,
      description,
      url: canonical,
    }),
  };
}

export default async function ArticlesPage({ searchParams }: Props) {
  const { issues, currentPage } = await getArchivePage(searchParams);

  return (
    <div className="max-w-[960px] mx-auto px-4 py-12">
      <h1 className="font-serif text-3xl font-bold text-text-primary mb-2">Articles</h1>
      <p className="text-text-secondary mb-8">
        AI news and analysis covering new models, research, benchmarks, and developer tools.
      </p>
      <ArticlePagination issues={issues} currentPage={currentPage} />
    </div>
  );
}
