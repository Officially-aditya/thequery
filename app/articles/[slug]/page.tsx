import Link from "next/link";
import { getAllIssues, getIssueBySlug } from "@/lib/articles";
import { getAllTerms } from "@/lib/glossary";
import { notFound } from "next/navigation";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  return getAllIssues().map((i) => ({ slug: i.slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const issue = getIssueBySlug(slug);
  if (!issue) return {};
  return {
    title: issue.title,
    description: issue.summary,
    openGraph: { title: issue.title, description: issue.summary },
  };
}

export default async function ArticlePage({ params }: Props) {
  const { slug } = await params;
  const issue = getIssueBySlug(slug);
  if (!issue) notFound();

  return (
    <div className="max-w-[720px] mx-auto px-4 py-12">
      <Link href="/articles" className="text-sm text-text-muted hover:text-text-secondary transition-colors mb-6 inline-block">
        &larr; All Articles
      </Link>

      <h1 className="font-serif text-3xl font-bold text-text-primary mb-8">
        {issue.title}
      </h1>

      <MarkdownRenderer content={issue.content} glossaryTerms={getAllTerms().map((t) => ({ name: t.name, slug: t.slug }))} />
    </div>
  );
}
