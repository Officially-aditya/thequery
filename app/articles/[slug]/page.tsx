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

  const jsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Article",
        headline: issue.title,
        description: issue.summary,
        datePublished: issue.date,
        url: `https://www.thequery.in/articles/${issue.slug}`,
        author: {
          "@type": "Person",
          name: "Addy",
          url: "https://www.thequery.in/about",
        },
        publisher: {
          "@type": "Organization",
          "@id": "https://www.thequery.in/#organization",
          name: "TheQuery",
        },
        mainEntityOfPage: {
          "@type": "WebPage",
          "@id": `https://www.thequery.in/articles/${issue.slug}`,
        },
        inLanguage: "en",
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Home", item: "https://www.thequery.in" },
          { "@type": "ListItem", position: 2, name: "Articles", item: "https://www.thequery.in/articles" },
          { "@type": "ListItem", position: 3, name: issue.title },
        ],
      },
    ],
  };

  return (
    <div className="max-w-[720px] mx-auto px-4 py-12">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <Link href="/articles" className="text-sm text-text-muted hover:text-text-secondary transition-colors mb-6 inline-block">
        &larr; All Articles
      </Link>

      <h1 className="font-serif text-3xl font-bold text-text-primary mb-2">
        {issue.title}
      </h1>
      <p className="text-sm text-text-muted mb-8">
        By Addy &middot; {new Date(issue.date).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
      </p>

      <MarkdownRenderer content={issue.content} glossaryTerms={getAllTerms().map((t) => ({ name: t.name, slug: t.slug }))} />
    </div>
  );
}
