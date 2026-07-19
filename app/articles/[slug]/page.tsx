import Link from "next/link";
import { getAllIssues, getIssueBySlug } from "@/lib/articles";
import { getAllTerms } from "@/lib/glossary";
import { notFound } from "next/navigation";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import ReadingProgress from "@/components/ReadingProgress";
import X402RealityCheck from "@/components/article/X402RealityCheck";
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
    openGraph: { title: issue.title, description: issue.summary, images: ["/opengraph-image"] },
  };
}

export default async function ArticlePage({ params }: Props) {
  const { slug } = await params;
  const issue = getIssueBySlug(slug);
  if (!issue) notFound();

  const glossaryTerms = getAllTerms().map((term) => ({
    name: term.name,
    slug: term.slug,
  }));
  const x402VisualizationAnchor =
    "The payment rail is becoming real. The market on top of it is not.";
  const hasX402Visualization =
    issue.slug === "x402-40-companies-agent-economy-demand-gap" &&
    issue.content.includes(x402VisualizationAnchor);
  const visualizationIndex = hasX402Visualization
    ? issue.content.indexOf(x402VisualizationAnchor) +
      x402VisualizationAnchor.length
    : -1;
  const contentBeforeVisualization = hasX402Visualization
    ? issue.content.slice(0, visualizationIndex)
    : issue.content;
  const contentAfterVisualization = hasX402Visualization
    ? issue.content.slice(visualizationIndex).trimStart()
    : "";
  // This article already has explicit glossary backlinks. Using the stateful
  // auto-linker across two MarkdownRenderer instances can produce different
  // server and client trees, so keep the split render deterministic.
  const renderedGlossaryTerms = hasX402Visualization ? [] : glossaryTerms;

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
    <>
      <ReadingProgress />
      <div data-reading-frame className="max-w-[720px] mx-auto px-4 py-12">
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

        <MarkdownRenderer
          content={contentBeforeVisualization}
          glossaryTerms={renderedGlossaryTerms}
        />
        {hasX402Visualization ? <X402RealityCheck /> : null}
        {contentAfterVisualization ? (
          <MarkdownRenderer
            content={contentAfterVisualization}
            glossaryTerms={renderedGlossaryTerms}
          />
        ) : null}
      </div>
    </>
  );
}
