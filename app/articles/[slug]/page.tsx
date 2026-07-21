import Link from "next/link";
import { getAllIssues, getIssueBySlug } from "@/lib/articles";
import { getAllTerms } from "@/lib/glossary";
import { notFound } from "next/navigation";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import ReadingProgress from "@/components/ReadingProgress";
import GeminiLeaderboardChart from "@/components/article/GeminiLeaderboardChart";
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
  const visualizationConfig =
    issue.slug === "x402-40-companies-agent-economy-demand-gap"
      ? {
          anchor:
            "The payment rail is becoming real. The market on top of it is not.",
          component: <X402RealityCheck />,
          placement: "inline" as const,
        }
      : issue.slug === "gemini-36-flash-google-outside-top-ten"
        ? {
            anchor:
              "The launch improves the economics of Gemini. It does not move Google's intelligence ceiling.",
            component: <GeminiLeaderboardChart />,
            placement: "right-rail" as const,
          }
        : null;
  const visualizationAnchor = visualizationConfig?.anchor ?? "";
  const hasEmbeddedVisualization =
    visualizationConfig !== null &&
    issue.content.includes(visualizationAnchor);
  const visualizationIndex = hasEmbeddedVisualization
    ? issue.content.indexOf(visualizationAnchor) + visualizationAnchor.length
    : -1;
  const contentBeforeVisualization = hasEmbeddedVisualization
    ? issue.content.slice(0, visualizationIndex)
    : issue.content;
  const contentAfterVisualization = hasEmbeddedVisualization
    ? issue.content.slice(visualizationIndex).trimStart()
    : "";
  // This article already has explicit glossary backlinks. Using the stateful
  // auto-linker across two MarkdownRenderer instances can produce different
  // server and client trees, so keep the split render deterministic.
  const renderedGlossaryTerms = hasEmbeddedVisualization ? [] : glossaryTerms;
  const hasRightRailVisualization =
    hasEmbeddedVisualization && visualizationConfig?.placement === "right-rail";

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
      <div
        data-reading-frame
        className={`${hasRightRailVisualization ? "max-w-[1280px]" : "max-w-[720px]"} mx-auto px-4 py-12`}
      >
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
        <div
          className={
            hasRightRailVisualization
              ? "mx-auto max-w-[720px] xl:mx-0 xl:max-w-[680px]"
              : undefined
          }
        >
          <Link href="/articles" className="text-sm text-text-muted hover:text-text-secondary transition-colors mb-6 inline-block">
            &larr; All Articles
          </Link>

          <h1 className="font-serif text-3xl font-bold text-text-primary mb-2">
            {issue.title}
          </h1>
          <p className="text-sm text-text-muted mb-8">
            By Addy &middot; {new Date(issue.date).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
          </p>
        </div>

        {hasRightRailVisualization ? (
          <div className="mx-auto max-w-[720px] xl:grid xl:max-w-none xl:grid-cols-[minmax(0,680px)_minmax(0,520px)] xl:gap-x-12">
            <div className="xl:col-start-1 xl:row-start-1">
              <MarkdownRenderer
                content={contentBeforeVisualization}
                glossaryTerms={renderedGlossaryTerms}
              />
            </div>
            <aside
              className="xl:col-start-2 xl:row-start-2"
              aria-label="Article data visualization"
            >
              {visualizationConfig?.component}
            </aside>
            {contentAfterVisualization ? (
              <div className="xl:col-start-1 xl:row-start-2">
                <MarkdownRenderer
                  content={contentAfterVisualization}
                  glossaryTerms={renderedGlossaryTerms}
                />
              </div>
            ) : null}
          </div>
        ) : (
          <>
            <MarkdownRenderer
              content={contentBeforeVisualization}
              glossaryTerms={renderedGlossaryTerms}
            />
            {hasEmbeddedVisualization ? visualizationConfig?.component : null}
            {contentAfterVisualization ? (
              <MarkdownRenderer
                content={contentAfterVisualization}
                glossaryTerms={renderedGlossaryTerms}
              />
            ) : null}
          </>
        )}
      </div>
    </>
  );
}
