import Link from "next/link";
import { getIssueBySlug } from "@/lib/articles";
import { getGlossaryIndex } from "@/lib/glossary";
import { notFound } from "next/navigation";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import ReadingProgress from "@/components/ReadingProgress";
import GeminiLeaderboardChart from "@/components/article/GeminiLeaderboardChart";
import OpusLeaderboardChart from "@/components/article/OpusLeaderboardChart";
import X402RealityCheck from "@/components/article/X402RealityCheck";
import ClaudeSharedChatsPrivacy from "@/components/article/ClaudeSharedChatsPrivacy";
import Grok46Chart from "@/components/article/Grok46Chart";
import Qwen27BChart from "@/components/article/Qwen27BChart";
import TerminalBench4Chart from "@/components/article/TerminalBench4Chart";
import ArticleImageRail from "@/components/article/ArticleImageRail";
import ContentBlocksRenderer, { SourcesList } from "@/components/content/ContentBlocksRenderer";
import CoverImage from "@/components/content/CoverImage";
import PreferredSourceButton from "@/components/PreferredSourceButton";
import {
  AUTHOR,
  ORGANIZATION_ID,
  ORGANIZATION_LOGO,
  SITE_URL,
  authorJsonLd,
  createOpenGraphMetadata,
  createTwitterMetadata,
} from "@/lib/site";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ slug: string }>;
}

export const revalidate = 300;

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const issue = await getIssueBySlug(slug);
  if (!issue) return {};
  return {
    title: issue.title,
    description: issue.summary,
    openGraph: createOpenGraphMetadata({
      title: issue.title,
      description: issue.summary,
      url: `${SITE_URL}/articles/${issue.slug}`,
      type: "article",
      image: issue.coverImageUrl,
    }),
    twitter: createTwitterMetadata({
      title: issue.title,
      description: issue.summary,
      image: issue.coverImageUrl,
    }),
  };
}

export default async function ArticlePage({ params }: Props) {
  const { slug } = await params;
  const issue = await getIssueBySlug(slug);
  if (!issue) notFound();

  const glossaryTerms = await getGlossaryIndex();
  const visualizationConfig =
    issue.slug === "x402-40-companies-agent-economy-demand-gap"
      ? {
          anchor:
            "The payment rail is becoming real. The market on top of it is not.",
          component: <X402RealityCheck />,
          placement: "right-rail" as const,
        }
      : issue.slug === "gemini-36-flash-google-outside-top-ten"
        ? {
            anchor:
              "The launch improves the economics of Gemini. It does not move Google's intelligence ceiling.",
            component: <GeminiLeaderboardChart />,
            placement: "right-rail" as const,
          }
        : issue.slug === "claude-opus-5-fable-5-benchmark-reaction"
          ? {
              anchor:
                "The benchmarks were never the bottleneck. The timing was.",
              component: <OpusLeaderboardChart />,
              placement: "right-rail" as const,
            }
          : issue.slug === "claude-shared-chats-google-indexed-privacy"
            ? {
                anchor:
                  "What users already bring to a chat has changed faster than the interface has.",
                component: <ClaudeSharedChatsPrivacy />,
                placement: "right-rail" as const,
              }
            : issue.slug === "grok-4-6-index-cost-efficiency-benchmarks"
              ? {
                  anchor:
                    "A composite score works the way a report card GPA does.",
                  component: <Grok46Chart />,
                  placement: "right-rail" as const,
                }
              : issue.slug === "qwen-3-8-27b-opus-4-6-vision-benchmarks"
                ? {
                    anchor:
                      "The [computer vision](/glossary/computer-vision) table looks far more dramatic than the text one, and this is where the framing driving the Twitter hype actually comes from.",
                    component: <Qwen27BChart />,
                    placement: "right-rail" as const,
                  }
                : issue.slug === "terminal-bench-4-agent-not-model"
                  ? {
                      anchor:
                        "Here is the current Terminal-Bench 4.0 leaderboard as published by the maintainers:",
                      component: <TerminalBench4Chart />,
                      placement: "right-rail" as const,
                    }
                : issue.slug === "glm-5-3-flash-ox-alpha-free-model-benchmarks"
                  ? {
                      anchor:
                        "On August 26, Z.ai ended the guessing: Ox Alpha was an early version of GLM-5.3-Flash.",
                      component: (
                        <ArticleImageRail
                          src="/glm-53-flash-benchmarks.jpg"
                          alt="Six benchmark panels comparing GLM-5.3-Flash with GLM-5.2, DeepSeek-V4-Vision-Exp, Claude Opus 4.8, GPT-5.6 Terra, and Gemini 3.7 Flash"
                          width={1800}
                          height={1122}
                          caption="Coding and agentic benchmark scores published by Z.ai on August 26, 2026."
                        />
                      ),
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
  const renderedGlossaryTerms =
    hasEmbeddedVisualization || issue.manualGlossaryLinks ? [] : glossaryTerms;
  const hasRightRailVisualization =
    hasEmbeddedVisualization && visualizationConfig?.placement === "right-rail";
  const hasStructuredBlocks = issue.blocks.some((block) => block.type !== "markdown");

  const jsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Article",
        headline: issue.title,
        description: issue.summary,
        datePublished: issue.date,
        dateModified: issue.date,
        url: `${SITE_URL}/articles/${issue.slug}`,
        author: { ...authorJsonLd },
        publisher: {
          "@type": "Organization",
          "@id": ORGANIZATION_ID,
          name: "TheQuery",
          logo: {
            "@type": "ImageObject",
            url: ORGANIZATION_LOGO,
          },
        },
        mainEntityOfPage: {
          "@type": "WebPage",
          "@id": `${SITE_URL}/articles/${issue.slug}`,
        },
        inLanguage: "en",
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Home", item: SITE_URL },
          { "@type": "ListItem", position: 2, name: "Articles", item: `${SITE_URL}/articles` },
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
        className={`${hasRightRailVisualization ? "max-w-[1440px]" : "max-w-[720px]"} mx-auto px-4 py-12`}
      >
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
        <div
          className="mx-auto max-w-[720px]"
        >
          <Link href="/articles" className="text-sm text-text-muted hover:text-text-secondary transition-colors mb-6 inline-block">
            &larr; All Articles
          </Link>

          <h1 className="font-serif text-3xl font-bold text-text-primary mb-2">
            {issue.title}
          </h1>
          <p className="text-sm text-text-muted mb-8">
            By <Link href={AUTHOR.url} className="text-accent hover:text-accent-hover transition-colors">{AUTHOR.name}</Link>
            {" "}&middot; {new Date(issue.date).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
            {" "}&middot; <Link href="/about#editorial-standards" className="hover:text-text-secondary transition-colors">Editorial standards</Link>
          </p>
          <CoverImage src={issue.coverImageUrl} alt={issue.coverImageAlt} title={issue.title} />
        </div>

        {hasStructuredBlocks ? (
          <div className="mx-auto max-w-[720px]">
            <ContentBlocksRenderer
              blocks={issue.blocks}
              sources={issue.sources}
              glossaryTerms={issue.manualGlossaryLinks ? [] : glossaryTerms}
            />
          </div>
        ) : hasRightRailVisualization ? (
          <div className="relative mx-auto max-w-[720px] 2xl:max-w-[1440px]">
            <div className="mx-auto max-w-[720px]">
              <MarkdownRenderer
                content={contentBeforeVisualization}
                glossaryTerms={renderedGlossaryTerms}
              />
            </div>
            <aside
              className="article-visualization-rail w-full 2xl:absolute 2xl:left-[calc(50%+400px)] 2xl:top-0"
              aria-label="Article data visualization"
            >
              {visualizationConfig?.component}
            </aside>
            {contentAfterVisualization ? (
              <div className="mx-auto max-w-[720px]">
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
        {!hasStructuredBlocks ? <SourcesList sources={issue.sources} /> : null}
        <PreferredSourceButton className="mx-auto mt-12 max-w-[720px]" />
      </div>
    </>
  );
}
