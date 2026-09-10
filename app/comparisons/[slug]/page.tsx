import Link from "next/link";
import { getAllComparisons, getComparisonBySlug, getComparisonPairs } from "@/lib/comparisons";
import { getGlossaryIndex } from "@/lib/glossary";
import { canonicalComparisonSlug, resolveComparisonSlug } from "@/lib/model-comparison-route";
import { buildModelComparisonBlocks, modelComparisonSources } from "@/lib/model-comparison";
import { getModelOptions, getModelsBySlugs } from "@/lib/models";
import { notFound, permanentRedirect } from "next/navigation";
import ContentBlocksRenderer from "@/components/content/ContentBlocksRenderer";
import CoverImage from "@/components/content/CoverImage";
import {
  ORGANIZATION_ID,
  ORGANIZATION_LOGO,
  SITE_URL,
  authorJsonLd,
  createOpenGraphMetadata,
} from "@/lib/site";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ slug: string }>;
}

export const revalidate = false;

export async function generateStaticParams() {
  return (await getAllComparisons()).map(({ slug }) => ({ slug }));
}

function pairKey(modelA: string, modelB: string): string {
  return [modelA, modelB].sort().join("::");
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const comparison = await getComparisonBySlug(slug);
  if (comparison) {
    const canonicalUrl = `${SITE_URL}/comparisons/${comparison.slug}`;
    return {
      title: comparison.title,
      description: comparison.summary,
      alternates: { canonical: canonicalUrl },
      openGraph: createOpenGraphMetadata({
        title: comparison.title,
        description: comparison.summary,
        url: canonicalUrl,
        type: "article",
        image: comparison.coverImageUrl,
      }),
    };
  }

  const modelCatalog = await getModelOptions();
  const resolved = resolveComparisonSlug(slug, modelCatalog.map((model) => model.slug));
  if (!resolved) return {};

  const [modelASlug, modelBSlug] = resolved;
  const modelA = modelCatalog.find((model) => model.slug === modelASlug);
  const modelB = modelCatalog.find((model) => model.slug === modelBSlug);
  if (!modelA || !modelB) return {};

  const canonicalSlug = canonicalComparisonSlug(modelA.slug, modelB.slug);
  const canonicalUrl = `${SITE_URL}/comparisons/${canonicalSlug}`;
  const title = `${modelA.name} vs ${modelB.name}`;
  const description = `Compare ${modelA.name} and ${modelB.name} across specifications, pricing, capabilities, coding, reasoning, knowledge, and agentic benchmarks.`;

  return {
    title,
    description,
    alternates: { canonical: canonicalUrl },
    openGraph: createOpenGraphMetadata({
      title: `${title} - TheQuery`,
      description,
      url: canonicalUrl,
    }),
  };
}

export default async function ComparisonPage({ params }: Props) {
  const { slug } = await params;
  const comparison = await getComparisonBySlug(slug);

  if (comparison) {
    const [modelCatalog, comparisonPairs, glossaryTerms] = await Promise.all([
      getModelOptions(),
      getComparisonPairs(),
      getGlossaryIndex(),
    ]);
    const modelOptions = modelCatalog.map(({ slug: modelSlug, name, developer }) => ({
      slug: modelSlug,
      name,
      developer,
    }));

    const jsonLd = {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "TechArticle",
          headline: comparison.title,
          description: comparison.summary,
          datePublished: comparison.date,
          dateModified: comparison.date,
          url: `${SITE_URL}/comparisons/${comparison.slug}`,
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
          inLanguage: "en",
        },
        {
          "@type": "BreadcrumbList",
          itemListElement: [
            { "@type": "ListItem", position: 1, name: "Home", item: SITE_URL },
            { "@type": "ListItem", position: 2, name: "Comparisons", item: `${SITE_URL}/comparisons` },
            { "@type": "ListItem", position: 3, name: comparison.title },
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
        <Link href="/comparisons" className="text-sm text-text-muted hover:text-text-secondary transition-colors mb-6 inline-block">
          &larr; All Comparisons
        </Link>

        <h1 className="font-serif text-3xl font-bold text-text-primary mb-2">
          {comparison.title}
        </h1>
        <CoverImage src={comparison.coverImageUrl} alt={comparison.coverImageAlt} title={comparison.title} />

        <ContentBlocksRenderer
          blocks={comparison.blocks}
          sources={comparison.sources}
          glossaryTerms={glossaryTerms}
          comparisonPicker={comparison.modelA && comparison.modelB ? {
            models: modelOptions,
            comparisons: comparisonPairs,
            modelA: comparison.modelA,
            modelB: comparison.modelB,
          } : undefined}
        />
      </div>
    );
  }

  const [modelCatalog, comparisonPairs, glossaryTerms] = await Promise.all([
    getModelOptions(),
    getComparisonPairs(),
    getGlossaryIndex(),
  ]);
  const resolved = resolveComparisonSlug(slug, modelCatalog.map((model) => model.slug));
  if (!resolved) notFound();

  const [modelASlug, modelBSlug] = resolved;
  const authored = comparisonPairs.find((item) => pairKey(item.modelA, item.modelB) === pairKey(modelASlug, modelBSlug));
  if (authored) permanentRedirect(`/comparisons/${authored.slug}`);

  const canonicalSlug = canonicalComparisonSlug(modelASlug, modelBSlug);
  if (slug !== canonicalSlug) permanentRedirect(`/comparisons/${canonicalSlug}`);

  const selectedModels = await getModelsBySlugs([modelASlug, modelBSlug]);
  const modelA = selectedModels.find((model) => model.slug === modelASlug);
  const modelB = selectedModels.find((model) => model.slug === modelBSlug);
  if (!modelA || !modelB) notFound();

  const title = `${modelA.name} vs ${modelB.name}`;
  const description = `Compare ${modelA.name} and ${modelB.name} across specifications, pricing, capabilities, coding, reasoning, knowledge, and agentic benchmarks.`;
  const canonicalUrl = `${SITE_URL}/comparisons/${canonicalSlug}`;
  const blocks = buildModelComparisonBlocks(modelA, modelB);
  const sources = modelComparisonSources(modelA, modelB);
  const modelOptions = modelCatalog.map(({ slug: modelSlug, name, developer }) => ({
    slug: modelSlug,
    name,
    developer,
  }));

  const jsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebPage",
        name: title,
        description,
        url: canonicalUrl,
        isPartOf: { "@type": "WebSite", name: "TheQuery", url: SITE_URL },
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Home", item: SITE_URL },
          { "@type": "ListItem", position: 2, name: "Comparisons", item: `${SITE_URL}/comparisons` },
          { "@type": "ListItem", position: 3, name: title, item: canonicalUrl },
        ],
      },
    ],
  };

  return (
    <div className="mx-auto max-w-[720px] px-4 py-12">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <Link href="/comparisons" className="mb-6 inline-block text-sm text-text-muted transition-colors hover:text-text-secondary">
        &larr; All Comparisons
      </Link>

      <h1 className="mb-2 font-serif text-3xl font-bold text-text-primary">{title}</h1>

      <ContentBlocksRenderer
        blocks={blocks}
        sources={sources}
        glossaryTerms={glossaryTerms}
        comparisonPicker={{
          models: modelOptions,
          comparisons: comparisonPairs,
          modelA: modelA.slug,
          modelB: modelB.slug,
        }}
      />
    </div>
  );
}
