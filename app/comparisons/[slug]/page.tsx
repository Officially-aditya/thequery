import Link from "next/link";
import { getAllComparisons, getComparisonBySlug } from "@/lib/comparisons";
import { getGlossaryIndex } from "@/lib/glossary";
import { getModels } from "@/lib/models";
import { notFound } from "next/navigation";
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

export const revalidate = 300;

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const comparison = await getComparisonBySlug(slug);
  if (!comparison) return {};
  return {
    title: comparison.title,
    description: comparison.summary,
    openGraph: createOpenGraphMetadata({
      title: comparison.title,
      description: comparison.summary,
      url: `${SITE_URL}/comparisons/${comparison.slug}`,
      type: "article",
      image: comparison.coverImageUrl,
    }),
  };
}

export default async function ComparisonPage({ params }: Props) {
  const { slug } = await params;
  const comparison = await getComparisonBySlug(slug);
  if (!comparison) notFound();

  const [models, comparisons, glossaryTerms] = await Promise.all([
    getModels(),
    getAllComparisons(),
    getGlossaryIndex(),
  ]);
  const modelOptions = models.map(({ slug: modelSlug, name, developer, access }) => ({
    slug: modelSlug,
    name,
    developer,
    access,
  }));
  const comparisonPairs = comparisons.flatMap((item) => item.modelA && item.modelB
    ? [{ modelA: item.modelA, modelB: item.modelB, slug: item.slug }]
    : []);

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
