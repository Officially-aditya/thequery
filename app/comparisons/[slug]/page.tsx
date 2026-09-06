import Link from "next/link";
import { getComparisonBySlug } from "@/lib/comparisons";
import { getGlossaryIndex } from "@/lib/glossary";
import { notFound } from "next/navigation";
import ContentBlocksRenderer, { SpecColumns } from "@/components/content/ContentBlocksRenderer";
import CoverImage from "@/components/content/CoverImage";
import type { SpecTableBlock } from "@/lib/content-types";
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

  const specTables = comparison.blocks.filter(
    (block): block is SpecTableBlock => block.type === "spec_table",
  );
  const stickyModels =
    specTables.length > 0 && specTables[0].columns.length >= 2
      ? [specTables[0].columns[0], specTables[0].columns[1]]
      : comparison.title.split(" vs ");

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
      {stickyModels.length >= 2 ? (
        <div className="sticky top-14 z-30 -mx-4 border-y border-border bg-bg-primary/95 px-4 backdrop-blur-md">
          <table className="w-full table-fixed border-collapse">
            <SpecColumns />
            <thead>
              <tr>
                <th className="py-2.5" aria-hidden="true" />
                <th scope="col" className="px-2 py-2.5 text-center text-sm font-semibold text-text-primary">
                  {stickyModels[0]}
                </th>
                <th scope="col" className="px-2 py-2.5 text-center text-sm font-semibold text-text-primary">
                  {stickyModels[1]}
                </th>
              </tr>
            </thead>
          </table>
        </div>
      ) : null}
      <CoverImage src={comparison.coverImageUrl} alt={comparison.coverImageAlt} title={comparison.title} />

      <ContentBlocksRenderer
        blocks={comparison.blocks}
        sources={comparison.sources}
        glossaryTerms={await getGlossaryIndex()}
      />
    </div>
  );
}
