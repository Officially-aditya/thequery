import Link from "next/link";
import { getGuideBySlug } from "@/lib/guides";
import { getAllTerms } from "@/lib/glossary";
import { notFound } from "next/navigation";
import ContentBlocksRenderer from "@/components/content/ContentBlocksRenderer";
import {
  AUTHOR,
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

export const dynamic = "force-dynamic";

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const guide = await getGuideBySlug(slug);
  if (!guide) return {};
  return {
    title: guide.title,
    description: guide.summary,
    openGraph: createOpenGraphMetadata({
      title: guide.title,
      description: guide.summary,
      url: `${SITE_URL}/guides/${guide.slug}`,
      type: "article",
    }),
  };
}

export default async function GuidePage({ params }: Props) {
  const { slug } = await params;
  const guide = await getGuideBySlug(slug);
  if (!guide) notFound();

  const jsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "TechArticle",
        headline: guide.title,
        description: guide.summary,
        datePublished: guide.date,
        dateModified: guide.date,
        url: `${SITE_URL}/guides/${guide.slug}`,
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
          { "@type": "ListItem", position: 2, name: "Guides", item: `${SITE_URL}/guides` },
          { "@type": "ListItem", position: 3, name: guide.title },
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
      <Link href="/guides" className="text-sm text-text-muted hover:text-text-secondary transition-colors mb-6 inline-block">
        &larr; All Guides
      </Link>

      <h1 className="font-serif text-3xl font-bold text-text-primary mb-2">
        {guide.title}
      </h1>
      <p className="text-sm text-text-muted mb-8">
        By <Link href={AUTHOR.url} className="text-accent hover:text-accent-hover transition-colors">{AUTHOR.name}</Link>
        {" "}&middot; {new Date(guide.date).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
        {" "}&middot; <Link href="/about#editorial-standards" className="hover:text-text-secondary transition-colors">Editorial standards</Link>
      </p>

      <ContentBlocksRenderer
        blocks={guide.blocks}
        sources={guide.sources}
        glossaryTerms={(await getAllTerms()).map((term) => ({ name: term.name, slug: term.slug }))}
      />
    </div>
  );
}
