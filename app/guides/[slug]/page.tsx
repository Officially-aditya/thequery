import Link from "next/link";
import { getAllGuides, getGuideBySlug } from "@/lib/guides";
import { getAllTerms } from "@/lib/glossary";
import { notFound } from "next/navigation";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  return getAllGuides().map((g) => ({ slug: g.slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const guide = getGuideBySlug(slug);
  if (!guide) return {};
  return {
    title: guide.title,
    description: guide.summary,
    openGraph: { title: guide.title, description: guide.summary },
  };
}

export default async function GuidePage({ params }: Props) {
  const { slug } = await params;
  const guide = getGuideBySlug(slug);
  if (!guide) notFound();

  return (
    <div className="max-w-[720px] mx-auto px-4 py-12">
      <Link href="/guides" className="text-sm text-text-muted hover:text-text-secondary transition-colors mb-6 inline-block">
        &larr; All Guides
      </Link>

      <h1 className="font-serif text-3xl font-bold text-text-primary mb-2">
        {guide.title}
      </h1>
      <p className="text-sm text-text-muted mb-8">
        {new Date(guide.date).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
      </p>

      <MarkdownRenderer content={guide.content} glossaryTerms={getAllTerms().map((t) => ({ name: t.name, slug: t.slug }))} />
    </div>
  );
}
