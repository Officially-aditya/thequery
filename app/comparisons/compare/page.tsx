import Link from "next/link";
import { notFound } from "next/navigation";
import ContentBlocksRenderer from "@/components/content/ContentBlocksRenderer";
import { getAllComparisons } from "@/lib/comparisons";
import { getGlossaryIndex } from "@/lib/glossary";
import { buildModelComparisonBlocks, modelComparisonSources } from "@/lib/model-comparison";
import { getModels } from "@/lib/models";
import { createOpenGraphMetadata, SITE_URL } from "@/lib/site";
import type { Metadata } from "next";

type Props = {
  searchParams: Promise<{ modelA?: string; modelB?: string }>;
};

export const revalidate = 300;

export async function generateMetadata({ searchParams }: Props): Promise<Metadata> {
  const { modelA: modelASlug, modelB: modelBSlug } = await searchParams;
  if (!modelASlug || !modelBSlug) return {};
  const models = await getModels();
  const modelA = models.find((model) => model.slug === modelASlug);
  const modelB = models.find((model) => model.slug === modelBSlug);
  if (!modelA || !modelB) return {};
  const title = `${modelA.name} vs ${modelB.name}`;
  const description = `Compare ${modelA.name} and ${modelB.name} across specifications, pricing, capabilities, coding, reasoning, knowledge, and agentic benchmarks.`;
  return {
    title,
    description,
    openGraph: createOpenGraphMetadata({
      title: `${title} - TheQuery`,
      description,
      url: `${SITE_URL}/comparisons/compare?modelA=${encodeURIComponent(modelASlug)}&modelB=${encodeURIComponent(modelBSlug)}`,
    }),
  };
}

export default async function DatabaseComparisonPage({ searchParams }: Props) {
  const { modelA: modelASlug, modelB: modelBSlug } = await searchParams;
  if (!modelASlug || !modelBSlug || modelASlug === modelBSlug) notFound();

  const [models, comparisons, glossaryTerms] = await Promise.all([
    getModels(),
    getAllComparisons(),
    getGlossaryIndex(),
  ]);
  const modelA = models.find((model) => model.slug === modelASlug);
  const modelB = models.find((model) => model.slug === modelBSlug);
  if (!modelA || !modelB) notFound();

  const title = `${modelA.name} vs ${modelB.name}`;
  const blocks = buildModelComparisonBlocks(modelA, modelB);
  const sources = modelComparisonSources(modelA, modelB);
  const modelOptions = models.map(({ slug, name, developer, access }) => ({ slug, name, developer, access }));
  const comparisonPairs = comparisons.flatMap((comparison) => comparison.modelA && comparison.modelB
    ? [{ modelA: comparison.modelA, modelB: comparison.modelB, slug: comparison.slug }]
    : []);

  return (
    <div className="mx-auto max-w-[720px] px-4 py-12">
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
