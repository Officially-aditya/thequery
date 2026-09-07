import { getComparisonPairs } from "@/lib/comparisons";
import { canonicalComparisonSlug } from "@/lib/model-comparison-route";
import { getModelOptions } from "@/lib/models";
import { notFound, permanentRedirect } from "next/navigation";

type Props = {
  searchParams: Promise<{ modelA?: string; modelB?: string }>;
};

function pairKey(modelA: string, modelB: string): string {
  return [modelA, modelB].sort().join("::");
}

export default async function LegacyDatabaseComparisonPage({ searchParams }: Props) {
  const { modelA: modelASlug, modelB: modelBSlug } = await searchParams;
  if (!modelASlug || !modelBSlug || modelASlug === modelBSlug) notFound();

  const [modelCatalog, comparisonPairs] = await Promise.all([
    getModelOptions(),
    getComparisonPairs(),
  ]);
  const known = new Set(modelCatalog.map((model) => model.slug));
  if (!known.has(modelASlug) || !known.has(modelBSlug)) notFound();

  const authored = comparisonPairs.find((item) => pairKey(item.modelA, item.modelB) === pairKey(modelASlug, modelBSlug));
  if (authored) permanentRedirect(`/comparisons/${authored.slug}`);

  permanentRedirect(`/comparisons/${canonicalComparisonSlug(modelASlug, modelBSlug)}`);
}
