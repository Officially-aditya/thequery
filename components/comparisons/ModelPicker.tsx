"use client";

import { useMemo } from "react";
import { useRouter } from "next/navigation";
import { canonicalComparisonSlug } from "@/lib/model-comparison-route";

export type PublicModelOption = {
  slug: string;
  name: string;
  developer: string;
};

export type ExistingComparisonPair = {
  modelA: string;
  modelB: string;
  slug: string;
};

function pairKey(modelA: string, modelB: string): string {
  return [modelA, modelB].sort().join("::");
}

export default function ModelHeaderSelect({
  side,
  models,
  comparisons,
  modelA,
  modelB,
}: {
  side: "a" | "b";
  models: PublicModelOption[];
  comparisons: ExistingComparisonPair[];
  modelA: string;
  modelB: string;
}) {
  const router = useRouter();
  const selected = side === "a" ? modelA : modelB;
  const other = side === "a" ? modelB : modelA;
  const groups = useMemo(
    () => Array.from(new Set(models.map((model) => model.developer))).sort((a, b) => a.localeCompare(b)),
    [models],
  );
  const comparisonByPair = useMemo(
    () => new Map(comparisons.map((comparison) => [pairKey(comparison.modelA, comparison.modelB), comparison.slug])),
    [comparisons],
  );

  function select(slug: string) {
    if (!slug || slug === other) return;
    const nextA = side === "a" ? slug : modelA;
    const nextB = side === "b" ? slug : modelB;
    if (!nextA || !nextB || nextA === nextB) return;

    const existing = comparisonByPair.get(pairKey(nextA, nextB));
    if (existing) {
      router.push(`/comparisons/${existing}`);
      return;
    }

    router.push(`/comparisons/${canonicalComparisonSlug(nextA, nextB)}`);
  }

  return (
    <select
      aria-label={side === "a" ? "Model A" : "Model B"}
      className="w-full min-w-0 cursor-pointer appearance-auto bg-transparent text-right text-xs font-semibold text-text-primary outline-none sm:text-sm"
      value={selected}
      onChange={(event) => select(event.target.value)}
    >
      {!selected ? <option value="">{side === "a" ? "Model A" : "Model B"}</option> : null}
      {groups.map((developer) => (
        <optgroup key={developer} label={developer}>
          {models
            .filter((model) => model.developer === developer)
            .map((model) => (
              <option key={model.slug} value={model.slug} disabled={model.slug === other}>
                {model.name}
              </option>
            ))}
        </optgroup>
      ))}
    </select>
  );
}
