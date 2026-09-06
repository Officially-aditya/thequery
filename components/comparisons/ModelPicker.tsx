"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";

export type PublicModelOption = {
  slug: string;
  name: string;
  developer: string;
  access: "proprietary" | "restricted" | "open_weights" | "open_source";
};

export type ExistingComparisonPair = {
  modelA: string;
  modelB: string;
  slug: string;
};

const fieldClass = "w-full rounded-md border border-border bg-bg-primary px-3 py-2.5 text-sm text-text-primary outline-none focus:border-accent";

function pairKey(modelA: string, modelB: string): string {
  return [modelA, modelB].sort().join("::");
}

function accessLabel(access: PublicModelOption["access"]): string {
  if (access === "open_source") return "open source";
  if (access === "open_weights") return "open weights";
  return access;
}

export default function ModelPicker({
  models,
  comparisons,
  initialModelA = "",
  initialModelB = "",
}: {
  models: PublicModelOption[];
  comparisons: ExistingComparisonPair[];
  initialModelA?: string;
  initialModelB?: string;
}) {
  const router = useRouter();
  const [modelA, setModelA] = useState(initialModelA);
  const [modelB, setModelB] = useState(initialModelB);

  const groups = useMemo(
    () => Array.from(new Set(models.map((model) => model.developer))).sort((a, b) => a.localeCompare(b)),
    [models],
  );
  const comparisonByPair = useMemo(
    () => new Map(comparisons.map((comparison) => [pairKey(comparison.modelA, comparison.modelB), comparison.slug])),
    [comparisons],
  );

  const canCompare = Boolean(modelA && modelB && modelA !== modelB);

  function compare() {
    if (!canCompare) return;
    const existing = comparisonByPair.get(pairKey(modelA, modelB));
    if (existing) {
      router.push(`/comparisons/${existing}`);
      return;
    }
    const query = new URLSearchParams({ modelA, modelB });
    router.push(`/comparisons/compare?${query.toString()}`);
  }

  function options(selectedOther: string) {
    return groups.map((developer) => (
      <optgroup key={developer} label={developer}>
        {models
          .filter((model) => model.developer === developer)
          .map((model) => (
            <option key={model.slug} value={model.slug} disabled={model.slug === selectedOther}>
              {model.name} · {accessLabel(model.access)}
            </option>
          ))}
      </optgroup>
    ));
  }

  return (
    <section className="mb-8 rounded-xl border border-border bg-bg-secondary p-4 sm:p-5">
      <div className="mb-4">
        <h2 className="font-serif text-lg font-semibold text-text-primary">Compare models</h2>
        <p className="mt-1 text-sm text-text-secondary">
          Pick any two models from TheQuery&apos;s verified model database.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] sm:items-end">
        <label className="text-sm font-medium text-text-secondary">
          Model A
          <select className={`${fieldClass} mt-1`} value={modelA} onChange={(event) => setModelA(event.target.value)}>
            <option value="">Choose a model…</option>
            {options(modelB)}
          </select>
        </label>

        <label className="text-sm font-medium text-text-secondary">
          Model B
          <select className={`${fieldClass} mt-1`} value={modelB} onChange={(event) => setModelB(event.target.value)}>
            <option value="">Choose a model…</option>
            {options(modelA)}
          </select>
        </label>

        <button
          type="button"
          onClick={compare}
          disabled={!canCompare}
          className="rounded-md bg-accent px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
        >
          Compare
        </button>
      </div>
    </section>
  );
}
