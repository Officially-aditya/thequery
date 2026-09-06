"use client";

import type { ContentBlock, Source } from "@/lib/content-types";
import type { EditableContent } from "./admin-client";

export type ModelAccess = "proprietary" | "restricted" | "open_weights" | "open_source";

export interface ModelCatalogEntry {
  slug: string;
  name: string;
  developer: string;
  releaseDate: string | null;
  access: ModelAccess;
  comparisonData: Record<string, string>;
  sources: Source[];
  notes: string | null;
  verifiedAt: string;
}

const fieldClass = "w-full rounded-md border border-border bg-bg-primary px-3 py-2 text-sm text-text-primary outline-none focus:border-accent";

function metadataSlug(metadata: Record<string, unknown>, key: "modelA" | "modelB"): string {
  return typeof metadata[key] === "string" ? metadata[key] : "";
}

function uniqueSources(current: Source[], additions: Source[]): Source[] {
  return Array.from(
    new Map([...current, ...additions].map((source) => [source.url, source])).values(),
  );
}

function fillSpecBlocks(
  blocks: ContentBlock[],
  modelA: ModelCatalogEntry | undefined,
  modelB: ModelCatalogEntry | undefined,
  replaceA: boolean,
  replaceB: boolean,
): ContentBlock[] {
  return blocks.map((block) => {
    if (block.type !== "spec_table") return block;

    const columns = [
      replaceA && modelA ? modelA.name : block.columns[0] ?? "Model A",
      replaceB && modelB ? modelB.name : block.columns[1] ?? "Model B",
    ];
    const rows = block.rows.map((row) => {
      const label = row[0] ?? "";
      return [
        label,
        replaceA && modelA ? modelA.comparisonData[label] ?? "" : row[1] ?? "",
        replaceB && modelB ? modelB.comparisonData[label] ?? "" : row[2] ?? "",
      ];
    });

    return { ...block, columns, rows };
  });
}

export function applyComparisonModels(
  editing: EditableContent,
  models: ModelCatalogEntry[],
  nextModelASlug: string,
  nextModelBSlug: string,
  changedSide: "a" | "b",
): Partial<EditableContent> {
  const modelA = models.find((model) => model.slug === nextModelASlug);
  const modelB = models.find((model) => model.slug === nextModelBSlug);
  const replaceA = changedSide === "a" && Boolean(modelA);
  const replaceB = changedSide === "b" && Boolean(modelB);

  const additions = [
    ...(replaceA && modelA ? modelA.sources : []),
    ...(replaceB && modelB ? modelB.sources : []),
  ];

  const title = !editing.title.trim() && modelA && modelB
    ? `${modelA.name} vs ${modelB.name}`
    : editing.title;

  return {
    title,
    blocks: fillSpecBlocks(editing.blocks, modelA, modelB, replaceA, replaceB),
    sources: uniqueSources(editing.sources, additions),
    metadata: {
      ...editing.metadata,
      modelA: nextModelASlug,
      modelB: nextModelBSlug,
    },
  };
}

function accessLabel(access: ModelAccess): string {
  if (access === "open_source") return "open source";
  if (access === "open_weights") return "open weights";
  return access;
}

export default function ComparisonModelPicker({
  editing,
  models,
  loading,
  onChange,
}: {
  editing: EditableContent;
  models: ModelCatalogEntry[];
  loading: boolean;
  onChange: (next: Partial<EditableContent>) => void;
}) {
  const selectedA = metadataSlug(editing.metadata, "modelA");
  const selectedB = metadataSlug(editing.metadata, "modelB");
  const groups = Array.from(new Set(models.map((model) => model.developer)));

  function select(side: "a" | "b", slug: string) {
    const nextA = side === "a" ? slug : selectedA;
    const nextB = side === "b" ? slug : selectedB;
    onChange(applyComparisonModels(editing, models, nextA, nextB, side));
  }

  return (
    <section className="rounded-xl border border-border bg-bg-secondary p-4">
      <div className="mb-3">
        <h2 className="font-serif text-base font-semibold text-text-primary">Comparison models</h2>
        <p className="mt-1 text-xs leading-relaxed text-text-muted">
          Choose verified catalog models to populate matching specification, pricing, capability, and benchmark rows. The copied values remain editable in the blocks below.
        </p>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {(["a", "b"] as const).map((side) => {
          const value = side === "a" ? selectedA : selectedB;
          return (
            <label key={side} className="text-sm font-medium text-text-secondary">
              Model {side.toUpperCase()}
              <select
                className={`${fieldClass} mt-1`}
                value={value}
                onChange={(event) => select(side, event.target.value)}
                disabled={loading}
              >
                <option value="">{loading ? "Loading model catalog…" : "Custom / manual"}</option>
                {groups.map((developer) => (
                  <optgroup key={developer} label={developer}>
                    {models.filter((model) => model.developer === developer).map((model) => (
                      <option key={model.slug} value={model.slug}>
                        {model.name} · {accessLabel(model.access)}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </label>
          );
        })}
      </div>
      <p className="mt-3 text-xs text-text-muted">
        Catalog values are snapshots from cited vendor sources. Missing values stay blank; selecting a model never invents unsupported specs or benchmark scores.
      </p>
    </section>
  );
}
