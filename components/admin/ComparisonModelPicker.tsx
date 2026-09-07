"use client";

import { useState } from "react";
import type { ContentBlock, Source } from "@/lib/content-types";
import type { EditableContent } from "./admin-client";

export type ModelAccess = "proprietary" | "restricted" | "open_weights" | "open_source";

// The initial editor request only needs these fields for the dropdown.
export interface ModelCatalogEntry {
  slug: string;
  name: string;
  developer: string;
  access: ModelAccess;
}

interface ModelCatalogDetail extends ModelCatalogEntry {
  releaseDate: string | null;
  comparisonData: Record<string, string>;
  sources: Source[];
  notes: string | null;
  verifiedAt: string;
}

const modelDetailRequests = new Map<string, Promise<ModelCatalogDetail>>();
const fieldClass = "w-full rounded-md border border-border bg-bg-primary px-3 py-2 text-sm text-text-primary outline-none focus:border-accent";

function metadataSlug(metadata: Record<string, unknown>, key: "modelA" | "modelB"): string {
  return typeof metadata[key] === "string" ? metadata[key] : "";
}

function uniqueSources(current: Source[], additions: Source[]): Source[] {
  return Array.from(
    new Map([...current, ...additions].map((source) => [source.url, source])).values(),
  );
}

async function loadModelDetail(slug: string): Promise<ModelCatalogDetail> {
  let request = modelDetailRequests.get(slug);
  if (!request) {
    request = (async () => {
      const response = await fetch(`/api/admin/models?slug=${encodeURIComponent(slug)}`, { cache: "no-store" });
      const payload: unknown = await response.json().catch(() => ({}));
      if (!response.ok) {
        const record = payload && typeof payload === "object" && !Array.isArray(payload)
          ? payload as Record<string, unknown>
          : {};
        throw new Error(typeof record.error === "string" ? record.error : "Unable to load model details.");
      }
      return payload as ModelCatalogDetail;
    })();
    modelDetailRequests.set(slug, request);
  }

  try {
    return await request;
  } catch (error) {
    modelDetailRequests.delete(slug);
    throw error;
  }
}

function fillSpecBlocks(
  blocks: ContentBlock[],
  model: ModelCatalogDetail,
  changedSide: "a" | "b",
): ContentBlock[] {
  const columnIndex = changedSide === "a" ? 0 : 1;
  const valueIndex = changedSide === "a" ? 1 : 2;

  return blocks.map((block) => {
    if (block.type !== "spec_table") return block;

    const columns = [...block.columns];
    columns[columnIndex] = model.name;
    const rows = block.rows.map((row) => {
      const next = [...row];
      const label = next[0] ?? "";
      next[valueIndex] = model.comparisonData[label] ?? "";
      return next;
    });

    return { ...block, columns, rows };
  });
}

function selectionUpdate(
  editing: EditableContent,
  models: ModelCatalogEntry[],
  nextModelASlug: string,
  nextModelBSlug: string,
  changedSide: "a" | "b",
  detail?: ModelCatalogDetail,
): Partial<EditableContent> {
  const modelA = models.find((model) => model.slug === nextModelASlug);
  const modelB = models.find((model) => model.slug === nextModelBSlug);
  const title = !editing.title.trim() && modelA && modelB
    ? `${modelA.name} vs ${modelB.name}`
    : editing.title;

  return {
    title,
    blocks: detail ? fillSpecBlocks(editing.blocks, detail, changedSide) : editing.blocks,
    sources: detail ? uniqueSources(editing.sources, detail.sources) : editing.sources,
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
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");

  async function select(side: "a" | "b", slug: string) {
    const nextA = side === "a" ? slug : selectedA;
    const nextB = side === "b" ? slug : selectedB;
    setDetailError("");

    if (!slug) {
      onChange(selectionUpdate(editing, models, nextA, nextB, side));
      return;
    }

    setDetailLoading(true);
    try {
      const detail = await loadModelDetail(slug);
      onChange(selectionUpdate(editing, models, nextA, nextB, side, detail));
    } catch (requestError) {
      setDetailError(requestError instanceof Error ? requestError.message : "Unable to load model details.");
    } finally {
      setDetailLoading(false);
    }
  }

  return (
    <section className="rounded-xl border border-border bg-bg-secondary p-4">
      <div className="mb-3">
        <h2 className="font-serif text-base font-semibold text-text-primary">Comparison models</h2>
        <p className="mt-1 text-xs leading-relaxed text-text-muted">
          Choose verified catalog models to populate matching specification, pricing, capability, and benchmark rows. Full model data is fetched only for the model you select.
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
                onChange={(event) => void select(side, event.target.value)}
                disabled={loading || detailLoading}
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
      {detailLoading ? <p className="mt-3 text-xs text-text-muted">Loading selected model details…</p> : null}
      {detailError ? <p className="mt-3 text-xs text-red-600">{detailError}</p> : null}
      <p className="mt-3 text-xs text-text-muted">
        Catalog values are snapshots from cited vendor sources. Missing values stay blank; selecting a model never invents unsupported specs or benchmark scores.
      </p>
    </section>
  );
}
