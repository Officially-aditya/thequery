"use client";

import { useEffect, useState } from "react";
import type { ChartBlock, ContentBlock } from "@/lib/content-types";

const fieldClass = "w-full rounded-md border border-border bg-bg-primary px-3 py-2 text-sm text-text-primary outline-none focus:border-accent";

function blockId(type: string): string {
  return `${type}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

function ChartDataEditor({ block, onChange }: { block: ChartBlock; onChange: (data: ChartBlock["data"]) => void }) {
  const [raw, setRaw] = useState(() => JSON.stringify(block.data, null, 2));
  const [error, setError] = useState("");

  useEffect(() => {
    setRaw(JSON.stringify(block.data, null, 2));
  }, [block.id, block.data]);

  function validate(value: string) {
    setRaw(value);
    try {
      const parsed: unknown = JSON.parse(value);
      if (!Array.isArray(parsed) || parsed.some((row) => !row || typeof row !== "object" || Array.isArray(row))) {
        throw new Error("Use an array of objects.");
      }
      onChange(parsed as ChartBlock["data"]);
      setError("");
    } catch (parseError) {
      setError(parseError instanceof Error ? parseError.message : "Enter valid JSON.");
    }
  }

  return (
    <label className="block text-sm font-medium text-text-secondary">
      Data
      <textarea
        className={`${fieldClass} mt-1 min-h-36 font-mono text-xs`}
        value={raw}
        onChange={(event) => validate(event.target.value)}
        spellCheck={false}
        aria-invalid={Boolean(error)}
      />
      <span className="mt-1 block text-xs text-text-muted">Use a <code>label</code> field plus one or more numeric series, for example <code>{`[{"label":"Model A","score":62}]`}</code>.</span>
      {error ? <span className="mt-1 block text-xs text-red-600">{error}</span> : null}
    </label>
  );
}

export default function EditorialBlocksEditor({ blocks, onChange }: { blocks: ContentBlock[]; onChange: (blocks: ContentBlock[]) => void }) {
  function update(index: number, nextBlock: ContentBlock) {
    onChange(blocks.map((block, blockIndex) => blockIndex === index ? nextBlock : block));
  }

  function move(index: number, direction: -1 | 1) {
    const nextIndex = index + direction;
    if (nextIndex < 0 || nextIndex >= blocks.length) return;
    const next = [...blocks];
    [next[index], next[nextIndex]] = [next[nextIndex], next[index]];
    onChange(next);
  }

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="font-serif text-lg font-semibold text-text-primary">Story blocks</h2>
          <p className="mt-1 text-sm text-text-muted">Compose the page in order. Markdown handles prose; tables and charts stay structured and editable.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button type="button" onClick={() => onChange([...blocks, { id: blockId("markdown"), type: "markdown", content: "" }])} className="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent">Add text</button>
          <button type="button" onClick={() => onChange([...blocks, { id: blockId("table"), type: "comparison_table", title: "", columns: ["Option", "Details"], rows: [["", ""]] }])} className="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent">Add comparison table</button>
          <button type="button" onClick={() => onChange([...blocks, { id: blockId("chart"), type: "chart", title: "New chart", chartType: "bar", data: [{ label: "Example", value: 0 }] }])} className="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent">Add chart</button>
        </div>
      </div>

      {blocks.map((block, index) => (
        <article key={block.id} className="rounded-xl border border-border bg-bg-secondary p-4">
          <div className="mb-4 flex items-center justify-between gap-3">
            <span className="rounded-full bg-bg-primary px-2.5 py-1 text-xs font-medium capitalize text-text-secondary">{block.type.replace("_", " ")}</span>
            <div className="flex items-center gap-2 text-xs">
              <button type="button" onClick={() => move(index, -1)} disabled={index === 0} className="text-text-secondary hover:text-accent disabled:opacity-30">Move up</button>
              <button type="button" onClick={() => move(index, 1)} disabled={index === blocks.length - 1} className="text-text-secondary hover:text-accent disabled:opacity-30">Move down</button>
              <button type="button" onClick={() => onChange(blocks.filter((_, blockIndex) => blockIndex !== index))} className="text-red-600 hover:text-red-700">Remove</button>
            </div>
          </div>

          {block.type === "markdown" ? (
            <textarea
              className={`${fieldClass} min-h-72 font-mono text-xs leading-6`}
              value={block.content}
              onChange={(event) => update(index, { ...block, content: event.target.value })}
              placeholder="Write in Markdown…"
              spellCheck={false}
            />
          ) : null}

          {block.type === "comparison_table" ? (
            <div className="grid gap-3">
              <input className={fieldClass} value={block.title ?? ""} onChange={(event) => update(index, { ...block, title: event.target.value })} placeholder="Table title (optional)" />
              <input className={fieldClass} value={block.caption ?? ""} onChange={(event) => update(index, { ...block, caption: event.target.value })} placeholder="Caption (optional)" />
              <label className="text-sm font-medium text-text-secondary">Columns (separate with <code>|</code>)
                <input className={`${fieldClass} mt-1`} value={block.columns.join(" | ")} onChange={(event) => update(index, { ...block, columns: event.target.value.split("|").map((column) => column.trim()).filter(Boolean) })} placeholder="Model | Context | Score" />
              </label>
              <label className="text-sm font-medium text-text-secondary">Rows (one row per line; cells use <code>|</code>)
                <textarea className={`${fieldClass} mt-1 min-h-32 font-mono text-xs`} value={block.rows.map((row) => row.join(" | ")).join("\n")} onChange={(event) => update(index, { ...block, rows: event.target.value.split("\n").map((row) => row.split("|").map((cell) => cell.trim())).filter((row) => row.some(Boolean)) })} placeholder="GPT-5 | 128k | 62\nModel B | 64k | 58" />
              </label>
              <input className={fieldClass} value={block.sourceNote ?? ""} onChange={(event) => update(index, { ...block, sourceNote: event.target.value })} placeholder="Source note (optional)" />
            </div>
          ) : null}

          {block.type === "chart" ? (
            <div className="grid gap-3">
              <input className={fieldClass} value={block.title} onChange={(event) => update(index, { ...block, title: event.target.value })} placeholder="Chart title" />
              <textarea className={`${fieldClass} min-h-20`} value={block.description ?? ""} onChange={(event) => update(index, { ...block, description: event.target.value })} placeholder="Explain what the chart shows (optional)" />
              <label className="text-sm font-medium text-text-secondary">Chart type
                <select className={`${fieldClass} mt-1`} value={block.chartType ?? "bar"} onChange={(event) => update(index, { ...block, chartType: event.target.value === "line" ? "line" : "bar" })}>
                  <option value="bar">Bar chart</option>
                  <option value="line">Line chart</option>
                </select>
              </label>
              <ChartDataEditor block={block} onChange={(data) => update(index, { ...block, data })} />
              <input className={fieldClass} value={block.sourceNote ?? ""} onChange={(event) => update(index, { ...block, sourceNote: event.target.value })} placeholder="Source note (optional)" />
            </div>
          ) : null}
        </article>
      ))}
    </section>
  );
}
