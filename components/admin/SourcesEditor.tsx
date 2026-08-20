"use client";

import type { Source } from "@/lib/content-types";

const fieldClass = "w-full rounded-md border border-border bg-bg-primary px-3 py-2 text-sm text-text-primary outline-none focus:border-accent";

export default function SourcesEditor({ sources, onChange, label = "Sources" }: { sources: Source[]; onChange: (sources: Source[]) => void; label?: string }) {
  function update(index: number, field: keyof Source, value: string) {
    onChange(sources.map((source, sourceIndex) => sourceIndex === index ? { ...source, [field]: value } : source));
  }

  return (
    <section className="rounded-xl border border-border bg-bg-secondary p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h3 className="font-serif text-base font-semibold text-text-primary">{label}</h3>
          <p className="mt-1 text-xs text-text-muted">These are shown in a structured sources section at the end of the page.</p>
        </div>
        <button
          type="button"
          onClick={() => onChange([...sources, { title: "", url: "" }])}
          className="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent"
        >
          Add source
        </button>
      </div>
      {sources.length > 0 ? (
        <div className="mt-4 space-y-3">
          {sources.map((source, index) => (
            <div key={`${index}-${source.url}`} className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto]">
              <input className={fieldClass} value={source.title} onChange={(event) => update(index, "title", event.target.value)} placeholder="Source title" />
              <input className={fieldClass} value={source.url} onChange={(event) => update(index, "url", event.target.value)} placeholder="https://…" type="url" />
              <button type="button" onClick={() => onChange(sources.filter((_, sourceIndex) => sourceIndex !== index))} className="px-2 text-xs text-red-600 hover:text-red-700">Remove</button>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}
