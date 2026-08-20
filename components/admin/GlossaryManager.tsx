"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import ContentBlocksRenderer from "@/components/content/ContentBlocksRenderer";
import type { ContentItem } from "@/lib/content-types";
import { apiRequest, markdownBlock, newContent, toEditableContent, type EditableContent } from "./admin-client";
import CoverImageFields from "./CoverImageFields";
import SourcesEditor from "./SourcesEditor";

const fieldClass = "w-full rounded-md border border-border bg-bg-primary px-3 py-2 text-sm text-text-primary outline-none focus:border-accent";
const categories = ["Foundations", "Models & Architectures", "Training & Inference", "Language, Vision & Retrieval", "Agents & Workflows", "Systems, Tools & Safety"];

function metadataText(metadata: Record<string, unknown>, key: string): string {
  return typeof metadata[key] === "string" ? metadata[key] : "";
}

function metadataList(metadata: Record<string, unknown>, key: string): string[] {
  return Array.isArray(metadata[key]) ? metadata[key].filter((value): value is string => typeof value === "string") : [];
}

export default function GlossaryManager() {
  const [items, setItems] = useState<ContentItem[]>([]);
  const [editing, setEditing] = useState<EditableContent | null>(null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const loadItems = useCallback(async () => {
    try {
      setItems(await apiRequest<ContentItem[]>("/api/admin/content/glossary"));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to load the glossary.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void loadItems(); }, [loadItems]);

  const visibleItems = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return normalized ? items.filter((item) => `${item.title} ${item.summary}`.toLowerCase().includes(normalized)) : items;
  }, [items, query]);

  function update(next: Partial<EditableContent>) {
    setEditing((current) => current ? { ...current, ...next } : current);
  }

  function updateMetadata(next: Record<string, unknown>) {
    setEditing((current) => current ? { ...current, metadata: { ...current.metadata, ...next } } : current);
  }

  function beginNew() {
    setEditing(newContent({ category: "Foundations", relatedTerms: [] }));
    setError("");
    setNotice("");
  }

  async function save() {
    if (!editing) return;
    setSaving(true);
    setError("");
    setNotice("");
    try {
      const saved = await apiRequest<ContentItem>("/api/admin/content/glossary", { method: "POST", body: JSON.stringify(editing) });
      setItems((current) => {
        const existing = current.findIndex((item) => item.id === saved.id);
        const next = existing < 0 ? [...current, saved] : current.map((item) => item.id === saved.id ? saved : item);
        return next.sort((a, b) => a.title.localeCompare(b.title));
      });
      setEditing(toEditableContent(saved));
      setNotice("Glossary entry saved.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to save the glossary entry.");
    } finally {
      setSaving(false);
    }
  }

  async function remove() {
    if (!editing?.id || !confirm("Delete this glossary entry? This cannot be undone.")) return;
    setSaving(true);
    try {
      await apiRequest("/api/admin/content/glossary", { method: "DELETE", body: JSON.stringify({ slug: editing.slug }) });
      setItems((current) => current.filter((item) => item.id !== editing.id));
      setEditing(null);
      setNotice("Glossary entry deleted.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to delete the glossary entry.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[260px_minmax(0,1fr)]">
      <aside className="rounded-xl border border-border bg-bg-secondary p-3 xl:sticky xl:top-8 xl:h-[calc(100vh-4rem)] xl:overflow-y-auto">
        <button onClick={beginNew} className="w-full rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-white hover:bg-accent-hover">New term</button>
        <input className={`${fieldClass} mt-3`} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search terms…" />
        <p className="mt-4 text-xs text-text-muted">{items.length} terms · {items.filter((item) => item.status === "draft").length} drafts</p>
        <div className="mt-3 space-y-1">
          {loading ? <p className="px-3 py-6 text-sm text-text-muted">Loading…</p> : visibleItems.map((item) => (
            <button key={item.id} onClick={() => { setEditing(toEditableContent(item)); setError(""); setNotice(""); }} className={`w-full rounded-lg px-3 py-2 text-left ${editing?.id === item.id ? "bg-bg-primary shadow-sm" : "hover:bg-bg-primary/70"}`}>
              <span className="block truncate text-sm font-medium text-text-primary">{item.title}</span>
              <span className="block truncate pt-1 text-xs text-text-muted">{metadataText(item.metadata, "category") || "Foundations"}</span>
            </button>
          ))}
        </div>
      </aside>

      <div className="min-w-0">
        {!editing ? (
          <div className="rounded-xl border border-dashed border-border px-6 py-16 text-center"><h2 className="font-serif text-xl font-semibold text-text-primary">Choose a term or create one</h2><p className="mt-2 text-sm text-text-secondary">The glossary editor keeps reader-facing definitions, SEO details, related concepts, and citations together.</p></div>
        ) : (
          <div className="space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm text-text-muted">{editing.id ? "Editing saved term" : "New draft"}</p>
              <div className="flex gap-2">
                {editing.id && editing.status === "published" ? <a href={`/glossary/${editing.slug}`} target="_blank" rel="noreferrer" className="rounded-md border border-border px-3 py-2 text-sm text-text-secondary hover:border-accent hover:text-accent">Open public page</a> : null}
                <button onClick={save} disabled={saving} className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-60">{saving ? "Saving…" : "Save changes"}</button>
              </div>
            </div>
            {notice ? <p className="rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-700">{notice}</p> : null}
            {error ? <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p> : null}

            <section className="grid gap-4 rounded-xl border border-border p-4 sm:grid-cols-2">
              <label className="text-sm font-medium text-text-secondary">Term name<input className={`${fieldClass} mt-1`} value={editing.title} onChange={(event) => update({ title: event.target.value })} /></label>
              <label className="text-sm font-medium text-text-secondary">URL slug<input className={`${fieldClass} mt-1`} value={editing.slug} onChange={(event) => update({ slug: event.target.value })} disabled={Boolean(editing.id)} placeholder="Generated from term name" /></label>
              <label className="text-sm font-medium text-text-secondary">Category<input className={`${fieldClass} mt-1`} list="glossary-categories" value={metadataText(editing.metadata, "category")} onChange={(event) => updateMetadata({ category: event.target.value })} /><datalist id="glossary-categories">{categories.map((category) => <option key={category} value={category} />)}</datalist></label>
              <label className="text-sm font-medium text-text-secondary">Status<select className={`${fieldClass} mt-1`} value={editing.status} onChange={(event) => update({ status: event.target.value === "draft" ? "draft" : "published" })}><option value="draft">Draft</option><option value="published">Published</option></select></label>
              <label className="sm:col-span-2 text-sm font-medium text-text-secondary">Short definition<textarea className={`${fieldClass} mt-1 min-h-20`} value={editing.summary} onChange={(event) => update({ summary: event.target.value })} /></label>
              <label className="sm:col-span-2 text-sm font-medium text-text-secondary">Detailed definition (Markdown)<textarea className={`${fieldClass} mt-1 min-h-80 font-mono text-xs leading-6`} value={editing.body} onChange={(event) => update({ body: event.target.value, blocks: [markdownBlock(event.target.value)] })} spellCheck={false} /></label>
              <label className="sm:col-span-2 text-sm font-medium text-text-secondary">Analogy (optional)<textarea className={`${fieldClass} mt-1 min-h-20`} value={metadataText(editing.metadata, "analogy")} onChange={(event) => updateMetadata({ analogy: event.target.value })} /></label>
              <label className="sm:col-span-2 text-sm font-medium text-text-secondary">Related term slugs (comma-separated)<input className={`${fieldClass} mt-1`} value={metadataList(editing.metadata, "relatedTerms").join(", ")} onChange={(event) => updateMetadata({ relatedTerms: event.target.value.split(",").map((term) => term.trim()).filter(Boolean) })} /></label>
            </section>

            <CoverImageFields title={editing.title} coverImageUrl={editing.coverImageUrl} coverImageAlt={editing.coverImageAlt} onChange={update} />
            <section className="grid gap-4 rounded-xl border border-border p-4 sm:grid-cols-2">
              <h2 className="sm:col-span-2 font-serif text-lg font-semibold text-text-primary">Search metadata</h2>
              <label className="sm:col-span-2 text-sm font-medium text-text-secondary">SEO description<input className={`${fieldClass} mt-1`} value={metadataText(editing.metadata, "seoDescription")} onChange={(event) => updateMetadata({ seoDescription: event.target.value })} maxLength={160} /></label>
              <label className="sm:col-span-2 text-sm font-medium text-text-secondary">SEO keywords (comma-separated)<input className={`${fieldClass} mt-1`} value={metadataList(editing.metadata, "seoKeywords").join(", ")} onChange={(event) => updateMetadata({ seoKeywords: event.target.value.split(",").map((keyword) => keyword.trim()).filter(Boolean) })} /></label>
            </section>

            <SourcesEditor label="References" sources={editing.sources} onChange={(sources) => update({ sources })} />
            <details className="rounded-xl border border-border bg-bg-secondary p-4"><summary className="cursor-pointer font-serif text-base font-semibold text-text-primary">Live definition preview</summary><div className="mt-5 rounded-lg bg-bg-primary p-4"><ContentBlocksRenderer blocks={[markdownBlock(editing.body)]} sources={editing.sources} disableMath /></div></details>
            <div className="flex items-center justify-between border-t border-border pt-5"><button onClick={() => setEditing(null)} className="text-sm text-text-secondary hover:text-accent">Close editor</button>{editing.id ? <button onClick={remove} disabled={saving} className="text-sm text-red-600 hover:text-red-700">Delete term</button> : null}</div>
          </div>
        )}
      </div>
    </div>
  );
}
