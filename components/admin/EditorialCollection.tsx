"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import ContentBlocksRenderer from "@/components/content/ContentBlocksRenderer";
import type { ContentItem } from "@/lib/content-types";
import { apiRequest, newContent, publicHref, toEditableContent, type EditableContent } from "./admin-client";
import CoverImageFields from "./CoverImageFields";
import EditorialBlocksEditor from "./EditorialBlocksEditor";
import SourcesEditor from "./SourcesEditor";

const fieldClass = "w-full rounded-md border border-border bg-bg-primary px-3 py-2 text-sm text-text-primary outline-none focus:border-accent";

type CollectionKind = "article" | "guide";

export default function EditorialCollection({ kind, noun, description }: { kind: CollectionKind; noun: string; description: string }) {
  const [items, setItems] = useState<ContentItem[]>([]);
  const [editing, setEditing] = useState<EditableContent | null>(null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  const loadItems = useCallback(async () => {
    try {
      setItems(await apiRequest<ContentItem[]>(`/api/admin/content/${kind}`));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to load content.");
    } finally {
      setLoading(false);
    }
  }, [kind]);

  useEffect(() => { void loadItems(); }, [loadItems]);

  const visibleItems = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return items;
    return items.filter((item) => `${item.title} ${item.slug}`.toLowerCase().includes(normalized));
  }, [items, query]);

  function beginNew() {
    setNotice("");
    setError("");
    setEditing(newContent(kind === "article" ? { manualGlossaryLinks: false } : {}));
  }

  function update(next: Partial<EditableContent>) {
    setEditing((current) => current ? { ...current, ...next } : current);
  }

  async function save() {
    if (!editing) return;
    setSaving(true);
    setError("");
    setNotice("");
    try {
      const saved = await apiRequest<ContentItem>(`/api/admin/content/${kind}`, {
        method: "POST",
        body: JSON.stringify(editing),
      });
      setItems((current) => {
        const existing = current.findIndex((item) => item.id === saved.id);
        const next = existing >= 0 ? current.map((item) => item.id === saved.id ? saved : item) : [saved, ...current];
        return [...next].sort((a, b) => (b.publishedAt ?? "").localeCompare(a.publishedAt ?? ""));
      });
      setEditing(toEditableContent(saved));
      setNotice(`${noun} saved.`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to save content.");
    } finally {
      setSaving(false);
    }
  }

  async function remove() {
    if (!editing?.id || !confirm(`Delete this ${noun.toLowerCase()}? This cannot be undone.`)) return;
    setSaving(true);
    try {
      await apiRequest(`/api/admin/content/${kind}`, { method: "DELETE", body: JSON.stringify({ slug: editing.slug }) });
      setItems((current) => current.filter((item) => item.id !== editing.id));
      setEditing(null);
      setNotice(`${noun} deleted.`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to delete content.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[260px_minmax(0,1fr)]">
      <aside className="rounded-xl border border-border bg-bg-secondary p-3 xl:sticky xl:top-8 xl:h-[calc(100vh-4rem)] xl:overflow-y-auto">
        <button onClick={beginNew} className="w-full rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-white hover:bg-accent-hover">New {noun}</button>
        <input className={`${fieldClass} mt-3`} value={query} onChange={(event) => setQuery(event.target.value)} placeholder={`Search ${noun.toLowerCase()}s…`} />
        <p className="mt-4 text-xs text-text-muted">{items.length} total · {items.filter((item) => item.status === "draft").length} drafts</p>
        <div className="mt-3 space-y-1">
          {loading ? <p className="px-3 py-6 text-sm text-text-muted">Loading…</p> : visibleItems.map((item) => (
            <button
              key={item.id}
              onClick={() => { setEditing(toEditableContent(item)); setNotice(""); setError(""); }}
              className={`w-full rounded-lg px-3 py-2 text-left transition-colors ${editing?.id === item.id ? "bg-bg-primary shadow-sm" : "hover:bg-bg-primary/70"}`}
            >
              <span className="block truncate text-sm font-medium text-text-primary">{item.title}</span>
              <span className="mt-1 flex items-center gap-2 text-xs text-text-muted"><span className={item.status === "published" ? "text-emerald-600" : "text-amber-600"}>{item.status}</span>{item.publishedAt || "No date"}</span>
            </button>
          ))}
        </div>
      </aside>

      <div className="min-w-0">
        {!editing ? (
          <div className="rounded-xl border border-dashed border-border px-6 py-16 text-center">
            <h2 className="font-serif text-xl font-semibold text-text-primary">Select a {noun.toLowerCase()} or start a new one</h2>
            <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-text-secondary">{description}</p>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm text-text-muted">{editing.id ? "Editing saved content" : "New draft"}</p>
              <div className="flex gap-2">
                {editing.id && editing.status === "published" ? <a href={publicHref(kind, editing.slug)} target="_blank" rel="noreferrer" className="rounded-md border border-border px-3 py-2 text-sm text-text-secondary hover:border-accent hover:text-accent">Open public page</a> : null}
                <button onClick={save} disabled={saving} className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-60">{saving ? "Saving…" : "Save changes"}</button>
              </div>
            </div>
            {notice ? <p className="rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-700">{notice}</p> : null}
            {error ? <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p> : null}

            <section className="grid gap-4 rounded-xl border border-border p-4 sm:grid-cols-2">
              <label className="text-sm font-medium text-text-secondary">Title<input className={`${fieldClass} mt-1`} value={editing.title} onChange={(event) => update({ title: event.target.value })} /></label>
              <label className="text-sm font-medium text-text-secondary">URL slug<input className={`${fieldClass} mt-1`} value={editing.slug} onChange={(event) => update({ slug: event.target.value })} disabled={Boolean(editing.id)} placeholder="Generated from title if empty" /></label>
              <label className="text-sm font-medium text-text-secondary">Publish date<input className={`${fieldClass} mt-1`} type="date" value={editing.publishedAt} onChange={(event) => update({ publishedAt: event.target.value })} /></label>
              <label className="text-sm font-medium text-text-secondary">Status<select className={`${fieldClass} mt-1`} value={editing.status} onChange={(event) => update({ status: event.target.value === "draft" ? "draft" : "published" })}><option value="draft">Draft</option><option value="published">Published</option></select></label>
              <label className="sm:col-span-2 text-sm font-medium text-text-secondary">Deck / summary<textarea className={`${fieldClass} mt-1 min-h-20`} value={editing.summary} onChange={(event) => update({ summary: event.target.value })} /></label>
              {kind === "article" ? <label className="sm:col-span-2 flex items-center gap-2 text-sm text-text-secondary"><input type="checkbox" checked={editing.metadata.manualGlossaryLinks === true} onChange={(event) => update({ metadata: { ...editing.metadata, manualGlossaryLinks: event.target.checked } })} />This article already contains its own glossary links</label> : null}
              {editing.id ? <p className="sm:col-span-2 text-xs text-text-muted">The slug is fixed after creation so existing links stay valid.</p> : null}
            </section>

            <CoverImageFields title={editing.title} coverImageUrl={editing.coverImageUrl} coverImageAlt={editing.coverImageAlt} onChange={update} />
            <EditorialBlocksEditor blocks={editing.blocks} onChange={(blocks) => update({ blocks })} />
            <SourcesEditor sources={editing.sources} onChange={(sources) => update({ sources })} />

            <details className="rounded-xl border border-border bg-bg-secondary p-4">
              <summary className="cursor-pointer font-serif text-base font-semibold text-text-primary">Live content preview</summary>
              <div className="mt-5 rounded-lg bg-bg-primary p-4">
                <ContentBlocksRenderer blocks={editing.blocks} sources={editing.sources} />
              </div>
            </details>

            <div className="flex items-center justify-between border-t border-border pt-5">
              <button onClick={() => setEditing(null)} className="text-sm text-text-secondary hover:text-accent">Close editor</button>
              {editing.id ? <button onClick={remove} disabled={saving} className="text-sm text-red-600 hover:text-red-700 disabled:opacity-60">Delete {noun.toLowerCase()}</button> : null}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
