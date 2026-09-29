"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ContentBlocksRenderer from "@/components/content/ContentBlocksRenderer";
import type { ContentItem } from "@/lib/content-types";
import {
  apiRequest,
  markdownBlock,
  newContent,
  toContentListItem,
  toEditableContent,
  type ContentListItem,
  type EditableContent,
} from "./admin-client";
import CoverImageFields from "./CoverImageFields";
import SourcesEditor from "./SourcesEditor";

const fieldClass =
  "w-full rounded-md border border-border bg-bg-primary px-3 py-2 text-sm text-text-primary outline-none focus:border-accent";
const categories = [
  "Foundations",
  "Models & Architectures",
  "Training & Inference",
  "Language, Vision & Retrieval",
  "Agents & Workflows",
  "Systems, Tools & Safety",
];

function metadataText(metadata: Record<string, unknown>, key: string): string {
  return typeof metadata[key] === "string" ? metadata[key] : "";
}

function metadataList(metadata: Record<string, unknown>, key: string): string[] {
  return Array.isArray(metadata[key])
    ? metadata[key].filter((value): value is string => typeof value === "string")
    : [];
}

export default function GlossaryManager() {
  const [items, setItems] = useState<ContentListItem[]>([]);
  const [editing, setEditing] = useState<EditableContent | null>(null);
  const [loadingSlug, setLoadingSlug] = useState<string | null>(null);
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState<"all" | "published" | "draft">("all");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const initialSnapshotRef = useRef<string | null>(null);

  const loadItems = useCallback(async () => {
    try {
      setItems(await apiRequest<ContentListItem[]>("/api/admin/content/glossary?summary=1"));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to load the glossary.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadItems();
  }, [loadItems]);

  async function selectItem(item: ContentListItem) {
    setLoadingSlug(item.slug);
    setError("");
    setNotice("");
    try {
      const fullItem = await apiRequest<ContentItem>(
        `/api/admin/content/glossary?slug=${encodeURIComponent(item.slug)}`
      );
      const editable = toEditableContent(fullItem);
      setEditing(editable);
      initialSnapshotRef.current = JSON.stringify(editable);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to load the glossary entry.");
    } finally {
      setLoadingSlug(null);
    }
  }

  const publishedCount = useMemo(() => items.filter((item) => item.status === "published").length, [items]);
  const draftCount = useMemo(() => items.filter((item) => item.status === "draft").length, [items]);

  const visibleItems = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return items.filter((item) => {
      if (statusFilter !== "all" && item.status !== statusFilter) return false;
      const cat = metadataText(item.metadata, "category") || "Foundations";
      if (categoryFilter !== "all" && cat !== categoryFilter) return false;
      if (!normalized) return true;
      return `${item.title} ${item.summary} ${item.slug}`.toLowerCase().includes(normalized);
    });
  }, [items, query, statusFilter, categoryFilter]);

  function update(next: Partial<EditableContent>) {
    setEditing((current) => (current ? { ...current, ...next } : current));
  }

  function updateMetadata(next: Record<string, unknown>) {
    setEditing((current) =>
      current ? { ...current, metadata: { ...current.metadata, ...next } } : current
    );
  }

  function beginNew() {
    const editable = newContent({ category: "Foundations", relatedTerms: [] });
    setEditing(editable);
    initialSnapshotRef.current = JSON.stringify(editable);
    setError("");
    setNotice("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function handleBack() {
    if (editing && initialSnapshotRef.current && JSON.stringify(editing) !== initialSnapshotRef.current) {
      if (!confirm("You have unsaved changes in this term. Discard changes and return to the glossary list?")) {
        return;
      }
    }
    setEditing(null);
    setError("");
    setNotice("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function save() {
    if (!editing) return;
    setSaving(true);
    setError("");
    setNotice("");
    try {
      const saved = await apiRequest<ContentItem>("/api/admin/content/glossary", {
        method: "POST",
        body: JSON.stringify(editing),
      });
      setItems((current) => {
        const existing = current.findIndex((item) => item.id === saved.id);
        const summary = toContentListItem(saved);
        const next = existing < 0 ? [...current, summary] : current.map((item) => (item.id === saved.id ? summary : item));
        return next.sort((a, b) => a.title.localeCompare(b.title));
      });
      const editable = toEditableContent(saved);
      setEditing(editable);
      initialSnapshotRef.current = JSON.stringify(editable);
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
      await apiRequest("/api/admin/content/glossary", {
        method: "DELETE",
        body: JSON.stringify({ slug: editing.slug }),
      });
      setItems((current) => current.filter((item) => item.id !== editing.id));
      setEditing(null);
      setNotice("Glossary entry deleted.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to delete the glossary entry.");
    } finally {
      setSaving(false);
    }
  }

  /* ========================================================================= */
  /* EDIT VIEW (Replaces list and opens at front with top navigation)           */
  /* ========================================================================= */
  if (editing) {
    return (
      <div className="space-y-6">
        {/* Sticky Top Navigation Bar */}
        <nav
          aria-label="Glossary editor navigation"
          className="sticky top-4 z-30 rounded-xl border border-border bg-bg-primary/95 p-4 shadow-sm backdrop-blur"
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={handleBack}
                className="inline-flex items-center gap-2 rounded-lg border border-border bg-bg-secondary px-3.5 py-2 text-sm font-medium text-text-primary hover:border-accent hover:text-accent hover:bg-bg-primary transition-colors cursor-pointer"
                title="Back to glossary terms"
              >
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                <span>Back to Glossary</span>
              </button>

              <div className="hidden h-5 w-px bg-border sm:block" />

              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">
                  {editing.id ? "Editing" : "New"}
                </span>
                <span
                  className="max-w-[200px] truncate text-sm font-bold text-text-primary sm:max-w-xs md:max-w-md"
                  title={editing.title || "Untitled Term"}
                >
                  {editing.title || "Untitled Term"}
                </span>
                <span className="rounded-md bg-bg-secondary px-2 py-0.5 text-xs text-text-muted">
                  {metadataText(editing.metadata, "category") || "Foundations"}
                </span>
                <span
                  className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${
                    editing.status === "published"
                      ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                      : "bg-amber-500/10 text-amber-600 dark:text-amber-400"
                  }`}
                >
                  {editing.status}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {editing.id && editing.status === "published" ? (
                <a
                  href={`/glossary/${editing.slug}`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-bg-secondary px-3.5 py-2 text-sm font-medium text-text-secondary hover:border-accent hover:text-accent hover:bg-bg-primary transition-colors"
                >
                  <span>View Live</span>
                  <span aria-hidden="true">↗</span>
                </a>
              ) : null}
              <button
                type="button"
                onClick={save}
                disabled={saving}
                className="inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-60 transition-colors shadow-sm cursor-pointer"
              >
                {saving ? (
                  <>
                    <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    <span>Saving…</span>
                  </>
                ) : (
                  <span>Save changes</span>
                )}
              </button>
            </div>
          </div>

          {/* Quick jump navigation links */}
          <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-border/60 pt-2.5 text-xs">
            <span className="text-text-muted mr-1 font-medium">Jump to:</span>
            <a
              href="#section-details"
              className="rounded-md px-2 py-1 text-text-secondary hover:bg-bg-secondary hover:text-text-primary transition-colors"
            >
              Details
            </a>
            <a
              href="#section-cover"
              className="rounded-md px-2 py-1 text-text-secondary hover:bg-bg-secondary hover:text-text-primary transition-colors"
            >
              Cover Image
            </a>
            <a
              href="#section-seo"
              className="rounded-md px-2 py-1 text-text-secondary hover:bg-bg-secondary hover:text-text-primary transition-colors"
            >
              Search & SEO
            </a>
            <a
              href="#section-sources"
              className="rounded-md px-2 py-1 text-text-secondary hover:bg-bg-secondary hover:text-text-primary transition-colors"
            >
              References ({editing.sources.length})
            </a>
            <a
              href="#section-preview"
              className="rounded-md px-2 py-1 text-text-secondary hover:bg-bg-secondary hover:text-text-primary transition-colors"
            >
              Live Preview
            </a>
          </div>
        </nav>

        {notice ? (
          <p className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800 dark:border-emerald-900/50 dark:bg-emerald-950/20 dark:text-emerald-300">
            {notice}
          </p>
        ) : null}
        {error ? (
          <p className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900/50 dark:bg-red-950/20 dark:text-red-300">
            {error}
          </p>
        ) : null}

        {/* Section: Term Details */}
        <section
          id="section-details"
          className="scroll-mt-32 rounded-xl border border-border bg-bg-secondary p-5"
        >
          <h2 className="mb-4 font-serif text-lg font-semibold text-text-primary">Term Details</h2>
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="text-sm font-medium text-text-secondary">
              Term name
              <input
                className={`${fieldClass} mt-1`}
                value={editing.title}
                onChange={(event) => update({ title: event.target.value })}
                placeholder="Term name…"
              />
            </label>
            <label className="text-sm font-medium text-text-secondary">
              URL slug
              <input
                className={`${fieldClass} mt-1`}
                value={editing.slug}
                onChange={(event) => update({ slug: event.target.value })}
                disabled={Boolean(editing.id)}
                placeholder="Generated from term name"
              />
            </label>
            <label className="text-sm font-medium text-text-secondary">
              Category
              <input
                className={`${fieldClass} mt-1`}
                list="glossary-categories"
                value={metadataText(editing.metadata, "category")}
                onChange={(event) => updateMetadata({ category: event.target.value })}
              />
              <datalist id="glossary-categories">
                {categories.map((category) => (
                  <option key={category} value={category} />
                ))}
              </datalist>
            </label>
            <label className="text-sm font-medium text-text-secondary">
              Status
              <select
                className={`${fieldClass} mt-1`}
                value={editing.status}
                onChange={(event) =>
                  update({ status: event.target.value === "draft" ? "draft" : "published" })
                }
              >
                <option value="draft">Draft</option>
                <option value="published">Published</option>
              </select>
            </label>
            <label className="sm:col-span-2 text-sm font-medium text-text-secondary">
              Short definition
              <textarea
                className={`${fieldClass} mt-1 min-h-20 leading-relaxed`}
                value={editing.summary}
                onChange={(event) => update({ summary: event.target.value })}
                placeholder="One or two sentences explaining the concept…"
              />
            </label>
            <label className="sm:col-span-2 text-sm font-medium text-text-secondary">
              Detailed definition (Markdown)
              <textarea
                className={`${fieldClass} mt-1 min-h-80 font-mono text-xs leading-6`}
                value={editing.body}
                onChange={(event) =>
                  update({ body: event.target.value, blocks: [markdownBlock(event.target.value)] })
                }
                spellCheck={false}
              />
            </label>
            <label className="sm:col-span-2 text-sm font-medium text-text-secondary">
              Related term slugs (comma-separated)
              <input
                className={`${fieldClass} mt-1`}
                value={metadataList(editing.metadata, "relatedTerms").join(", ")}
                onChange={(event) =>
                  updateMetadata({
                    relatedTerms: event.target.value
                      .split(",")
                      .map((term) => term.trim())
                      .filter(Boolean),
                  })
                }
                placeholder="e.g. transformer, attention-mechanism, llm"
              />
            </label>
          </div>
        </section>

        {/* Section: Cover Image */}
        <div id="section-cover" className="scroll-mt-32">
          <CoverImageFields
            title={editing.title}
            coverImageUrl={editing.coverImageUrl}
            coverImageAlt={editing.coverImageAlt}
            onChange={update}
          />
        </div>

        {/* Section: Search Metadata */}
        <section
          id="section-seo"
          className="scroll-mt-32 grid gap-4 rounded-xl border border-border bg-bg-secondary p-5 sm:grid-cols-2"
        >
          <h2 className="sm:col-span-2 font-serif text-lg font-semibold text-text-primary">
            Search metadata
          </h2>
          <label className="sm:col-span-2 text-sm font-medium text-text-secondary">
            SEO description
            <input
              className={`${fieldClass} mt-1`}
              value={metadataText(editing.metadata, "seoDescription")}
              onChange={(event) => updateMetadata({ seoDescription: event.target.value })}
              maxLength={160}
              placeholder="Search engine snippet (up to 160 characters)…"
            />
          </label>
          <label className="sm:col-span-2 text-sm font-medium text-text-secondary">
            SEO keywords (comma-separated)
            <input
              className={`${fieldClass} mt-1`}
              value={metadataList(editing.metadata, "seoKeywords").join(", ")}
              onChange={(event) =>
                updateMetadata({
                  seoKeywords: event.target.value
                    .split(",")
                    .map((keyword) => keyword.trim())
                    .filter(Boolean),
                })
              }
              placeholder="e.g. ai, machine learning, deep learning"
            />
          </label>
        </section>

        {/* Section: Sources */}
        <div id="section-sources" className="scroll-mt-32">
          <SourcesEditor
            label="References"
            sources={editing.sources}
            onChange={(sources) => update({ sources })}
          />
        </div>

        {/* Section: Preview */}
        <details
          id="section-preview"
          className="scroll-mt-32 rounded-xl border border-border bg-bg-secondary p-4"
        >
          <summary className="cursor-pointer font-serif text-base font-semibold text-text-primary">
            Live definition preview
          </summary>
          <div className="mt-5 rounded-lg bg-bg-primary p-4 sm:p-6">
            <ContentBlocksRenderer
              blocks={[markdownBlock(editing.body)]}
              sources={editing.sources}
            />
          </div>
        </details>

        {/* Bottom Actions */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-t border-border pt-6 pb-12">
          <button
            type="button"
            onClick={handleBack}
            className="inline-flex items-center gap-1.5 text-sm font-medium text-text-secondary hover:text-accent transition-colors cursor-pointer"
          >
            <span aria-hidden="true">←</span> Back to Glossary
          </button>
          <div className="flex items-center gap-3">
            {editing.id ? (
              <button
                type="button"
                onClick={remove}
                disabled={saving}
                className="rounded-lg border border-red-200 bg-red-50/50 px-3.5 py-2 text-sm font-medium text-red-600 hover:bg-red-100 disabled:opacity-60 dark:border-red-900/50 dark:bg-red-950/20 dark:text-red-400 dark:hover:bg-red-950/40 transition-colors cursor-pointer"
              >
                Delete term
              </button>
            ) : null}
            <button
              type="button"
              onClick={save}
              disabled={saving}
              className="rounded-lg bg-accent px-5 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-60 transition-colors shadow-sm cursor-pointer"
            >
              {saving ? "Saving…" : "Save changes"}
            </button>
          </div>
        </div>
      </div>
    );
  }

  /* ========================================================================= */
  /* LIST VIEW (Front and full width)                                          */
  /* ========================================================================= */
  return (
    <div className="space-y-6">
      {notice ? (
        <p className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800 dark:border-emerald-900/50 dark:bg-emerald-950/20 dark:text-emerald-300">
          {notice}
        </p>
      ) : null}
      {error ? (
        <p className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900/50 dark:bg-red-950/20 dark:text-red-300">
          {error}
        </p>
      ) : null}

      {/* Description & Action Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <p className="max-w-2xl text-sm leading-relaxed text-text-secondary">
          Manage reader-facing definitions, SEO details, related concepts, and citations across all
          AI glossary terms.
        </p>
        <button
          type="button"
          onClick={beginNew}
          className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-white hover:bg-accent-hover transition-colors shadow-sm cursor-pointer"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
          </svg>
          <span>New term</span>
        </button>
      </div>

      {/* Search & Filter Toolbar */}
      <div className="flex flex-col gap-3 rounded-xl border border-border bg-bg-secondary p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative flex-1">
          <input
            className={`${fieldClass} pr-8`}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search terms by name or definition…"
          />
          {query ? (
            <button
              type="button"
              onClick={() => setQuery("")}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-text-muted hover:text-text-primary"
              title="Clear search"
            >
              ✕
            </button>
          ) : null}
        </div>

        {/* Category filter */}
        <select
          className={`${fieldClass} sm:w-48`}
          value={categoryFilter}
          onChange={(event) => setCategoryFilter(event.target.value)}
        >
          <option value="all">All Categories</option>
          {categories.map((cat) => (
            <option key={cat} value={cat}>
              {cat}
            </option>
          ))}
        </select>

        {/* Status filter tabs */}
        <div className="flex items-center gap-1.5 self-start sm:self-center">
          <button
            type="button"
            onClick={() => setStatusFilter("all")}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer ${
              statusFilter === "all"
                ? "bg-accent text-white"
                : "border border-border bg-bg-primary text-text-secondary hover:text-text-primary"
            }`}
          >
            All ({items.length})
          </button>
          <button
            type="button"
            onClick={() => setStatusFilter("published")}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer ${
              statusFilter === "published"
                ? "bg-emerald-600 text-white"
                : "border border-border bg-bg-primary text-text-secondary hover:text-emerald-600"
            }`}
          >
            Published ({publishedCount})
          </button>
          <button
            type="button"
            onClick={() => setStatusFilter("draft")}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer ${
              statusFilter === "draft"
                ? "bg-amber-600 text-white"
                : "border border-border bg-bg-primary text-text-secondary hover:text-amber-600"
            }`}
          >
            Drafts ({draftCount})
          </button>
        </div>
      </div>

      {/* Content List Area */}
      {loading ? (
        <div className="space-y-3">
          {[1, 2, 3].map((index) => (
            <div
              key={index}
              className="h-24 animate-pulse rounded-xl border border-border bg-bg-secondary p-5"
            />
          ))}
        </div>
      ) : visibleItems.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border px-6 py-16 text-center">
          <h2 className="font-serif text-lg font-semibold text-text-primary">No glossary terms found</h2>
          <p className="mx-auto mt-2 max-w-md text-sm text-text-secondary">
            {query || categoryFilter !== "all" || statusFilter !== "all"
              ? "Try adjusting your search query, category, or status filter to see more terms."
              : "Get started by adding your first glossary term."}
          </p>
          {query || categoryFilter !== "all" || statusFilter !== "all" ? (
            <button
              type="button"
              onClick={() => {
                setQuery("");
                setCategoryFilter("all");
                setStatusFilter("all");
              }}
              className="mt-4 rounded-lg border border-border bg-bg-primary px-3.5 py-2 text-xs font-medium text-text-primary hover:border-accent hover:text-accent transition-colors cursor-pointer"
            >
              Reset filters
            </button>
          ) : (
            <button
              type="button"
              onClick={beginNew}
              className="mt-4 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover transition-colors cursor-pointer"
            >
              Create term
            </button>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          {visibleItems.map((item) => {
            const isLoadingThis = loadingSlug === item.slug;
            return (
              <div
                key={item.id}
                onClick={() => void selectItem(item)}
                className={`group relative flex flex-col justify-between gap-3 rounded-xl border p-5 transition-all cursor-pointer ${
                  isLoadingThis
                    ? "border-accent bg-accent/5"
                    : "border-border bg-bg-secondary hover:border-accent hover:bg-bg-primary hover:shadow-sm"
                }`}
              >
                <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <h3 className="font-serif text-base font-semibold text-text-primary group-hover:text-accent transition-colors">
                        {item.title}
                      </h3>
                      <span className="rounded-md border border-border bg-bg-primary px-2 py-0.5 text-xs text-text-muted">
                        {metadataText(item.metadata, "category") || "Foundations"}
                      </span>
                      <span
                        className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
                          item.status === "published"
                            ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                            : "bg-amber-500/10 text-amber-600 dark:text-amber-400"
                        }`}
                      >
                        {item.status}
                      </span>
                    </div>

                    <p className="mt-1 font-mono text-xs text-text-muted">/glossary/{item.slug}</p>

                    {item.summary ? (
                      <p className="mt-2 line-clamp-2 text-sm text-text-secondary leading-relaxed">
                        {item.summary}
                      </p>
                    ) : null}
                  </div>
                </div>

                <div className="flex items-center justify-between border-t border-border/60 pt-3">
                  <span className="text-xs text-text-muted">
                    {item.updatedAt ? `Updated ${new Date(item.updatedAt).toLocaleDateString()}` : ""}
                  </span>

                  <div className="flex items-center gap-2">
                    {item.status === "published" ? (
                      <a
                        href={`/glossary/${item.slug}`}
                        target="_blank"
                        rel="noreferrer"
                        onClick={(event) => event.stopPropagation()}
                        className="inline-flex items-center gap-1 rounded-md border border-border bg-bg-primary px-3 py-1.5 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent hover:bg-bg-secondary transition-colors"
                      >
                        <span>View Live</span>
                        <span aria-hidden="true">↗</span>
                      </a>
                    ) : null}

                    <button
                      type="button"
                      disabled={Boolean(loadingSlug)}
                      onClick={(event) => {
                        event.stopPropagation();
                        void selectItem(item);
                      }}
                      className="inline-flex items-center gap-1.5 rounded-md border border-border bg-bg-primary px-3 py-1.5 text-xs font-semibold text-text-primary group-hover:border-accent group-hover:bg-accent group-hover:text-white transition-colors cursor-pointer"
                    >
                      {isLoadingThis ? (
                        <>
                          <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
                          <span>Opening…</span>
                        </>
                      ) : (
                        <>
                          <span>Edit term</span>
                          <span aria-hidden="true">→</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
