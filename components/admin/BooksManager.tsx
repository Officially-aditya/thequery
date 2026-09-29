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
  today,
  type ContentListItem,
  type EditableContent,
} from "./admin-client";
import CoverImageFields from "./CoverImageFields";

const fieldClass =
  "w-full rounded-md border border-border bg-bg-primary px-3 py-2 text-sm text-text-primary outline-none focus:border-accent";

function metadataText(metadata: Record<string, unknown>, key: string): string {
  return typeof metadata[key] === "string" ? metadata[key] : "";
}

function newBook(): EditableContent {
  return { ...newContent({ author: "Addy", lastModified: today() }), blocks: [] };
}

function newChapter(parentSlug: string, sortOrder: number): EditableContent {
  return {
    ...newContent({ lastModified: today() }),
    parentSlug,
    summary: "",
    blocks: [markdownBlock()],
    sortOrder,
  };
}

export default function BooksManager() {
  const [books, setBooks] = useState<ContentListItem[]>([]);
  const [editingBook, setEditingBook] = useState<EditableContent | null>(null);
  const [loadingBookSlug, setLoadingBookSlug] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "published" | "draft">("all");
  const [chapters, setChapters] = useState<ContentListItem[]>([]);
  const [editingChapter, setEditingChapter] = useState<EditableContent | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingChapters, setLoadingChapters] = useState(false);
  const [savingBook, setSavingBook] = useState(false);
  const [savingChapter, setSavingChapter] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const initialBookSnapshotRef = useRef<string | null>(null);

  const loadBooks = useCallback(async () => {
    try {
      setBooks(await apiRequest<ContentListItem[]>("/api/admin/content/book?summary=1"));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to load books.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadBooks();
  }, [loadBooks]);

  async function selectBook(book: ContentListItem) {
    setLoadingBookSlug(book.slug);
    setEditingChapter(null);
    setError("");
    setNotice("");
    setLoadingChapters(true);
    try {
      const [fullBook, nextChapters] = await Promise.all([
        apiRequest<ContentItem>(`/api/admin/content/book?slug=${encodeURIComponent(book.slug)}`),
        apiRequest<ContentListItem[]>(
          `/api/admin/content/chapter?parentSlug=${encodeURIComponent(book.slug)}&summary=1`
        ),
      ]);
      const editable = toEditableContent(fullBook);
      setEditingBook(editable);
      initialBookSnapshotRef.current = JSON.stringify(editable);
      setChapters(nextChapters);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to load chapters.");
    } finally {
      setLoadingChapters(false);
      setLoadingBookSlug(null);
    }
  }

  async function selectChapter(chapter: ContentListItem) {
    if (!editingBook?.slug) return;
    setEditingChapter(null);
    setError("");
    try {
      const fullChapter = await apiRequest<ContentItem>(
        `/api/admin/content/chapter?parentSlug=${encodeURIComponent(editingBook.slug)}&slug=${encodeURIComponent(chapter.slug)}`
      );
      setEditingChapter(toEditableContent(fullChapter));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to load the chapter.");
    }
  }

  const publishedCount = useMemo(() => books.filter((item) => item.status === "published").length, [books]);
  const draftCount = useMemo(() => books.filter((item) => item.status === "draft").length, [books]);

  const visibleBooks = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return books.filter((item) => {
      if (statusFilter !== "all" && item.status !== statusFilter) return false;
      if (!normalized) return true;
      return `${item.title} ${item.summary} ${item.slug}`.toLowerCase().includes(normalized);
    });
  }, [books, query, statusFilter]);

  function updateBook(next: Partial<EditableContent>) {
    setEditingBook((current) => (current ? { ...current, ...next } : current));
  }

  function updateBookMetadata(next: Record<string, unknown>) {
    setEditingBook((current) =>
      current ? { ...current, metadata: { ...current.metadata, ...next } } : current
    );
  }

  function updateChapter(next: Partial<EditableContent>) {
    setEditingChapter((current) => (current ? { ...current, ...next } : current));
  }

  function updateChapterMetadata(next: Record<string, unknown>) {
    setEditingChapter((current) =>
      current ? { ...current, metadata: { ...current.metadata, ...next } } : current
    );
  }

  function beginNewBook() {
    const editable = newBook();
    setEditingBook(editable);
    initialBookSnapshotRef.current = JSON.stringify(editable);
    setEditingChapter(null);
    setChapters([]);
    setError("");
    setNotice("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function handleBackToBooks() {
    if (editingBook && initialBookSnapshotRef.current && JSON.stringify(editingBook) !== initialBookSnapshotRef.current) {
      if (!confirm("You have unsaved changes in this book. Discard changes and return to the books list?")) {
        return;
      }
    }
    setEditingBook(null);
    setEditingChapter(null);
    setChapters([]);
    setError("");
    setNotice("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function saveBook() {
    if (!editingBook) return;
    setSavingBook(true);
    setError("");
    setNotice("");
    try {
      const saved = await apiRequest<ContentItem>("/api/admin/content/book", {
        method: "POST",
        body: JSON.stringify(editingBook),
      });
      setBooks((current) => {
        const existing = current.findIndex((book) => book.id === saved.id);
        const summary = toContentListItem(saved);
        const next = existing < 0 ? [...current, summary] : current.map((book) => (book.id === saved.id ? summary : book));
        return next.sort((a, b) => a.title.localeCompare(b.title));
      });
      const editable = toEditableContent(saved);
      setEditingBook(editable);
      initialBookSnapshotRef.current = JSON.stringify(editable);
      setNotice("Book saved. You can now add chapters.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to save book.");
    } finally {
      setSavingBook(false);
    }
  }

  async function deleteBook() {
    if (!editingBook?.id || !confirm("Delete this book and all of its chapters? This cannot be undone.")) return;
    setSavingBook(true);
    try {
      await apiRequest("/api/admin/content/book", {
        method: "DELETE",
        body: JSON.stringify({ slug: editingBook.slug }),
      });
      setBooks((current) => current.filter((book) => book.id !== editingBook.id));
      setEditingBook(null);
      setEditingChapter(null);
      setChapters([]);
      setNotice("Book and its chapters deleted.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to delete book.");
    } finally {
      setSavingBook(false);
    }
  }

  async function saveChapter() {
    if (!editingBook?.id || !editingChapter) return;
    setSavingChapter(true);
    setError("");
    setNotice("");
    try {
      const saved = await apiRequest<ContentItem>("/api/admin/content/chapter", {
        method: "POST",
        body: JSON.stringify({ ...editingChapter, parentSlug: editingBook.slug }),
      });
      setChapters((current) => {
        const existing = current.findIndex((chapter) => chapter.id === saved.id);
        const summary = toContentListItem(saved);
        const next = existing < 0 ? [...current, summary] : current.map((chapter) => (chapter.id === saved.id ? summary : chapter));
        return next.sort((a, b) => a.sortOrder - b.sortOrder);
      });
      setEditingChapter(toEditableContent(saved));
      setNotice("Chapter saved.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to save chapter.");
    } finally {
      setSavingChapter(false);
    }
  }

  async function deleteChapter() {
    if (!editingBook || !editingChapter?.id || !confirm("Delete this chapter? This cannot be undone.")) return;
    setSavingChapter(true);
    try {
      await apiRequest("/api/admin/content/chapter", {
        method: "DELETE",
        body: JSON.stringify({ slug: editingChapter.slug, parentSlug: editingBook.slug }),
      });
      setChapters((current) => current.filter((chapter) => chapter.id !== editingChapter.id));
      setEditingChapter(null);
      setNotice("Chapter deleted.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to delete chapter.");
    } finally {
      setSavingChapter(false);
    }
  }

  /* ========================================================================= */
  /* EDIT VIEW (Replaces list and opens at front with top navigation)           */
  /* ========================================================================= */
  if (editingBook) {
    return (
      <div className="space-y-6">
        {/* Sticky Top Navigation Bar */}
        <nav
          aria-label="Book editor navigation"
          className="sticky top-4 z-30 rounded-xl border border-border bg-bg-primary/95 p-4 shadow-sm backdrop-blur"
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={handleBackToBooks}
                className="inline-flex items-center gap-2 rounded-lg border border-border bg-bg-secondary px-3.5 py-2 text-sm font-medium text-text-primary hover:border-accent hover:text-accent hover:bg-bg-primary transition-colors cursor-pointer"
                title="Back to books list"
              >
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                <span>Back to Books</span>
              </button>

              <div className="hidden h-5 w-px bg-border sm:block" />

              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">
                  {editingBook.id ? "Editing" : "New"}
                </span>
                <span
                  className="max-w-[200px] truncate text-sm font-bold text-text-primary sm:max-w-xs md:max-w-md"
                  title={editingBook.title || "Untitled Book"}
                >
                  {editingBook.title || "Untitled Book"}
                </span>
                <span
                  className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${
                    editingBook.status === "published"
                      ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                      : "bg-amber-500/10 text-amber-600 dark:text-amber-400"
                  }`}
                >
                  {editingBook.status}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={saveBook}
                disabled={savingBook}
                className="inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-60 transition-colors shadow-sm cursor-pointer"
              >
                {savingBook ? (
                  <>
                    <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    <span>Saving…</span>
                  </>
                ) : (
                  <span>Save book</span>
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
              href="#section-chapters"
              className="rounded-md px-2 py-1 text-text-secondary hover:bg-bg-secondary hover:text-text-primary transition-colors"
            >
              Chapters ({chapters.length})
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

        {/* Section: Book Details */}
        <section
          id="section-details"
          className="scroll-mt-32 rounded-xl border border-border bg-bg-secondary p-5"
        >
          <h2 className="mb-4 font-serif text-lg font-semibold text-text-primary">Book Details</h2>
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="text-sm font-medium text-text-secondary">
              Book title
              <input
                className={`${fieldClass} mt-1`}
                value={editingBook.title}
                onChange={(event) => updateBook({ title: event.target.value })}
                placeholder="Book title…"
              />
            </label>
            <label className="text-sm font-medium text-text-secondary">
              URL slug
              <input
                className={`${fieldClass} mt-1`}
                value={editingBook.slug}
                disabled={Boolean(editingBook.id)}
                onChange={(event) => updateBook({ slug: event.target.value })}
                placeholder="Generated from title"
              />
            </label>
            <label className="text-sm font-medium text-text-secondary">
              Author
              <input
                className={`${fieldClass} mt-1`}
                value={metadataText(editingBook.metadata, "author")}
                onChange={(event) => updateBookMetadata({ author: event.target.value })}
              />
            </label>
            <label className="text-sm font-medium text-text-secondary">
              Status
              <select
                className={`${fieldClass} mt-1`}
                value={editingBook.status}
                onChange={(event) =>
                  updateBook({ status: event.target.value === "draft" ? "draft" : "published" })
                }
              >
                <option value="draft">Draft</option>
                <option value="published">Published</option>
              </select>
            </label>
            <label className="sm:col-span-2 text-sm font-medium text-text-secondary">
              Description
              <textarea
                className={`${fieldClass} mt-1 min-h-24 leading-relaxed`}
                value={editingBook.summary}
                onChange={(event) => updateBook({ summary: event.target.value })}
                placeholder="Overview of this book…"
              />
            </label>
            <label className="text-sm font-medium text-text-secondary">
              Last modified
              <input
                className={`${fieldClass} mt-1`}
                type="date"
                value={metadataText(editingBook.metadata, "lastModified")}
                onChange={(event) => updateBookMetadata({ lastModified: event.target.value })}
              />
            </label>
            {editingBook.id ? (
              <p className="self-end text-xs text-text-muted">
                The slug is locked to protect reader links.
              </p>
            ) : null}
          </div>
        </section>

        {/* Section: Cover Image */}
        <div id="section-cover" className="scroll-mt-32">
          <CoverImageFields
            title={editingBook.title}
            coverImageUrl={editingBook.coverImageUrl}
            coverImageAlt={editingBook.coverImageAlt}
            onChange={updateBook}
          />
        </div>

        {/* Section: Chapters */}
        <div id="section-chapters" className="scroll-mt-32">
          {editingBook.id ? (
            <section className="grid gap-5 rounded-xl border border-border bg-bg-secondary p-5 lg:grid-cols-[260px_minmax(0,1fr)]">
              <aside className="border-b border-border pb-4 lg:border-b-0 lg:border-r lg:pr-4">
                <div className="flex items-center justify-between gap-2">
                  <h2 className="font-serif text-lg font-semibold text-text-primary">Chapters</h2>
                  <button
                    type="button"
                    onClick={() => setEditingChapter(newChapter(editingBook.slug, chapters.length))}
                    className="rounded-md border border-border bg-bg-primary px-3 py-1.5 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent transition-colors cursor-pointer"
                  >
                    + Add chapter
                  </button>
                </div>
                <div className="mt-3 space-y-1">
                  {loadingChapters ? (
                    <p className="text-sm text-text-muted">Loading chapters…</p>
                  ) : chapters.length === 0 ? (
                    <p className="py-4 text-xs text-text-muted">No chapters yet. Add one!</p>
                  ) : (
                    chapters.map((chapter) => (
                      <button
                        key={chapter.id}
                        type="button"
                        onClick={() => void selectChapter(chapter)}
                        className={`w-full rounded-md px-2.5 py-2 text-left transition-colors cursor-pointer ${
                          editingChapter?.id === chapter.id
                            ? "bg-bg-primary font-medium shadow-sm border border-border"
                            : "hover:bg-bg-primary/70"
                        }`}
                      >
                        <span className="block truncate text-sm text-text-primary">
                          {chapter.sortOrder + 1}. {chapter.title}
                        </span>
                        <span className="text-xs text-text-muted">{chapter.status}</span>
                      </button>
                    ))
                  )}
                </div>
              </aside>

              <div className="min-w-0">
                {!editingChapter ? (
                  <div className="py-12 text-center text-sm text-text-muted">
                    Select a chapter from the list to edit, or click <strong>+ Add chapter</strong>.
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
                      <h3 className="font-serif text-base font-semibold text-text-primary">
                        {editingChapter.id ? "Edit Chapter" : "New Chapter"}
                      </h3>
                      <button
                        type="button"
                        onClick={saveChapter}
                        disabled={savingChapter}
                        className="rounded-md bg-accent px-4 py-1.5 text-xs font-medium text-white hover:bg-accent-hover disabled:opacity-60 transition-colors shadow-sm cursor-pointer"
                      >
                        {savingChapter ? "Saving…" : "Save chapter"}
                      </button>
                    </div>

                    <div className="grid gap-3 sm:grid-cols-2">
                      <label className="text-sm font-medium text-text-secondary">
                        Title
                        <input
                          className={`${fieldClass} mt-1`}
                          value={editingChapter.title}
                          onChange={(event) => updateChapter({ title: event.target.value })}
                        />
                      </label>
                      <label className="text-sm font-medium text-text-secondary">
                        URL slug
                        <input
                          className={`${fieldClass} mt-1`}
                          value={editingChapter.slug}
                          disabled={Boolean(editingChapter.id)}
                          onChange={(event) => updateChapter({ slug: event.target.value })}
                        />
                      </label>
                      <label className="text-sm font-medium text-text-secondary">
                        Last modified
                        <input
                          className={`${fieldClass} mt-1`}
                          type="date"
                          value={metadataText(editingChapter.metadata, "lastModified")}
                          onChange={(event) =>
                            updateChapterMetadata({ lastModified: event.target.value })
                          }
                        />
                      </label>
                      <label className="text-sm font-medium text-text-secondary">
                        Position
                        <input
                          className={`${fieldClass} mt-1`}
                          type="number"
                          min="0"
                          value={editingChapter.sortOrder}
                          onChange={(event) =>
                            updateChapter({ sortOrder: Number(event.target.value) })
                          }
                        />
                      </label>
                      <label className="text-sm font-medium text-text-secondary">
                        Status
                        <select
                          className={`${fieldClass} mt-1`}
                          value={editingChapter.status}
                          onChange={(event) =>
                            updateChapter({
                              status: event.target.value === "draft" ? "draft" : "published",
                            })
                          }
                        >
                          <option value="draft">Draft</option>
                          <option value="published">Published</option>
                        </select>
                      </label>
                    </div>

                    <CoverImageFields
                      title={editingChapter.title}
                      coverImageUrl={editingChapter.coverImageUrl}
                      coverImageAlt={editingChapter.coverImageAlt}
                      onChange={updateChapter}
                    />

                    <label className="block text-sm font-medium text-text-secondary">
                      Chapter content (Markdown)
                      <textarea
                        className={`${fieldClass} mt-1 min-h-96 font-mono text-xs leading-6`}
                        value={editingChapter.body}
                        onChange={(event) =>
                          updateChapter({
                            body: event.target.value,
                            blocks: [markdownBlock(event.target.value)],
                          })
                        }
                        spellCheck={false}
                      />
                    </label>

                    <details className="rounded-lg border border-border bg-bg-primary p-3">
                      <summary className="cursor-pointer text-sm font-medium text-text-primary">
                        Preview chapter
                      </summary>
                      <div className="mt-4 rounded-md bg-bg-secondary p-3 sm:p-4">
                        <ContentBlocksRenderer blocks={editingChapter.blocks} />
                      </div>
                    </details>

                    <div className="flex items-center justify-between pt-2">
                      <button
                        type="button"
                        onClick={() => setEditingChapter(null)}
                        className="text-xs text-text-secondary hover:text-accent transition-colors cursor-pointer"
                      >
                        Close chapter editor
                      </button>
                      {editingChapter.id ? (
                        <button
                          type="button"
                          onClick={deleteChapter}
                          disabled={savingChapter}
                          className="text-xs text-red-600 hover:text-red-700 transition-colors cursor-pointer"
                        >
                          Delete chapter
                        </button>
                      ) : null}
                    </div>
                  </div>
                )}
              </div>
            </section>
          ) : (
            <p className="rounded-lg border border-dashed border-border px-4 py-5 text-sm text-text-muted">
              Save the book first, then add its chapters.
            </p>
          )}
        </div>

        {/* Bottom Actions */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-t border-border pt-6 pb-12">
          <button
            type="button"
            onClick={handleBackToBooks}
            className="inline-flex items-center gap-1.5 text-sm font-medium text-text-secondary hover:text-accent transition-colors cursor-pointer"
          >
            <span aria-hidden="true">←</span> Back to Books list
          </button>
          <div className="flex items-center gap-3">
            {editingBook.id ? (
              <button
                type="button"
                onClick={deleteBook}
                disabled={savingBook}
                className="rounded-lg border border-red-200 bg-red-50/50 px-3.5 py-2 text-sm font-medium text-red-600 hover:bg-red-100 disabled:opacity-60 dark:border-red-900/50 dark:bg-red-950/20 dark:text-red-400 dark:hover:bg-red-950/40 transition-colors cursor-pointer"
              >
                Delete book
              </button>
            ) : null}
            <button
              type="button"
              onClick={saveBook}
              disabled={savingBook}
              className="rounded-lg bg-accent px-5 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-60 transition-colors shadow-sm cursor-pointer"
            >
              {savingBook ? "Saving…" : "Save book"}
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
          Build structured books and manage their ordered chapters from a unified editorial
          workspace.
        </p>
        <button
          type="button"
          onClick={beginNewBook}
          className="inline-flex h-10 shrink-0 items-center justify-center gap-2 rounded-lg bg-accent px-4 text-sm font-medium text-white hover:bg-accent-hover transition-colors shadow-sm cursor-pointer"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
          </svg>
          <span>New book</span>
        </button>
      </div>

      {/* Search & Filter Toolbar */}
      <div className="flex flex-col gap-3 rounded-xl border border-border bg-bg-secondary p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative flex-1">
          <input
            className="h-10 w-full rounded-lg border border-border bg-bg-primary px-3.5 pr-9 text-sm text-text-primary outline-none focus:border-accent"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search books by title or summary…"
          />
          {query ? (
            <button
              type="button"
              onClick={() => setQuery("")}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-text-muted hover:text-text-primary cursor-pointer"
              title="Clear search"
            >
              ✕
            </button>
          ) : null}
        </div>

        {/* Status filter tabs */}
        <div className="flex items-center gap-2 self-stretch sm:self-auto">
          <button
            type="button"
            onClick={() => setStatusFilter("all")}
            className={`h-10 rounded-lg px-3.5 text-xs sm:text-sm font-medium inline-flex items-center justify-center transition-colors cursor-pointer ${
              statusFilter === "all"
                ? "bg-accent text-white shadow-sm"
                : "border border-border bg-bg-primary text-text-secondary hover:text-text-primary"
            }`}
          >
            All ({books.length})
          </button>
          <button
            type="button"
            onClick={() => setStatusFilter("published")}
            className={`h-10 rounded-lg px-3.5 text-xs sm:text-sm font-medium inline-flex items-center justify-center transition-colors cursor-pointer ${
              statusFilter === "published"
                ? "bg-emerald-600 text-white shadow-sm"
                : "border border-border bg-bg-primary text-text-secondary hover:text-emerald-600"
            }`}
          >
            Published ({publishedCount})
          </button>
          <button
            type="button"
            onClick={() => setStatusFilter("draft")}
            className={`h-10 rounded-lg px-3.5 text-xs sm:text-sm font-medium inline-flex items-center justify-center transition-colors cursor-pointer ${
              statusFilter === "draft"
                ? "bg-amber-600 text-white shadow-sm"
                : "border border-border bg-bg-primary text-text-secondary hover:text-amber-600"
            }`}
          >
            Drafts ({draftCount})
          </button>
        </div>
      </div>

      {/* Books List Area */}
      {loading ? (
        <div className="space-y-3">
          {[1, 2].map((index) => (
            <div
              key={index}
              className="h-28 animate-pulse rounded-xl border border-border bg-bg-secondary p-5"
            />
          ))}
        </div>
      ) : visibleBooks.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border px-6 py-16 text-center">
          <h2 className="font-serif text-lg font-semibold text-text-primary">No books found</h2>
          <p className="mx-auto mt-2 max-w-md text-sm text-text-secondary">
            {query || statusFilter !== "all"
              ? "Try adjusting your search query or status filter to see more books."
              : "Get started by creating your first book."}
          </p>
          {query || statusFilter !== "all" ? (
            <button
              type="button"
              onClick={() => {
                setQuery("");
                setStatusFilter("all");
              }}
              className="mt-4 rounded-lg border border-border bg-bg-primary px-3.5 py-2 text-xs font-medium text-text-primary hover:border-accent hover:text-accent transition-colors cursor-pointer"
            >
              Reset filters
            </button>
          ) : (
            <button
              type="button"
              onClick={beginNewBook}
              className="mt-4 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover transition-colors cursor-pointer"
            >
              Create book
            </button>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          {visibleBooks.map((book) => {
            const isLoadingThis = loadingBookSlug === book.slug;
            return (
              <div
                key={book.id}
                onClick={() => void selectBook(book)}
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
                        {book.title}
                      </h3>
                      <span
                        className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
                          book.status === "published"
                            ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                            : "bg-amber-500/10 text-amber-600 dark:text-amber-400"
                        }`}
                      >
                        {book.status}
                      </span>
                    </div>

                    <p className="mt-1 font-mono text-xs text-text-muted">/books/{book.slug}</p>

                    {book.summary ? (
                      <p className="mt-2 line-clamp-2 text-sm text-text-secondary leading-relaxed">
                        {book.summary}
                      </p>
                    ) : null}
                  </div>
                  <div className="shrink-0 text-left sm:text-right">
                    {isLoadingThis ? (
                      <span className="inline-flex items-center gap-1.5 text-xs font-medium text-accent">
                        <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
                        Opening…
                      </span>
                    ) : null}
                  </div>
                </div>

                {book.updatedAt ? (
                  <div className="border-t border-border/60 pt-2.5 text-xs text-text-muted">
                    Updated {new Date(book.updatedAt).toLocaleDateString()}
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
