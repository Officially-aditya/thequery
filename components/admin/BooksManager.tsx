"use client";

import { useCallback, useEffect, useState } from "react";
import ContentBlocksRenderer from "@/components/content/ContentBlocksRenderer";
import type { ContentItem } from "@/lib/content-types";
import { apiRequest, markdownBlock, newContent, toContentListItem, toEditableContent, today, type ContentListItem, type EditableContent } from "./admin-client";
import CoverImageFields from "./CoverImageFields";

const fieldClass = "w-full rounded-md border border-border bg-bg-primary px-3 py-2 text-sm text-text-primary outline-none focus:border-accent";

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
  const [chapters, setChapters] = useState<ContentListItem[]>([]);
  const [editingChapter, setEditingChapter] = useState<EditableContent | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingChapters, setLoadingChapters] = useState(false);
  const [savingBook, setSavingBook] = useState(false);
  const [savingChapter, setSavingChapter] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const loadBooks = useCallback(async () => {
    try {
      setBooks(await apiRequest<ContentListItem[]>("/api/admin/content/book?summary=1"));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to load books.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void loadBooks(); }, [loadBooks]);

  async function selectBook(book: ContentListItem) {
    setEditingBook(null);
    setEditingChapter(null);
    setError("");
    setNotice("");
    setLoadingChapters(true);
    try {
      const [fullBook, nextChapters] = await Promise.all([
        apiRequest<ContentItem>(`/api/admin/content/book?slug=${encodeURIComponent(book.slug)}`),
        apiRequest<ContentListItem[]>(`/api/admin/content/chapter?parentSlug=${encodeURIComponent(book.slug)}&summary=1`),
      ]);
      setEditingBook(toEditableContent(fullBook));
      setChapters(nextChapters);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to load chapters.");
    } finally {
      setLoadingChapters(false);
    }
  }

  async function selectChapter(chapter: ContentListItem) {
    if (!editingBook?.slug) return;
    setEditingChapter(null);
    setError("");
    try {
      const fullChapter = await apiRequest<ContentItem>(
        `/api/admin/content/chapter?parentSlug=${encodeURIComponent(editingBook.slug)}&slug=${encodeURIComponent(chapter.slug)}`,
      );
      setEditingChapter(toEditableContent(fullChapter));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to load the chapter.");
    }
  }

  function updateBook(next: Partial<EditableContent>) {
    setEditingBook((current) => current ? { ...current, ...next } : current);
  }

  function updateBookMetadata(next: Record<string, unknown>) {
    setEditingBook((current) => current ? { ...current, metadata: { ...current.metadata, ...next } } : current);
  }

  function updateChapter(next: Partial<EditableContent>) {
    setEditingChapter((current) => current ? { ...current, ...next } : current);
  }

  function updateChapterMetadata(next: Record<string, unknown>) {
    setEditingChapter((current) => current ? { ...current, metadata: { ...current.metadata, ...next } } : current);
  }

  async function saveBook() {
    if (!editingBook) return;
    setSavingBook(true);
    setError("");
    setNotice("");
    try {
      const saved = await apiRequest<ContentItem>("/api/admin/content/book", { method: "POST", body: JSON.stringify(editingBook) });
      setBooks((current) => {
        const existing = current.findIndex((book) => book.id === saved.id);
        const summary = toContentListItem(saved);
        const next = existing < 0 ? [...current, summary] : current.map((book) => book.id === saved.id ? summary : book);
        return next.sort((a, b) => a.title.localeCompare(b.title));
      });
      setEditingBook(toEditableContent(saved));
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
      await apiRequest("/api/admin/content/book", { method: "DELETE", body: JSON.stringify({ slug: editingBook.slug }) });
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
        const next = existing < 0 ? [...current, summary] : current.map((chapter) => chapter.id === saved.id ? summary : chapter);
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
      await apiRequest("/api/admin/content/chapter", { method: "DELETE", body: JSON.stringify({ slug: editingChapter.slug, parentSlug: editingBook.slug }) });
      setChapters((current) => current.filter((chapter) => chapter.id !== editingChapter.id));
      setEditingChapter(null);
      setNotice("Chapter deleted.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to delete chapter.");
    } finally {
      setSavingChapter(false);
    }
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[260px_minmax(0,1fr)]">
      <aside className="rounded-xl border border-border bg-bg-secondary p-3 xl:sticky xl:top-8 xl:h-[calc(100vh-4rem)] xl:overflow-y-auto">
        <button onClick={() => { setEditingBook(newBook()); setEditingChapter(null); setChapters([]); setError(""); setNotice(""); }} className="w-full rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-white hover:bg-accent-hover">New book</button>
        <p className="mt-4 text-xs text-text-muted">{books.length} books · {books.filter((book) => book.status === "draft").length} drafts</p>
        <div className="mt-3 space-y-1">
          {loading ? <p className="px-3 py-6 text-sm text-text-muted">Loading…</p> : books.map((book) => (
            <button key={book.id} onClick={() => void selectBook(book)} className={`w-full rounded-lg px-3 py-2 text-left ${editingBook?.id === book.id ? "bg-bg-primary shadow-sm" : "hover:bg-bg-primary/70"}`}>
              <span className="block truncate text-sm font-medium text-text-primary">{book.title}</span>
              <span className="mt-1 block text-xs text-text-muted">{book.status}</span>
            </button>
          ))}
        </div>
      </aside>

      <div className="min-w-0">
        {!editingBook ? (
          <div className="rounded-xl border border-dashed border-border px-6 py-16 text-center"><h2 className="font-serif text-xl font-semibold text-text-primary">Choose a book or create one</h2><p className="mt-2 text-sm text-text-secondary">Build a book, then manage its ordered chapters from the same editorial workspace.</p></div>
        ) : (
          <div className="space-y-7">
            <div className="flex flex-wrap items-center justify-between gap-3"><p className="text-sm text-text-muted">{editingBook.id ? "Editing saved book" : "New book"}</p><div className="flex gap-2">{editingBook.id && editingBook.status === "published" ? <a href={`/books/${editingBook.slug}`} target="_blank" rel="noreferrer" className="rounded-md border border-border px-3 py-2 text-sm text-text-secondary hover:border-accent hover:text-accent">Open public page</a> : null}<button onClick={saveBook} disabled={savingBook} className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-60">{savingBook ? "Saving…" : "Save book"}</button></div></div>
            {notice ? <p className="rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-700">{notice}</p> : null}
            {error ? <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p> : null}

            <section className="grid gap-4 rounded-xl border border-border p-4 sm:grid-cols-2">
              <label className="text-sm font-medium text-text-secondary">Book title<input className={`${fieldClass} mt-1`} value={editingBook.title} onChange={(event) => updateBook({ title: event.target.value })} /></label>
              <label className="text-sm font-medium text-text-secondary">URL slug<input className={`${fieldClass} mt-1`} value={editingBook.slug} disabled={Boolean(editingBook.id)} onChange={(event) => updateBook({ slug: event.target.value })} placeholder="Generated from title" /></label>
              <label className="text-sm font-medium text-text-secondary">Author<input className={`${fieldClass} mt-1`} value={metadataText(editingBook.metadata, "author")} onChange={(event) => updateBookMetadata({ author: event.target.value })} /></label>
              <label className="text-sm font-medium text-text-secondary">Status<select className={`${fieldClass} mt-1`} value={editingBook.status} onChange={(event) => updateBook({ status: event.target.value === "draft" ? "draft" : "published" })}><option value="draft">Draft</option><option value="published">Published</option></select></label>
              <label className="sm:col-span-2 text-sm font-medium text-text-secondary">Description<textarea className={`${fieldClass} mt-1 min-h-24`} value={editingBook.summary} onChange={(event) => updateBook({ summary: event.target.value })} /></label>
              <label className="text-sm font-medium text-text-secondary">Last modified<input className={`${fieldClass} mt-1`} type="date" value={metadataText(editingBook.metadata, "lastModified")} onChange={(event) => updateBookMetadata({ lastModified: event.target.value })} /></label>
              {editingBook.id ? <p className="self-end text-xs text-text-muted">The slug is locked to protect reader links.</p> : null}
            </section>

            <CoverImageFields title={editingBook.title} coverImageUrl={editingBook.coverImageUrl} coverImageAlt={editingBook.coverImageAlt} onChange={updateBook} />
            {editingBook.id ? (
              <section className="grid gap-5 rounded-xl border border-border p-4 lg:grid-cols-[240px_minmax(0,1fr)]">
                <aside className="border-b border-border pb-4 lg:border-b-0 lg:border-r lg:pr-4">
                  <div className="flex items-center justify-between gap-2"><h2 className="font-serif text-lg font-semibold text-text-primary">Chapters</h2><button onClick={() => setEditingChapter(newChapter(editingBook.slug, chapters.length))} className="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent">Add chapter</button></div>
                  <div className="mt-3 space-y-1">{loadingChapters ? <p className="text-sm text-text-muted">Loading…</p> : chapters.map((chapter) => <button key={chapter.id} onClick={() => void selectChapter(chapter)} className={`w-full rounded-md px-2 py-2 text-left ${editingChapter?.id === chapter.id ? "bg-bg-secondary" : "hover:bg-bg-secondary"}`}><span className="block truncate text-sm text-text-primary">{chapter.sortOrder + 1}. {chapter.title}</span><span className="text-xs text-text-muted">{chapter.status}</span></button>)}</div>
                </aside>
                <div className="min-w-0">
                  {!editingChapter ? <p className="py-8 text-sm text-text-muted">Select a chapter or create a new one.</p> : <div className="space-y-4">
                    <div className="flex flex-wrap justify-between gap-3"><h3 className="font-serif text-lg font-semibold text-text-primary">{editingChapter.id ? "Edit chapter" : "New chapter"}</h3><div className="flex gap-2"><button onClick={saveChapter} disabled={savingChapter} className="rounded-md bg-accent px-3 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-60">{savingChapter ? "Saving…" : "Save chapter"}</button></div></div>
                    <div className="grid gap-3 sm:grid-cols-2"><label className="text-sm font-medium text-text-secondary">Title<input className={`${fieldClass} mt-1`} value={editingChapter.title} onChange={(event) => updateChapter({ title: event.target.value })} /></label><label className="text-sm font-medium text-text-secondary">URL slug<input className={`${fieldClass} mt-1`} value={editingChapter.slug} disabled={Boolean(editingChapter.id)} onChange={(event) => updateChapter({ slug: event.target.value })} /></label><label className="text-sm font-medium text-text-secondary">Last modified<input className={`${fieldClass} mt-1`} type="date" value={metadataText(editingChapter.metadata, "lastModified")} onChange={(event) => updateChapterMetadata({ lastModified: event.target.value })} /></label><label className="text-sm font-medium text-text-secondary">Position<input className={`${fieldClass} mt-1`} type="number" min="0" value={editingChapter.sortOrder} onChange={(event) => updateChapter({ sortOrder: Number(event.target.value) })} /></label><label className="text-sm font-medium text-text-secondary">Status<select className={`${fieldClass} mt-1`} value={editingChapter.status} onChange={(event) => updateChapter({ status: event.target.value === "draft" ? "draft" : "published" })}><option value="draft">Draft</option><option value="published">Published</option></select></label></div>
                    <CoverImageFields title={editingChapter.title} coverImageUrl={editingChapter.coverImageUrl} coverImageAlt={editingChapter.coverImageAlt} onChange={updateChapter} />
                    <label className="block text-sm font-medium text-text-secondary">Chapter content (Markdown)<textarea className={`${fieldClass} mt-1 min-h-96 font-mono text-xs leading-6`} value={editingChapter.body} onChange={(event) => updateChapter({ body: event.target.value, blocks: [markdownBlock(event.target.value)] })} spellCheck={false} /></label>
                    <details className="rounded-lg border border-border bg-bg-secondary p-3"><summary className="cursor-pointer text-sm font-medium text-text-primary">Preview chapter</summary><div className="mt-4 rounded-md bg-bg-primary p-3"><ContentBlocksRenderer blocks={editingChapter.blocks} /></div></details>
                    <div className="flex items-center justify-between"><button onClick={() => setEditingChapter(null)} className="text-sm text-text-secondary hover:text-accent">Close chapter</button>{editingChapter.id ? <button onClick={deleteChapter} disabled={savingChapter} className="text-sm text-red-600 hover:text-red-700">Delete chapter</button> : null}</div>
                  </div>}
                </div>
              </section>
            ) : <p className="rounded-lg border border-dashed border-border px-4 py-5 text-sm text-text-muted">Save the book first, then add its chapters.</p>}

            <div className="flex items-center justify-between border-t border-border pt-5"><button onClick={() => { setEditingBook(null); setEditingChapter(null); }} className="text-sm text-text-secondary hover:text-accent">Close editor</button>{editingBook.id ? <button onClick={deleteBook} disabled={savingBook} className="text-sm text-red-600 hover:text-red-700">Delete book</button> : null}</div>
          </div>
        )}
      </div>
    </div>
  );
}
