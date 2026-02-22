"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";

interface Issue {
  title: string;
  slug: string;
  date: string;
  summary: string;
  content: string;
}

const emptyIssue: Issue = {
  title: "",
  slug: "",
  date: new Date().toISOString().split("T")[0],
  summary: "",
  content: "",
};

export default function AdminDigestPage() {
  const router = useRouter();
  const [issues, setIssues] = useState<Issue[]>([]);
  const [editing, setEditing] = useState<Issue | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchIssues = useCallback(async () => {
    const res = await fetch("/api/admin/digest");
    if (res.status === 401) { router.push("/admin"); return; }
    setIssues(await res.json());
    setLoading(false);
  }, [router]);

  useEffect(() => { fetchIssues(); }, [fetchIssues]);

  const handleSave = async () => {
    if (!editing) return;
    const issue = {
      ...editing,
      slug: editing.slug || editing.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, ""),
    };
    await fetch("/api/admin/digest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(issue),
    });
    setEditing(null);
    fetchIssues();
  };

  const handleDelete = async (slug: string) => {
    if (!confirm("Delete this issue?")) return;
    await fetch("/api/admin/digest", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slug }),
    });
    fetchIssues();
  };

  if (loading) return <p className="text-sm text-text-muted">Loading...</p>;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="font-serif text-2xl font-bold text-text-primary">Digest Admin</h1>
        <button
          onClick={() => setEditing({ ...emptyIssue })}
          className="text-sm px-4 py-2 bg-accent text-white rounded-lg font-medium hover:bg-accent-hover transition-colors"
        >
          Add Issue
        </button>
      </div>

      {editing && (
        <div className="mb-8 p-6 border border-border rounded-lg bg-bg-secondary">
          <h2 className="font-serif text-lg font-semibold mb-4">
            {editing.slug ? "Edit" : "New"} Issue
          </h2>
          <div className="space-y-3">
            <input
              placeholder="Title"
              value={editing.title}
              onChange={(e) => setEditing({ ...editing, title: e.target.value })}
              className="w-full px-3 py-2 border border-border rounded bg-bg-primary text-text-primary text-sm focus:outline-none focus:border-accent"
            />
            <input
              placeholder="Slug (auto-generated if empty)"
              value={editing.slug}
              onChange={(e) => setEditing({ ...editing, slug: e.target.value })}
              className="w-full px-3 py-2 border border-border rounded bg-bg-primary text-text-primary text-sm focus:outline-none focus:border-accent"
            />
            <input
              type="date"
              value={editing.date}
              onChange={(e) => setEditing({ ...editing, date: e.target.value })}
              className="w-full px-3 py-2 border border-border rounded bg-bg-primary text-text-primary text-sm focus:outline-none focus:border-accent"
            />
            <input
              placeholder="Summary"
              value={editing.summary}
              onChange={(e) => setEditing({ ...editing, summary: e.target.value })}
              className="w-full px-3 py-2 border border-border rounded bg-bg-primary text-text-primary text-sm focus:outline-none focus:border-accent"
            />
            <textarea
              placeholder="Content (markdown)"
              rows={10}
              value={editing.content}
              onChange={(e) => setEditing({ ...editing, content: e.target.value })}
              className="w-full px-3 py-2 border border-border rounded bg-bg-primary text-text-primary text-sm focus:outline-none focus:border-accent font-mono"
            />
            <div className="flex gap-2">
              <button onClick={handleSave} className="px-4 py-2 bg-accent text-white rounded text-sm font-medium hover:bg-accent-hover transition-colors">
                Save
              </button>
              <button onClick={() => setEditing(null)} className="px-4 py-2 border border-border rounded text-sm text-text-secondary hover:bg-bg-secondary transition-colors">
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="space-y-2">
        {issues.map((issue) => (
          <div key={issue.slug} className="flex items-center justify-between p-3 border border-border rounded-lg">
            <div>
              <p className="text-sm font-medium text-text-primary">{issue.title}</p>
              <p className="text-xs text-text-muted">{issue.date}</p>
            </div>
            <div className="flex gap-2">
              <button onClick={() => setEditing({ ...issue })} className="text-xs text-accent hover:text-accent-hover transition-colors">
                Edit
              </button>
              <button onClick={() => handleDelete(issue.slug)} className="text-xs text-red-500 hover:text-red-400 transition-colors">
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
