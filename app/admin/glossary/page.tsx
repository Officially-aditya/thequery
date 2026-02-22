"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";

interface Term {
  name: string;
  slug: string;
  shortDef: string;
  fullDef: string;
  category: string;
  relatedTerms: string[];
  lastUpdated: string;
}

const CATEGORIES = ["Fundamentals", "Deep Learning", "NLP", "Computer Vision", "Reinforcement Learning", "MLOps"];

const emptyTerm: Term = {
  name: "",
  slug: "",
  shortDef: "",
  fullDef: "",
  category: "Fundamentals",
  relatedTerms: [],
  lastUpdated: new Date().toISOString().split("T")[0],
};

export default function AdminGlossaryPage() {
  const router = useRouter();
  const [terms, setTerms] = useState<Term[]>([]);
  const [editing, setEditing] = useState<Term | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchTerms = useCallback(async () => {
    const res = await fetch("/api/admin/glossary");
    if (res.status === 401) { router.push("/admin"); return; }
    setTerms(await res.json());
    setLoading(false);
  }, [router]);

  useEffect(() => { fetchTerms(); }, [fetchTerms]);

  const handleSave = async () => {
    if (!editing) return;
    const term = {
      ...editing,
      slug: editing.slug || editing.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, ""),
      lastUpdated: new Date().toISOString().split("T")[0],
    };
    await fetch("/api/admin/glossary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(term),
    });
    setEditing(null);
    fetchTerms();
  };

  const handleDelete = async (slug: string) => {
    if (!confirm("Delete this term?")) return;
    await fetch("/api/admin/glossary", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slug }),
    });
    fetchTerms();
  };

  if (loading) return <p className="text-sm text-text-muted">Loading...</p>;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="font-serif text-2xl font-bold text-text-primary">Glossary Admin</h1>
        <button
          onClick={() => setEditing({ ...emptyTerm })}
          className="text-sm px-4 py-2 bg-accent text-white rounded-lg font-medium hover:bg-accent-hover transition-colors"
        >
          Add Term
        </button>
      </div>

      {editing && (
        <div className="mb-8 p-6 border border-border rounded-lg bg-bg-secondary">
          <h2 className="font-serif text-lg font-semibold mb-4">
            {editing.slug ? "Edit" : "New"} Term
          </h2>
          <div className="space-y-3">
            <input
              placeholder="Name"
              value={editing.name}
              onChange={(e) => setEditing({ ...editing, name: e.target.value })}
              className="w-full px-3 py-2 border border-border rounded bg-bg-primary text-text-primary text-sm focus:outline-none focus:border-accent"
            />
            <input
              placeholder="Slug (auto-generated if empty)"
              value={editing.slug}
              onChange={(e) => setEditing({ ...editing, slug: e.target.value })}
              className="w-full px-3 py-2 border border-border rounded bg-bg-primary text-text-primary text-sm focus:outline-none focus:border-accent"
            />
            <input
              placeholder="Short definition"
              value={editing.shortDef}
              onChange={(e) => setEditing({ ...editing, shortDef: e.target.value })}
              className="w-full px-3 py-2 border border-border rounded bg-bg-primary text-text-primary text-sm focus:outline-none focus:border-accent"
            />
            <textarea
              placeholder="Full definition (markdown)"
              rows={5}
              value={editing.fullDef}
              onChange={(e) => setEditing({ ...editing, fullDef: e.target.value })}
              className="w-full px-3 py-2 border border-border rounded bg-bg-primary text-text-primary text-sm focus:outline-none focus:border-accent font-mono"
            />
            <select
              value={editing.category}
              onChange={(e) => setEditing({ ...editing, category: e.target.value })}
              className="w-full px-3 py-2 border border-border rounded bg-bg-primary text-text-primary text-sm focus:outline-none focus:border-accent"
            >
              {CATEGORIES.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
            <input
              placeholder="Related term slugs (comma-separated)"
              value={editing.relatedTerms.join(", ")}
              onChange={(e) => setEditing({ ...editing, relatedTerms: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })}
              className="w-full px-3 py-2 border border-border rounded bg-bg-primary text-text-primary text-sm focus:outline-none focus:border-accent"
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
        {terms.sort((a, b) => a.name.localeCompare(b.name)).map((term) => (
          <div key={term.slug} className="flex items-center justify-between p-3 border border-border rounded-lg">
            <div>
              <p className="text-sm font-medium text-text-primary">{term.name}</p>
              <p className="text-xs text-text-muted">{term.category}</p>
            </div>
            <div className="flex gap-2">
              <button onClick={() => setEditing({ ...term })} className="text-xs text-accent hover:text-accent-hover transition-colors">
                Edit
              </button>
              <button onClick={() => handleDelete(term.slug)} className="text-xs text-red-500 hover:text-red-400 transition-colors">
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
