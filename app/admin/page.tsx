"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import AdminShell from "@/components/admin/AdminShell";
import { ApiError, apiRequest } from "@/components/admin/admin-client";
import type { ContentKind } from "@/lib/content-types";

type DashboardData = {
  email: string;
  counts: Record<ContentKind, number>;
};

const sections: Array<{ kind: "article" | "guide" | "glossary" | "book"; href: string; name: string; detail: string }> = [
  { kind: "article", href: "/admin/articles", name: "Articles", detail: "News analysis, data stories, and structured sources." },
  { kind: "guide", href: "/admin/guides", name: "Guides", detail: "Evergreen explainers with tables, charts, and citations." },
  { kind: "glossary", href: "/admin/glossary", name: "Glossary", detail: "Definitions, references, related concepts, and SEO." },
  { kind: "book", href: "/admin/books", name: "Books", detail: "Books and their ordered, reader-ready chapters." },
];

export default function AdminPage() {
  const [authenticated, setAuthenticated] = useState<boolean | null>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);

  const loadDashboard = useCallback(async () => {
    try {
      const data = await apiRequest<DashboardData>("/api/admin/dashboard");
      setDashboard(data);
      setAuthenticated(true);
    } catch (requestError) {
      setDashboard(null);
      setAuthenticated(false);
      if (!(requestError instanceof ApiError && requestError.status === 401)) {
        setError(requestError instanceof Error ? requestError.message : "Unable to load dashboard.");
      }
    }
  }, []);

  useEffect(() => {
    void loadDashboard();
  }, [loadDashboard]);

  async function signIn(event: React.FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      await apiRequest("/api/admin/auth", { method: "POST", body: JSON.stringify({ email, password }) });
      setPassword("");
      setAuthenticated(null);
      await loadDashboard();
    } catch {
      setError("The email or password was not recognized.");
    } finally {
      setSubmitting(false);
    }
  }

  if (authenticated === null) return <p className="py-12 text-center text-sm text-text-muted">Checking editorial access…</p>;

  if (!authenticated) {
    return (
      <div className="mx-auto max-w-md py-10">
        <Link href="/" className="font-serif text-lg font-semibold text-text-primary">TheQuery</Link>
        <div className="mt-5 rounded-2xl border border-border bg-bg-secondary p-6 sm:p-8">
          <p className="text-xs font-medium uppercase tracking-[0.16em] text-accent">Editorial desk</p>
          <h1 className="mt-2 font-serif text-3xl font-bold text-text-primary">Sign in to publish</h1>
          <p className="mt-3 text-sm leading-relaxed text-text-secondary">Use the configured editorial account to manage articles, guides, glossary terms, books, charts, tables, and sources.</p>
          <form onSubmit={signIn} className="mt-6 space-y-4">
            <label className="block text-sm font-medium text-text-secondary">Email<input className="mt-1 w-full rounded-md border border-border bg-bg-primary px-3 py-2 text-text-primary outline-none focus:border-accent" type="email" autoComplete="username" value={email} onChange={(event) => setEmail(event.target.value)} required /></label>
            <label className="block text-sm font-medium text-text-secondary">Password<input className="mt-1 w-full rounded-md border border-border bg-bg-primary px-3 py-2 text-text-primary outline-none focus:border-accent" type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} required /></label>
            {error ? <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p> : null}
            <button type="submit" disabled={submitting} className="w-full rounded-md bg-accent px-4 py-2.5 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-60">{submitting ? "Signing in…" : "Sign in"}</button>
          </form>
          <p className="mt-5 text-xs leading-relaxed text-text-muted">Credentials are checked against a salted hash in Neon. Session tokens are random, HTTP-only, and expire after 24 hours.</p>
        </div>
      </div>
    );
  }

  return (
    <AdminShell initialEmail={dashboard?.email} title="Editorial overview" actions={<Link href="/admin/articles" className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover">Create content</Link>}>
      <div className="rounded-xl border border-border bg-bg-secondary p-5">
        <p className="text-sm text-text-secondary">Signed in as <span className="font-medium text-text-primary">{dashboard?.email ?? "…"}</span>. Everything below is served from Neon; public changes are revalidated as soon as you save.</p>
      </div>
      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        {sections.map((section) => (
          <Link key={section.kind} href={section.href} className="group rounded-xl border border-border p-5 transition-colors hover:border-accent hover:bg-bg-secondary">
            <div className="flex items-start justify-between gap-4"><h2 className="font-serif text-xl font-semibold text-text-primary group-hover:text-accent">{section.name}</h2><span className="font-mono text-2xl text-text-primary">{dashboard?.counts[section.kind] ?? "—"}</span></div>
            <p className="mt-2 text-sm leading-relaxed text-text-secondary">{section.detail}</p>
            <span className="mt-4 inline-block text-sm font-medium text-accent">Manage {section.name.toLowerCase()} →</span>
          </Link>
        ))}
      </div>
      <div className="mt-6 grid gap-4 md:grid-cols-3">
        <div className="rounded-xl border border-border p-4"><p className="text-xs font-medium uppercase tracking-wide text-text-muted">Chapters</p><p className="mt-2 font-serif text-2xl text-text-primary">{dashboard?.counts.chapter ?? "—"}</p></div>
        <div className="rounded-xl border border-border p-4"><p className="text-xs font-medium uppercase tracking-wide text-text-muted">Structured blocks</p><p className="mt-2 text-sm leading-relaxed text-text-secondary">Comparison tables and charts are editable blocks—not fragments hidden in Markdown.</p></div>
        <div className="rounded-xl border border-border p-4"><p className="text-xs font-medium uppercase tracking-wide text-text-muted">Sources</p><p className="mt-2 text-sm leading-relaxed text-text-secondary">Citations are managed separately and rendered at the bottom of each published page.</p></div>
      </div>
    </AdminShell>
  );
}
