"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { apiRequest } from "./admin-client";

const navigation = [
  { href: "/admin", label: "Overview" },
  { href: "/admin/articles", label: "Articles" },
  { href: "/admin/guides", label: "Guides" },
  { href: "/admin/glossary", label: "Glossary" },
  { href: "/admin/books", label: "Books" },
];

interface AdminShellProps {
  children: React.ReactNode;
  title: string;
  actions?: React.ReactNode;
  initialEmail?: string;
}

export default function AdminShell({ children, title, actions, initialEmail }: AdminShellProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [email, setEmail] = useState(initialEmail ?? "");
  const [checking, setChecking] = useState(initialEmail === undefined);

  useEffect(() => {
    if (initialEmail !== undefined) return;

    let cancelled = false;
    apiRequest<{ authenticated: boolean; email: string | null }>("/api/admin/auth")
      .then((session) => {
        if (cancelled) return;
        if (!session.authenticated) router.replace("/admin");
        else setEmail(session.email ?? "");
      })
      .catch(() => {
        if (!cancelled) router.replace("/admin");
      })
      .finally(() => {
        if (!cancelled) setChecking(false);
      });

    return () => {
      cancelled = true;
    };
  }, [initialEmail, router]);

  async function signOut() {
    await apiRequest("/api/admin/auth", { method: "DELETE" }).catch(() => undefined);
    router.replace("/admin");
  }

  if (checking) return <p className="py-12 text-sm text-text-muted">Checking editorial access…</p>;

  return (
    <div className="grid gap-8 lg:grid-cols-[190px_minmax(0,1fr)]">
      <aside className="lg:sticky lg:top-8 lg:h-fit">
        <Link href="/" className="font-serif text-lg font-semibold text-text-primary">TheQuery <span className="text-accent">Editor</span></Link>
        <nav className="mt-6 flex gap-2 overflow-x-auto lg:flex-col" aria-label="Admin navigation">
          {navigation.map((item) => {
            const active = item.href === "/admin" ? pathname === item.href : pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`rounded-lg px-3 py-2 text-sm transition-colors ${active ? "bg-accent text-white" : "text-text-secondary hover:bg-bg-secondary hover:text-text-primary"}`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="mt-6 border-t border-border pt-4 text-xs text-text-muted">
          <p className="truncate">{email}</p>
          <button onClick={signOut} className="mt-2 text-text-secondary hover:text-accent">Sign out</button>
        </div>
      </aside>
      <section className="min-w-0">
        <div className="mb-7 flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-xs font-medium uppercase tracking-[0.16em] text-accent">Editorial desk</p>
            <h1 className="mt-1 font-serif text-3xl font-bold text-text-primary">{title}</h1>
          </div>
          {actions}
        </div>
        {children}
      </section>
    </div>
  );
}
