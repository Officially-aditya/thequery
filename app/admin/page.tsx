"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

export default function AdminPage() {
  const [loggedIn, setLoggedIn] = useState(false);
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    // Check if already authenticated by trying to fetch admin data
    fetch("/api/admin/glossary")
      .then((r) => {
        if (r.ok) setLoggedIn(true);
      })
      .finally(() => setChecking(false));
  }, []);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    const res = await fetch("/api/admin/auth", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });
    if (res.ok) {
      setLoggedIn(true);
      setPassword("");
    } else {
      setError("Invalid password");
    }
  };

  const handleLogout = async () => {
    await fetch("/api/admin/auth", { method: "DELETE" });
    setLoggedIn(false);
  };

  if (checking) {
    return <p className="text-sm text-text-muted">Loading...</p>;
  }

  if (!loggedIn) {
    return (
      <div className="max-w-sm mx-auto">
        <h1 className="font-serif text-2xl font-bold text-text-primary mb-6">Admin Login</h1>
        <form onSubmit={handleLogin} className="space-y-4">
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            className="w-full px-4 py-2 border border-border rounded-lg bg-bg-primary text-text-primary placeholder-text-muted focus:outline-none focus:border-accent text-sm"
          />
          {error && <p className="text-sm text-red-500">{error}</p>}
          <button
            type="submit"
            className="w-full px-4 py-2 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent-hover transition-colors"
          >
            Log In
          </button>
        </form>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <h1 className="font-serif text-2xl font-bold text-text-primary">Admin Panel</h1>
        <button
          onClick={handleLogout}
          className="text-sm text-text-muted hover:text-text-secondary transition-colors"
        >
          Log Out
        </button>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Link
          href="/admin/glossary"
          className="block p-6 border border-border rounded-lg hover:border-accent transition-colors"
        >
          <h2 className="font-serif text-lg font-semibold text-text-primary mb-1">Glossary</h2>
          <p className="text-sm text-text-secondary">Manage glossary terms</p>
        </Link>
        <Link
          href="/admin/articles"
          className="block p-6 border border-border rounded-lg hover:border-accent transition-colors"
        >
          <h2 className="font-serif text-lg font-semibold text-text-primary mb-1">Articles</h2>
          <p className="text-sm text-text-secondary">Manage articles</p>
        </Link>
      </div>
    </div>
  );
}
