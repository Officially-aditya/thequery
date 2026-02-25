"use client";

import Link from "next/link";
import ThemeToggle from "./ThemeToggle";
import Logo from "./Logo";
import { usePathname } from "next/navigation";

const navLinks = [
  { href: "/books", label: "Books" },
  { href: "/glossary", label: "Glossary" },
  { href: "/articles", label: "Articles" },
];

export default function Header() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 bg-bg-primary/80 backdrop-blur-md border-b border-border">
      <div className="px-6 h-14 flex items-center justify-between">
        <Link href="/" className="hover:opacity-80 transition-opacity">
          <Logo />
        </Link>
        <nav className="flex items-center gap-6">
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`text-sm transition-colors ${
                pathname?.startsWith(link.href)
                  ? "text-accent font-medium"
                  : "text-text-secondary hover:text-text-primary"
              }`}
            >
              {link.label}
            </Link>
          ))}
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}
