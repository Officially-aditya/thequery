import Link from "next/link";

export default function Footer() {
  return (
    <footer className="border-t border-border py-8 mt-16">
      <div className="max-w-[960px] mx-auto px-4 flex items-center justify-between text-sm text-text-muted">
        <p>&copy; {new Date().getFullYear()} TheQuery.in</p>
        <nav className="flex items-center gap-4">
          <Link href="/books" className="hover:text-text-secondary transition-colors">Books</Link>
          <Link href="/glossary" className="hover:text-text-secondary transition-colors">Glossary</Link>
          <Link href="/digest" className="hover:text-text-secondary transition-colors">Digest</Link>
          <a href="mailto:hello@thequery.in" className="hover:text-text-secondary transition-colors">Contact</a>
        </nav>
      </div>
    </footer>
  );
}
