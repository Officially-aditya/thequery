import Link from "next/link";

export default function Footer() {
  return (
    <footer className="border-t border-border py-8 mt-16">
      <div className="max-w-[960px] mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-text-muted">
        <p>&copy; {new Date().getFullYear()} TheQuery.in</p>
        <nav className="flex items-center gap-4 flex-wrap justify-center">
          <Link href="/books" className="hover:text-text-secondary transition-colors">Books</Link>
          <Link href="/guides" className="hover:text-text-secondary transition-colors">Guides</Link>
          <Link href="/glossary" className="hover:text-text-secondary transition-colors">Glossary</Link>
          <Link href="/articles" className="hover:text-text-secondary transition-colors">Articles</Link>
          <Link href="/about" className="hover:text-text-secondary transition-colors">About</Link>
          <Link href="/privacy" className="hover:text-text-secondary transition-colors">Privacy</Link>
          <a href="mailto:addy@thequery.in" className="hover:text-text-secondary transition-colors">Contact</a>
        </nav>
      </div>
    </footer>
  );
}
