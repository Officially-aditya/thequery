import Link from "next/link";
import PreferredSourceButton from "./PreferredSourceButton";

export default function Footer() {
  return (
    <footer className="border-t border-border mt-16">
      <div className="max-w-[960px] mx-auto px-4 py-10 text-sm text-text-muted">
        <div className="grid grid-cols-1 gap-8 sm:grid-cols-3 sm:gap-16">
          <div>
            <p className="font-serif text-3xl font-semibold tracking-tight text-text-primary">TheQuery</p>
            <PreferredSourceButton className="mt-5" />
          </div>

          <nav aria-label="Explore TheQuery" className="flex flex-col items-start gap-3">
            <p className="font-serif text-base font-semibold text-text-primary">Explore</p>
            <Link href="/books" className="hover:text-text-secondary transition-colors">Books</Link>
            <Link href="/guides" className="hover:text-text-secondary transition-colors">Guides</Link>
            <Link href="/glossary" className="hover:text-text-secondary transition-colors">Glossary</Link>
            <Link href="/articles" className="hover:text-text-secondary transition-colors">Articles</Link>
          </nav>

          <nav aria-label="Privacy and contact" className="justify-self-end flex flex-col items-start gap-3">
            <p className="font-serif text-base font-semibold text-text-primary">Privacy &amp; Contact</p>
            <Link href="/about" className="hover:text-text-secondary transition-colors">About</Link>
            <Link href="/privacy" className="hover:text-text-secondary transition-colors">Privacy</Link>
            <a href="mailto:addy@thequery.in" className="hover:text-text-secondary transition-colors">Contact</a>
          </nav>
        </div>

        <div className="border-t border-border mt-10 pt-4">
          <p>&copy; {new Date().getFullYear()} TheQuery.in</p>
        </div>
      </div>
    </footer>
  );
}
