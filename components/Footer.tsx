export default function Footer() {
  return (
    <footer className="border-t border-border py-8 mt-16">
      <div className="max-w-[960px] mx-auto px-4 flex items-center justify-between text-sm text-text-muted">
        <p>&copy; {new Date().getFullYear()} TheQuery.in</p>
        <a href="mailto:hello@thequery.in" className="hover:text-text-secondary transition-colors">
          Contact
        </a>
      </div>
    </footer>
  );
}
