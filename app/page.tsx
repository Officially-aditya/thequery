import Link from "next/link";

const sections = [
  {
    title: "Books",
    description: "In-depth guides on AI and machine learning, written from first principles. Read online for free.",
    href: "/books",
  },
  {
    title: "Glossary",
    description: "Clear, concise definitions of key AI and ML terms. Search and browse by category.",
    href: "/glossary",
  },
  {
    title: "Weekly Digest",
    description: "A curated summary of the most important AI developments, research, and news each week.",
    href: "/digest",
  },
];

export default function Home() {
  return (
    <div className="max-w-[960px] mx-auto px-4 py-16">
      <div className="text-center mb-16">
        <h1 className="font-serif text-4xl font-bold text-text-primary mb-4">
          TheQuery
        </h1>
        <p className="text-lg text-text-secondary">
          AI knowledge from first principles
        </p>
      </div>

      <div className="space-y-8">
        {sections.map((section) => (
          <div key={section.href} className="border-b border-border pb-8 last:border-b-0">
            <h2 className="font-serif text-xl font-semibold text-text-primary mb-2">
              {section.title}
            </h2>
            <p className="text-text-secondary leading-relaxed mb-4">
              {section.description}
            </p>
            <Link
              href={section.href}
              className="inline-flex items-center gap-1.5 text-sm font-medium text-accent hover:text-accent-hover transition-colors"
            >
              Explore
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
            </Link>
          </div>
        ))}
      </div>
    </div>
  );
}
