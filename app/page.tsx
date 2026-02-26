import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "TheQuery - AI Knowledge from First Principles",
  description: "TheQuery is where developers go to understand AI, not just use it. Glossary, books, and articles covering AI from first principles.",
  openGraph: {
    title: "TheQuery - AI Knowledge from First Principles",
    description: "TheQuery is where developers go to understand AI, not just use it. Glossary, books, and articles covering AI from first principles.",
  },
};

const sections = [
  {
    title: "Books",
    description: "Two free technical books on AI fundamentals and RAG systems - written from first principles, readable online or downloadable as PDF.",
    href: "/books",
  },
  {
    title: "Glossary",
    description: "100+ AI and ML terms explained clearly - from backpropagation to knowledge graphs, always up to date.",
    href: "/glossary",
  },
  {
    title: "Articles",
    description: "Detailed reports on what is happening in the field of AI, updated regularly.",
    href: "/articles",
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
