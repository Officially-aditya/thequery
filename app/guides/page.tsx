import Link from "next/link";
import { getAllGuides } from "@/lib/guides";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Guides",
  description: "Evergreen study guides on AI concepts - written to build understanding, not just familiarity.",
  openGraph: {
    title: "Guides - TheQuery",
    description: "Evergreen study guides on AI concepts - written to build understanding, not just familiarity.",
  },
};

export default function GuidesPage() {
  const guides = getAllGuides();

  return (
    <div className="max-w-[960px] mx-auto px-4 py-12">
      <h1 className="font-serif text-3xl font-bold text-text-primary mb-2">Guides</h1>
      <p className="text-text-secondary mb-8">
        Evergreen study guides on AI concepts - written to build understanding, not just familiarity.
      </p>

      {guides.length === 0 ? (
        <p className="text-sm text-text-muted text-center py-12">No guides yet. Check back soon!</p>
      ) : (
        <div className="space-y-4">
          {guides.map((guide) => (
            <Link
              key={guide.slug}
              href={`/guides/${guide.slug}`}
              className="block p-5 border border-border rounded-lg hover:border-accent transition-colors group"
            >
              <h2 className="font-serif text-lg font-semibold text-text-primary group-hover:text-accent transition-colors mb-1">
                {guide.title}
              </h2>
              <p className="text-xs text-text-muted mb-2">
                {new Date(guide.date).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
              </p>
              <p className="text-sm text-text-secondary leading-relaxed">
                {guide.summary}
              </p>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
