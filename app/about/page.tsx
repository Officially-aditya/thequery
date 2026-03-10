import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "About",
  description:
    "Learn about TheQuery - an independent AI education platform built for developers who want to understand AI from first principles.",
  openGraph: {
    title: "About - TheQuery",
    description:
      "Learn about TheQuery - an independent AI education platform built for developers who want to understand AI from first principles.",
  },
};

export default function AboutPage() {
  return (
    <div className="max-w-[720px] mx-auto px-4 py-12">
      <h1 className="font-serif text-3xl font-bold text-text-primary mb-8">
        About TheQuery
      </h1>

      <div className="space-y-6 text-text-secondary leading-relaxed">
        <p>
          TheQuery is an independent AI education platform created by Addy. It
          is built for developers who want to understand AI, not just use it.
          Every article, glossary entry, and book chapter is written from first
          principles -- starting with the foundational math, building up through
          core algorithms, and arriving at practical implementation.
        </p>

        <h2 className="font-serif text-xl font-semibold text-text-primary pt-4">
          Who Makes This
        </h2>
        <p>
          TheQuery is written and maintained by Addy. All content -- from the
          two free technical books on AI fundamentals and RAG systems, to 233+
          glossary definitions, analytical field reports, and study guides -- is
          researched, written, and reviewed by Addy.
        </p>

        <h2 className="font-serif text-xl font-semibold text-text-primary pt-4">
          Editorial Approach
        </h2>
        <p>
          Content on TheQuery references primary sources -- research papers,
          official documentation, and benchmark results -- and aims for technical
          accuracy over simplification. When AI tools are used in the drafting
          process, all content undergoes human review and editing for accuracy,
          voice, and completeness.
        </p>

        <h2 className="font-serif text-xl font-semibold text-text-primary pt-4">
          Contact
        </h2>
        <p>
          For corrections, feedback, or collaboration inquiries, reach out at{" "}
          <a
            href="mailto:addy@thequery.in"
            className="text-accent hover:text-accent-hover transition-colors"
          >
            addy@thequery.in
          </a>
          .
        </p>
      </div>
    </div>
  );
}
