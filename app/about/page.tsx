import type { Metadata } from "next";
import { AUTHOR, ORGANIZATION_ID, SITE_URL, authorJsonLd } from "@/lib/site";

export const metadata: Metadata = {
  title: "About",
  description:
    "Learn about TheQuery - an independent AI education platform built for developers who want to understand AI from first principles.",
  openGraph: {
    title: "About - TheQuery",
    description:
      "Learn about TheQuery - an independent AI education platform built for developers who want to understand AI from first principles.",
    images: ["/opengraph-image"],
  },
};

export default function AboutPage() {
  const personJsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        ...authorJsonLd,
        "@id": `${SITE_URL}/about#addy`,
        mainEntityOfPage: `${SITE_URL}/about`,
        knowsAbout: [
          "Artificial intelligence",
          "Machine learning",
          "Retrieval-augmented generation",
          "AI agents",
        ],
      },
      {
        "@type": "Organization",
        "@id": ORGANIZATION_ID,
        name: "TheQuery",
        url: SITE_URL,
        founder: { "@id": `${SITE_URL}/about#addy` },
      },
    ],
  };

  return (
    <div className="max-w-[720px] mx-auto px-4 py-12">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(personJsonLd) }}
      />
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
          two free technical books on AI fundamentals and RAG systems, to 300+
          glossary definitions, analytical field reports, and study guides -- is
          researched, written, and reviewed by Addy.
        </p>

        <h2 id="editorial-standards" className="font-serif text-xl font-semibold text-text-primary pt-4">
          Editorial Approach
        </h2>
        <p>
          TheQuery prioritizes primary sources -- research papers, official
          documentation, model cards, and benchmark methodology -- and aims for
          technical accuracy over simplification. Reported claims are separated
          from TheQuery&apos;s own analysis, and article source lists identify the
          material used for verification. Book and glossary pages show their
          update dates because AI products and benchmarks change quickly.
        </p>
        <p>
          When AI tools are used in the drafting process, all content undergoes
          human review and editing for accuracy, voice, and completeness. The
          site does not treat AI-assisted drafting as a substitute for checking
          a primary source.
        </p>

        <h2 className="font-serif text-xl font-semibold text-text-primary pt-4">
          Corrections and Conflicts
        </h2>
        <p>
          If a claim is out of date, a source is misrepresented, or a link is
          broken, email <a href={`mailto:${AUTHOR.email}`} className="text-accent hover:text-accent-hover transition-colors">{AUTHOR.email}</a> with the page URL and supporting evidence. Corrections are reviewed against the relevant primary source and reflected in the page&apos;s update date. TheQuery does not accept payment for editorial conclusions or rankings.
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
