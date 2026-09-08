import Link from "next/link";
import type { Metadata } from "next";
import { SITE_URL } from "@/lib/site";

export const metadata: Metadata = {
  title: "For journalists",
  description: "Contact TheQuery for AI model data, benchmark context, educational resources, and corrections.",
  alternates: { canonical: `${SITE_URL}/press` },
};

export default function PressPage() {
  return <div className="max-w-[720px] mx-auto px-4 py-12 space-y-6 text-text-secondary leading-relaxed">
    <h1 className="font-serif text-3xl font-bold text-text-primary">For journalists</h1>
    <p>TheQuery is an independent AI education platform created and maintained by Addy. It publishes technical books, guides, glossary definitions, and analysis of AI models and systems.</p>
    <h2 className="font-serif text-xl font-semibold text-text-primary">Data and context</h2>
    <p>For questions about AI model specifications, benchmark interpretation, retrieval-augmented generation, and agentic systems, contact <a href="mailto:addy@thequery.in" className="text-accent underline">addy@thequery.in</a>. Include your publication, question, deadline, and time zone.</p>
    <p>Our <Link href="/research" className="text-accent underline">research and data page</Link> provides downloadable model data, provenance notes, and citation guidance. TheQuery compiles reported evaluations; it does not present them as its own laboratory results.</p>
    <h2 className="font-serif text-xl font-semibold text-text-primary">Attribution</h2>
    <p>Use “TheQuery” as the publication name and “Addy” as the published byline. Link to the specific resource supporting your claim. Attribute individual benchmark results to their original evaluator as well as the catalog.</p>
    <p>See our <Link href="/about#editorial-standards" className="text-accent underline">editorial standards</Link> for sourcing and corrections. Contact us for republication, translation, or image permissions.</p>
  </div>;
}
